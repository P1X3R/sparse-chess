use std::path::PathBuf;

use fastrand::Rng;
use sparse_chess::{
    chess::{BottomLayerParams, ChessModel},
    coder::column_wise_one_hot,
    sph::LayerParams,
    training_data::{lc0_to_csdr, load_records_from_bytes, read_chunk_file, read_dataset_dir},
};

use mlflow_client::Mlflow;

const BODY_LEN: usize = 2;

const BOTTOM_PARAMS: BottomLayerParams = BottomLayerParams {
    encoder_lr: 0.2,
    radius: 1,
    learning_radius: 1,
    choice: 1e-4,
    vigilance: 0.0833,
    active_ratio: 0.2,
    policy_lr: 0.08,
    policy_half_dendrites: 2,
    policy_scale: 1.0,
    value_lr: 0.08,
    value_half_dendrites: 6,
    value_scale: 1.0,
};

const PARAMS: [LayerParams; BODY_LEN] = [
    LayerParams {
        decoder_lr: 0.02,
        encoder_lr: 0.1,
        radius: 1,
        learning_radius: 1,
        choice: 1e-4,
        vigilance: 0.866,
        active_ratio: 0.2,
        half_dendrites: 2,
        decoder_scale: 8.0,
    },
    LayerParams {
        decoder_lr: 0.02,
        encoder_lr: 0.1,
        radius: 1,
        learning_radius: 1,
        choice: 1e-4,
        vigilance: 0.9,
        active_ratio: 0.1,
        half_dendrites: 3,
        decoder_scale: 8.0,
    },
];

const PIPELINE_SIZES: [(usize, usize, usize); BODY_LEN + 1] =
    [(8, 8, 128), (8, 8, 128), (8, 8, 128)];

const VALIDATION_SAMPLES: usize = 200;
const VALIDATE_EVERY: usize = 1000;
const LOG_EVERY: usize = 100;
const LAMBDA: f32 = 0.25;

const MIN_LR: f32 = 0.001;
const MAX_LR: f32 = 0.06;

#[inline]
fn cosine_annealing(games: f32, total_games: f32, max_lr: f32, min_lr: f32) -> f32 {
    let progress = (std::f32::consts::PI * games / total_games).cos();
    min_lr + 0.5 * (max_lr - min_lr) * (1.0 + progress)
}

#[inline]
fn cross_entropy(pred: &[f32], target: &[f32]) -> f32 {
    debug_assert_eq!(pred.len(), target.len());
    pred.iter()
        .zip(target)
        .map(|(&p, &t)| -t * p.max(1e-7).ln())
        .sum()
}

#[inline]
fn std_dev_sum(x: &[f32], mean: f32) -> f32 {
    x.iter()
        .map(|&v| {
            let diff = v as f32 - mean;
            diff * diff
        })
        .sum::<f32>()
}

fn evaluate_validation(
    rng: &mut Rng,
    run: &mut mlflow_client::MlflowRunWriter,
    step: usize,
    model: &mut ChessModel,
    val_files: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    let sample_files = rng.choose_multiple(val_files.iter(), VALIDATION_SAMPLES);

    let commitment_rates = model.get_commited_rates();

    for (i, &rate) in commitment_rates.iter().enumerate() {
        run.log_metric(
            &format!("val/layer{}_commitment", i),
            rate as f64,
            Some(step as i64),
        )?;
    }

    let mut num_steps = 0;

    let mut total_policy_cce = 0.0;
    let mut total_value_loss = 0.0;
    let mut std_dev = 0.0;
    let mut top_1_correct = 0;

    let mut planes = [0u16; ChessModel::PLANES_SIZE.cols];
    let mut aux = [0u16; ChessModel::AUXILIARY_SIZE.cols];

    for file in sample_files {
        let bytes = read_chunk_file(&file).unwrap();
        let game_buf = load_records_from_bytes(&bytes);

        for pos in game_buf {
            lc0_to_csdr(&pos, &mut planes, &mut aux);

            let value_target = pos.get_blended_wdl(LAMBDA);
            let policy_target = pos.get_policy_target();
            let legality_mask = pos.get_legality_mask();

            let output = model.step(&[&planes, &aux], &legality_mask, None);

            debug_assert!(
                output
                    .policy
                    .iter()
                    .zip(legality_mask)
                    .filter(|&(_, is_legal)| !is_legal)
                    .all(|(&p, _)| p == 0.0),
                "found policy head output where an illegal move's probability is greater than zero"
            );

            debug_assert!(
                output
                    .policy
                    .iter()
                    .zip(legality_mask)
                    .filter(|&(_, is_legal)| is_legal)
                    .all(|(&p, _)| p > 0.0),
                "found policy head output where a legal move's probability is less than or equal to zero"
            );

            let pred_best = column_wise_one_hot(&output.policy);
            let best = column_wise_one_hot(&policy_target);

            if pred_best == best {
                top_1_correct += 1;
            }

            total_value_loss += cross_entropy(&output.value, &value_target);
            total_policy_cce += cross_entropy(&output.policy, &policy_target);

            std_dev += std_dev_sum(&output.value, 1.0 / 3.0);
        }

        num_steps += game_buf.len();
    }

    if num_steps > 0 {
        let n = num_steps as f32;
        let policy_loss = total_policy_cce / n;
        let value_loss = total_value_loss / n;
        let top1_acc = (top_1_correct as f32 / n) * 100.0;
        let val_std_dev = (std_dev / n).sqrt();

        println!(
            " -> [Val @ Step {:>8}] Policy CCE: {:.4} | Value CCE: {:.4} | Top-1 Acc: {:.2}%",
            step, policy_loss, value_loss, top1_acc
        );

        run.log_metric("val/value_cce", value_loss as f64, Some(step as i64))?;
        run.log_metric("val/policy_cce", policy_loss as f64, Some(step as i64))?;
        run.log_metric("val/top_1_accuracy", top1_acc as f64, Some(step as i64))?;
        run.log_metric("val/value_std_dev", val_std_dev as f64, Some(step as i64))?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    fastrand::seed(123);
    let mut dataset_rng = fastrand::Rng::with_seed(1234);

    let train_dataset_path = "training-dataset/train";
    let val_dataset_path = "training-dataset/val";
    let model_path = "model.bin";

    let train_files = read_dataset_dir(train_dataset_path);
    let val_files = read_dataset_dir(val_dataset_path);
    let total_games = train_files.len();

    let mlflow = Mlflow::new("http://localhost:5000")?;
    let experiment = mlflow.create_experiment_if_not_exists("sparse-chess", Default::default())?;
    let mut run = experiment.start_run("sph")?;

    run.log_params("bottom_params", BOTTOM_PARAMS)?;
    for (i, layer) in PARAMS.iter().enumerate() {
        run.log_params(&format!("body_params/layer_{i}"), layer)?;
    }
    run.log_params("total_games", total_games)?;
    run.log_params("max_lr", MAX_LR)?;
    run.log_params("min_lr", MIN_LR)?;
    run.log_params("lambda", LAMBDA)?;

    let mut model = match ChessModel::load_from_file(model_path) {
        Ok(model) => {
            println!("loaded model from {}", model_path);
            model
        }
        Err(err) => {
            println!("error loading model file: {}, initializing new model", err);
            ChessModel::new(&PIPELINE_SIZES, &PARAMS, &BOTTOM_PARAMS)
        }
    };

    let mut planes = [0u16; ChessModel::PLANES_SIZE.cols];
    let mut aux = [0u16; ChessModel::AUXILIARY_SIZE.cols];

    evaluate_validation(&mut dataset_rng, &mut run, 0, &mut model, &val_files)?;

    let mut window_policy_ce = 0.0;
    let mut total_policy_ce = 0.0;
    let mut window_value_ce = 0.0;
    let mut total_value_ce = 0.0;
    let mut window_train_pos = 0;
    let mut total_train_pos = 0;

    for (game_idx, file) in train_files.iter().enumerate() {
        let bytes = read_chunk_file(file).unwrap();
        let game_buf = load_records_from_bytes(&bytes);

        let new_lr = cosine_annealing(game_idx as f32, total_games as f32, MAX_LR, MIN_LR);
        model.policy_head.lr = new_lr;
        model.value_head.lr = new_lr;

        for pos in game_buf {
            lc0_to_csdr(pos, &mut planes, &mut aux);

            let policy_target = pos.get_policy_target();
            let legality_mask = pos.get_legality_mask();
            let value_target = pos.get_blended_wdl(LAMBDA);

            let output = model.step(
                &[&planes, &aux],
                &legality_mask,
                Some((&policy_target, &value_target)),
            );

            let value_loss = cross_entropy(&output.value, &value_target);
            window_value_ce += value_loss;
            total_value_ce += value_loss;

            let policy_loss = cross_entropy(&output.policy, &policy_target);
            window_policy_ce += policy_loss;
            total_policy_ce += policy_loss;

            total_train_pos += 1;
            window_train_pos += 1;
        }

        model.body.clean_learning_state();

        if total_train_pos > 0 && game_idx > 0 && game_idx % LOG_EVERY == 0 {
            let total_pos = total_train_pos as f32;
            let window_pos = window_train_pos as f32;

            let cumulative_val_loss = total_value_ce / total_pos;
            let window_val_loss = window_value_ce / window_pos;
            let cumulative_pol_loss = total_policy_ce / total_pos;
            let window_pol_loss = window_policy_ce / window_pos;

            println!(
                "[Train] Game {:>6}/{} | Pos: {:>8} | Val Loss: {:.4} | Pol Loss: {:.4} | LR: {:.6}",
                game_idx, total_games, total_train_pos, window_val_loss, window_pol_loss, new_lr
            );

            run.log_metric(
                "train/window_value_cce",
                window_val_loss as f64,
                Some(total_train_pos as i64),
            )?;
            run.log_metric(
                "train/window_policy_cce",
                window_pol_loss as f64,
                Some(total_train_pos as i64),
            )?;
            run.log_metric(
                "train/total_value_cce",
                cumulative_val_loss as f64,
                Some(total_train_pos as i64),
            )?;
            run.log_metric(
                "train/total_policy_cce",
                cumulative_pol_loss as f64,
                Some(total_train_pos as i64),
            )?;
            run.log_metric(
                "train/heads_lr",
                new_lr as f64,
                Some(total_train_pos as i64),
            )?;

            window_value_ce = 0.0;
            window_policy_ce = 0.0;
            window_train_pos = 0;

            model.save_to_file(model_path)?;
        }

        if game_idx > 0 && game_idx % VALIDATE_EVERY == 0 {
            evaluate_validation(
                &mut dataset_rng,
                &mut run,
                total_train_pos,
                &mut model,
                &val_files,
            )?;
        }
    }

    model.save_to_file(model_path)?;

    run.finish()?;
    Ok(())
}
