use std::time::Instant;

use sparse_chess::{
    chess::{BottomLayerParams, ChessModel},
    coder::column_wise_one_hot,
    sph::LayerParams,
    training_data::{lc0_to_csdr, load_records_from_bytes, read_chunk_file},
};

const BODY_LEN: usize = 2;

const BOTTOM_PARAMS: BottomLayerParams = BottomLayerParams {
    encoder_lr: 0.2,
    radius: 1,
    learning_radius: 1,
    choice: 1e-4,
    vigilance: 0.5,
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
        vigilance: 0.75,
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
        active_ratio: 0.2,
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
const TOTAL_GAMES: usize = 50000;

struct ValSample {
    encoded: [u16; ChessModel::INPUT_SIZE.cols],
    legality_mask: [bool; 1858],
    policy_target: [f32; 1858],
    value_target: [f32; 3],
}

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

fn evaluate_validation(model: &mut ChessModel, samples: &[ValSample]) {
    println!("---");

    let (bottom_rate, body_rates) = model.get_commited_rates();

    println!(
        "Committed rates (bottom layer -> body layers): {:.4} -> {:.4?}",
        bottom_rate, body_rates
    );

    let mut total_policy_cce = 0.0;
    let mut total_value_loss = 0.0;
    let mut std_dev = 0.0;
    let mut target_std_dev = 0.0;
    let mut top_1_correct = 0;
    let start = Instant::now();

    for sample in samples {
        let output = model.step(&sample.encoded, &sample.legality_mask, None);

        debug_assert!(
            output
                .policy
                .iter()
                .zip(sample.legality_mask)
                .filter(|&(_, is_legal)| !is_legal)
                .all(|(&p, _)| p == 0.0),
            "found policy head output where an illegal move's probability is greater than zero"
        );

        debug_assert!(
            output
                .policy
                .iter()
                .zip(sample.legality_mask)
                .filter(|&(_, is_legal)| is_legal)
                .all(|(&p, _)| p > 0.0),
            "found policy head output where a legal move's probability is less than or equal to zero"
        );

        let pred_best = column_wise_one_hot(&output.policy);
        let best = column_wise_one_hot(&sample.policy_target);

        if pred_best == best {
            top_1_correct += 1;
        }

        total_value_loss += cross_entropy(&output.value, &sample.value_target);
        total_policy_cce += cross_entropy(&output.policy, &sample.policy_target);

        std_dev += std_dev_sum(&output.value, 1.0 / 3.0);
        target_std_dev += std_dev_sum(&sample.value_target, 1.0 / 3.0);
    }

    let duration = Instant::now() - start;

    if !samples.is_empty() {
        let n = samples.len() as f32;
        let policy_loss = total_policy_cce / n;
        let value_loss = total_value_loss / n;

        println!(
            "Validation Set Loss | Policy CCE: {policy_loss:.4}, Value CCE: {value_loss:.4}, Top 1 accuracy: {:.2}%",
            (top_1_correct as f32 / n) * 100.0
        );

        println!(
            "Value Head Standard Deviation | Prediction: {:.4}, Target: {:.4}",
            (std_dev / n).sqrt(),
            (target_std_dev / n).sqrt()
        );

        println!("Throughput: {:.2?}/pos", duration.div_f32(n));
    }

    println!("---");
}

fn main() -> std::io::Result<()> {
    fastrand::seed(42);

    let dir_path = "training-run1-test80-20240401-0017";
    let model_path = "model.bin";

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

    let mut gz_files: Vec<_> = std::fs::read_dir(dir_path)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gz"))
        .collect();

    fastrand::shuffle(&mut gz_files);

    let mut encoded_buf = [0; ChessModel::INPUT_SIZE.cols];

    // 10% split ratio
    let val_split_idx = ((gz_files.len() as f32) * 0.1).round() as usize;
    let (total_val_files, train_files) = gz_files.split_at(val_split_idx.max(1));
    let val_files = fastrand::choose_multiple(total_val_files.iter(), VALIDATION_SAMPLES);

    println!("Buffering validation set into RAM...");
    let mut val_buffer: Vec<ValSample> = Vec::new();

    for val_path in val_files {
        let bytes = read_chunk_file(val_path)?;
        let game = load_records_from_bytes(&bytes);

        for pos in game {
            let mut encoded = [0; ChessModel::INPUT_SIZE.cols];
            lc0_to_csdr(&pos, &mut encoded);

            val_buffer.push(ValSample {
                encoded,
                legality_mask: pos.probabilities.map(|p| p >= 0.0),
                policy_target: pos.probabilities.map(|p| p.max(0.0)),
                value_target: pos.get_blended_wdl(LAMBDA),
            });
        }
    }
    println!("Loaded {} validation samples into RAM.", val_buffer.len());

    evaluate_validation(&mut model, &val_buffer);

    let mut window_policy_ce = 0.0;
    let mut total_policy_ce = 0.0;
    let mut window_value_ce = 0.0;
    let mut total_value_ce = 0.0;
    let mut window_train_pos = 0;
    let mut total_train_pos = 0;

    for (count, chunk_path) in train_files.iter().enumerate() {
        if count >= TOTAL_GAMES {
            evaluate_validation(&mut model, &val_buffer);
            break;
        }

        let new_lr = cosine_annealing(count as f32, TOTAL_GAMES as f32, MAX_LR, MIN_LR);
        model.policy_head.lr = new_lr;
        model.value_head.lr = new_lr;

        let bytes = read_chunk_file(&chunk_path)?;
        let game = load_records_from_bytes(&bytes).to_vec();

        for training_pos in &game {
            lc0_to_csdr(training_pos, &mut encoded_buf);

            let policy_target = training_pos.probabilities.map(|p| p.max(0.0));
            let legality_mask = training_pos.probabilities.map(|p| p >= 0.0);
            let value_target = training_pos.get_blended_wdl(LAMBDA);

            let output = model.step(
                &encoded_buf,
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

        if total_train_pos > 0 && count > 0 && count % LOG_EVERY == 0 {
            let total_pos = total_train_pos as f32;
            let window_pos = window_train_pos as f32;

            println!(
                "Train Set ({count} games) | Val CE: {:.4} (Win: {:.4}) | Pol CE: {:.4} (Win: {:.4}) | Value LR: {:.4} | Policy LR: {:.4}",
                total_value_ce / total_pos,
                window_value_ce / window_pos,
                total_policy_ce / total_pos,
                window_policy_ce / window_pos,
                model.value_head.lr,
                model.policy_head.lr,
            );

            window_value_ce = 0.0;
            window_policy_ce = 0.0;
            window_train_pos = 0;

            model.save_to_file(model_path)?;
        }

        if count > 0 && count % VALIDATE_EVERY == 0 {
            evaluate_validation(&mut model, &val_buffer);
        }
    }

    model.save_to_file(model_path)?;
    Ok(())
}
