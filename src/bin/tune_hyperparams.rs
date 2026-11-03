use std::env;
use std::path::PathBuf;

use optimizer::prelude::*;
use sparse_chess::{
    chess::{BottomLayerParams, ChessModel},
    hyperparams::*,
    sph::LayerParams,
    training_data::{lc0_to_csdr, load_records_from_bytes, read_chunk_file, read_dataset_dir},
};

const SAMPLES: usize = 200;

#[inline]
fn cross_entropy(pred: &[f32], target: &[f32]) -> f32 {
    debug_assert_eq!(pred.len(), target.len());
    pred.iter()
        .zip(target)
        .map(|(&p, &t)| -t * p.max(1e-7).ln())
        .sum()
}

#[inline]
fn get_field_cnt(r: isize) -> isize {
    let diameter = r * 2 + 1;
    diameter * diameter - 1
}

#[inline]
fn get_active_ratio(allowed: i64, r: isize) -> f32 {
    let field_cnt = get_field_cnt(r) as f32;
    (allowed as f32 + 0.5).clamp(0.0, field_cnt) / field_cnt
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let dataset_path = match args.get(1) {
        Some(p) => PathBuf::from(p),
        None => panic!("Usage: {} <dataset path>", args[0]),
    };
    println!("Using dataset path: {}", dataset_path.display());

    let val_files = read_dataset_dir(&dataset_path.to_string_lossy());
    let mut rng = fastrand::Rng::with_seed(42);
    let sample_files = rng.choose_multiple(val_files.iter(), SAMPLES);

    let bottom_p_complement = FloatParam::new(0.02, 0.25)
        .name("bottom_p_complement")
        .log_scale();
    let bottom_allowed = IntParam::new(0, 8).name("bottom_allowed");

    let body_hypers: Vec<(FloatParam, IntParam)> = (0..BODY_LEN)
        .map(|idx| {
            (
                FloatParam::new(0.008, 0.2)
                    .name(format!("layer{}_p_complement", idx))
                    .log_scale(),
                IntParam::new(0, 8).name(format!("layer{}_allowed", idx)),
            )
        })
        .collect();

    let sampler = TpeSampler::builder().seed(123).build().unwrap();
    let study: Study<f64> = Study::with_sampler(Direction::Minimize, sampler);

    println!("Starting Hyperparameter Optimization for vigilance and active_ratio...");

    study.optimize(50, |trial: &mut optimizer::Trial| {
        let bottom_p_complement = bottom_p_complement.suggest(trial)?;
        let bottom_allowed = bottom_allowed.suggest(trial)?;
        let bottom_params = BottomLayerParams {
            vigilance: 1.0 - bottom_p_complement as f32,
            active_ratio: get_active_ratio(bottom_allowed, BOTTOM_PARAMS.learning_radius),
            ..BOTTOM_PARAMS
        };

        let body_params: Vec<LayerParams> = PARAMS
            .iter()
            .zip(&body_hypers)
            .map(|(&p, (p_complement, allowed))| {
                let p_complement = p_complement.suggest(trial).unwrap();
                let allowed = allowed.suggest(trial).unwrap();
                LayerParams {
                    vigilance: 1.0 - p_complement as f32,
                    active_ratio: get_active_ratio(allowed, p.learning_radius),
                    ..p
                }
            })
            .collect();

        fastrand::seed(123);
        let mut model = ChessModel::new(&PIPELINE_SIZES, &body_params, &bottom_params);

        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut num_steps = 0u64;
        let mut planes = [0u16; ChessModel::PLANES_SIZE.cols];
        let mut aux = [0u16; ChessModel::AUXILIARY_SIZE.cols];

        for file in &sample_files {
            if let Ok(bytes) = read_chunk_file(file) {
                let game_buf = load_records_from_bytes(&bytes);
                for pos in game_buf {
                    lc0_to_csdr(&pos, &mut planes, &mut aux);
                    let policy_target = pos.get_policy_target();
                    let value_target = pos.get_blended_wdl(LAMBDA);
                    let legality_mask = pos.get_legality_mask();

                    let output = model.step(
                        &[&planes, &aux],
                        &legality_mask,
                        Some((&policy_target, &value_target)),
                    );

                    total_value_loss += cross_entropy(&output.value, &value_target);
                    total_policy_loss += cross_entropy(&output.policy, &policy_target);

                    num_steps += 1;
                }
            }
        }

        let absolute_loss = (total_value_loss + total_policy_loss) / num_steps as f32;

        print!(
            "[Trial {}] Loss: {:.4} | bv: {:.3}, bar: {:.3}",
            trial.id(),
            absolute_loss,
            bottom_params.vigilance,
            bottom_params.active_ratio,
        );

        for (idx, p) in body_params.iter().enumerate() {
            let l_str = format!("l{}", idx);
            print!(
                " | {}v: {:.3}, {}ar: {:.3}",
                l_str, p.vigilance, l_str, p.active_ratio
            );
        }

        println!();

        Ok::<_, Error>(absolute_loss as f64)
    })?;

    let best = study.best_trial()?;
    println!("Optimization finished!");
    println!("Best Hyperparameters Configuration:");

    println!(
        "bottom vigilange = {:.4}",
        1.0 - best.get(&bottom_p_complement).unwrap()
    );
    println!(
        "bottom active_ratio = {:.4}",
        get_active_ratio(
            best.get(&bottom_allowed).unwrap(),
            BOTTOM_PARAMS.learning_radius
        )
    );

    for (idx, (p, (p_complement, allowed))) in PARAMS.iter().zip(&body_hypers).enumerate() {
        let l_str = format!("layer{}", idx);

        println!(
            "{} vigilange = {:.4}",
            l_str,
            1.0 - best.get(p_complement).unwrap()
        );
        println!(
            "{} active_ratio = {:.4}",
            l_str,
            get_active_ratio(best.get(allowed).unwrap(), p.learning_radius)
        );
    }

    println!("Best Validation Loss: {:.2}", best.value);

    Ok(())
}
