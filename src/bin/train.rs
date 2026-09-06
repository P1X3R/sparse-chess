use std::time::Instant;

use sparse_chess::{
    chess::{BottomLayerParams, ChessModel},
    sph::LayerParams,
    training_data::{lc0_to_csdr, load_records_from_bytes, read_chunk_file},
};

const BODY_LEN: usize = 2;

const BOTTOM_PARAMS: BottomLayerParams = BottomLayerParams {
    encoder_lr: 0.3,
    radius: 2,
    learning_radius: 2,
    choice: 0.001,
    vigilance: 0.9,
    active_ratio: 0.1,
    policy_lr: 0.03,
    policy_half_dendrites: 2,
    policy_scale: 8.0,
    value_lr: 0.03,
    value_half_dendrites: 2,
    value_scale: 8.0,
};

const PARAMS: [LayerParams; BODY_LEN] = [
    LayerParams {
        decoder_lr: 0.02,
        encoder_lr: 0.2,
        radius: 1,
        learning_radius: 1,
        choice: 0.001,
        vigilance: 0.9,
        active_ratio: 0.1,
        half_dendrites: 4,
        decoder_scale: 8.0,
    },
    LayerParams {
        decoder_lr: 0.01,
        encoder_lr: 0.1,
        radius: 1,
        learning_radius: 1,
        choice: 0.001,
        vigilance: 0.9,
        active_ratio: 0.1,
        half_dendrites: 6,
        decoder_scale: 8.0,
    },
];

const PIPELINE_SIZES: [(usize, usize, usize); BODY_LEN + 1] = [(8, 8, 32), (4, 4, 64), (2, 2, 128)];

const VALIDATE_EVERY: usize = 1000;
const LOG_EVERY: usize = 100;
const LAMBDA: f32 = 0.25;

struct ValSample {
    encoded: [u16; ChessModel::INPUT_SIZE.cols],
    legality_mask: [bool; 1858],
    policy_target: [f32; 1858],
    value_target: [f32; 3],
}

fn cross_entropy(pred: &[f32], target: &[f32]) -> f32 {
    debug_assert_eq!(pred.len(), target.len());
    pred.iter()
        .zip(target)
        .map(|(&p, &t)| -t * p.max(1e-7).ln())
        .sum()
}

fn main() -> std::io::Result<()> {
    fastrand::seed(42);

    let dir_path = "training-run2-test91-20251220-0017";
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

    // 1% split ratio
    let val_split_idx = ((gz_files.len() as f32) * 0.01).round() as usize;
    let (val_files, train_files) = gz_files.split_at(val_split_idx.max(1));

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

    let evaluate_validation = |model: &mut ChessModel, samples: &[ValSample]| {
        let mut total_policy_cce = 0.0;
        let mut total_value_loss = 0.0;
        let start = Instant::now();

        for sample in samples {
            let output = model.step(&sample.encoded, &sample.legality_mask, None);

            total_value_loss += cross_entropy(&output.value, &sample.value_target);
            total_policy_cce += cross_entropy(&output.policy, &sample.policy_target);
        }

        let duration = Instant::now() - start;

        if !samples.is_empty() {
            let n = samples.len() as f32;
            println!(
                "Validation Set Loss | Policy CCE: {:.4}, Value CCE: {:.4} | Throughput: {:.2?}/pos",
                total_policy_cce / n,
                total_value_loss / n,
                duration.div_f32(n)
            );
        }
    };

    evaluate_validation(&mut model, &val_buffer);

    let mut policy_cce = 0.0;
    let mut value_cce = 0.0;
    let mut window_train_positions = 0usize;
    let mut total_train_positions = 0usize;

    for (count, chunk_path) in train_files.iter().enumerate() {
        let bytes = read_chunk_file(&chunk_path)?;
        let game = load_records_from_bytes(&bytes);

        for training_pos in game {
            let target_value = training_pos.get_blended_wdl(LAMBDA);
            let target_policy = training_pos.probabilities.map(|p| p.max(0.0));

            lc0_to_csdr(training_pos, &mut encoded_buf);
            let legality_mask: [bool; 1858] = training_pos.probabilities.map(|p| p >= 0.0);

            let output = model.step(
                &encoded_buf,
                &legality_mask,
                Some((&target_policy, &target_value)),
            );

            policy_cce += cross_entropy(&output.policy, &target_policy);
            value_cce += cross_entropy(&output.value, &target_value);

            window_train_positions += 1;
            total_train_positions += 1;
        }

        if total_train_positions > 0 && count > 0 && count % LOG_EVERY == 0 {
            let n = window_train_positions as f32;
            println!(
                "{}. Train Set Loss | Policy CCE: {:.4}, Value CCE: {:.4}",
                count,
                policy_cce / n,
                value_cce / n
            );

            policy_cce = 0.0;
            value_cce = 0.0;
            window_train_positions = 0;

            model.save_to_file(model_path)?;
        }

        if count > 0 && count % VALIDATE_EVERY == 0 {
            print!("{}. ", count);
            evaluate_validation(&mut model, &val_buffer);
        }
    }

    model.save_to_file(model_path)?;
    Ok(())
}
