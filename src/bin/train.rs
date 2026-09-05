use std::{fs::File, io::Read, path::Path, time::Instant};

use flate2::read::GzDecoder;
use sparse_chess::{
    chess::{BottomLayerParams, ChessModel, PosAuxiliarDim},
    flat_index,
    sph::LayerParams,
};
use zerocopy::FromBytes;
use zerocopy_derive::*;

const BODY_LEN: usize = 2;

const BOTTOM_PARAMS: BottomLayerParams = BottomLayerParams {
    encoder_lr: 0.25,
    radius: 2,
    learning_radius: 2,
    choice: 0.001,
    vigilance: 0.9,
    active_ratio: 0.1,
    policy_lr: 0.025,
    policy_half_dendrites: 2,
    policy_scale: 8.0,
    value_lr: 0.025,
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

#[repr(C, packed)]
#[derive(FromBytes, Immutable, KnownLayout, Debug, Clone, Copy)]
pub struct TrainingData {
    pub version: u32,
    pub input_format: u32,
    pub probabilities: [f32; 1858],
    pub planes: [u64; 104],
    pub castling_us_ooo: u8,
    pub castling_us_oo: u8,
    pub castling_them_ooo: u8,
    pub castling_them_oo: u8,
    pub side_to_move_or_enpassant: u8,
    pub rule50_count: u8,
    pub invariance_info: u8,
    pub dummy: u8,
    pub root_q: f32,
    pub best_q: f32,
    pub root_d: f32,
    pub best_d: f32,
    pub root_m: f32,
    pub best_m: f32,
    pub plies_left: f32,
    pub result_q: f32,
    pub result_d: f32,
    pub played_q: f32,
    pub played_d: f32,
    pub played_m: f32,
    pub orig_q: f32,
    pub orig_d: f32,
    pub orig_m: f32,
    pub visits: u32,
    pub played_idx: u16,
    pub best_idx: u16,
    pub policy_kld: f32,
    pub q_st: f32,
}

impl TrainingData {
    const BYTES: usize = core::mem::size_of::<TrainingData>();

    #[inline]
    fn q_d_to_win(q: f32, d: f32) -> f32 {
        (1.0 + q - d) * 0.5
    }

    #[inline]
    pub fn get_wdl(&self) -> [f32; 3] {
        [
            Self::q_d_to_win(self.root_q, self.root_d),
            self.root_d, // Draw
            Self::q_d_to_win(-self.root_q, self.root_d),
        ]
    }

    pub fn get_blended_wdl(&self, lambda: f32) -> [f32; 3] {
        let w_search = Self::q_d_to_win(self.root_q, self.root_d);
        let d_search = self.root_d;
        let l_search = Self::q_d_to_win(-self.root_q, self.root_d);

        let w_result = Self::q_d_to_win(self.result_q, self.result_d);
        let d_result = self.result_d;
        let l_result = Self::q_d_to_win(-self.result_q, self.result_d);

        [
            (1.0 - lambda) * w_search + lambda * w_result,
            (1.0 - lambda) * d_search + lambda * d_result,
            (1.0 - lambda) * l_search + lambda * l_result,
        ]
    }

    #[inline]
    pub fn get_pos_result(&self) -> usize {
        if self.result_q > 0.5 {
            0 // Win
        } else if self.result_q < -0.5 {
            2 // Loss
        } else {
            1 // Draw
        }
    }
}

const _: () = assert!(TrainingData::BYTES == 8356);

pub fn load_records_from_bytes(bytes: &[u8]) -> &[TrainingData] {
    let (records, _remainder) =
        <[TrainingData]>::ref_from_prefix(bytes).expect("Byte stream alignment or size issue");

    records
}

pub fn read_chunk_file<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut buffer = Vec::new();

    let mut decoder = GzDecoder::new(file);
    decoder.read_to_end(&mut buffer)?;

    Ok(buffer)
}

fn lc0_to_csdr(data: &TrainingData, out: &mut [u16; ChessModel::INPUT_SIZE.cols]) {
    out.fill(0);

    // Lc0 stores 8 history states, 13 planes each (12 pieces + 1 repetition).
    // The current position is at t=0, so we only need planes 0..11.
    // Map Lc0 plane indices to your `piece_cell` format: ((is_us * 6) + role)
    // Us: P=7, N=8, B=9, R=10, Q=11, K=12
    // Them: P=1, N=2, B=3, R=4, Q=5, K=6
    const PLANE_TO_CELL: [u16; 12] = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6];

    // 1. Map piece bitboards to the 8x8 grid
    for sq in 0..64 {
        let bit = 1u64 << sq;

        // Since Lc0's planes are already canonically transformed,
        // we map the bit directly to the coordinates.
        let rank = sq / 8;
        let file = sq % 8;

        for (p, &piece_cell) in PLANE_TO_CELL.iter().enumerate() {
            if (data.planes[p] & bit) != 0 {
                let piece_col = flat_index!(
                    [ChessModel::INPUT_SIZE.x, ChessModel::INPUT_SIZE.y],
                    [rank, file]
                );
                out[piece_col] = piece_cell;
                break; // A square can only have one piece
            }
        }
    }

    // 2. Map Auxiliary Metadata
    let meta_base = flat_index!(
        [ChessModel::INPUT_SIZE.x, ChessModel::INPUT_SIZE.y],
        [PosAuxiliarDim::AUXILIAR_X, 0]
    );
    let meta = &mut out[meta_base..(meta_base + ChessModel::INPUT_SIZE.y)];

    // Castling rights (1 or 0 in V6 byte fields)
    let friendly_castle_ks = (data.castling_us_oo != 0) as u16;
    let friendly_castle_qs = (data.castling_us_ooo != 0) as u16;
    let enemy_castle_ks = (data.castling_them_oo != 0) as u16;
    let enemy_castle_qs = (data.castling_them_ooo != 0) as u16;

    // EP File: side_to_move_or_enpassant acts as a column mask
    // We mask out bit 7 just in case it leaks side-to-move info
    let ep_mask = data.side_to_move_or_enpassant & 0x7F;
    let ep_file = if ep_mask != 0 {
        (ep_mask.trailing_zeros() + 1) as u16
    } else {
        0
    };

    meta[PosAuxiliarDim::FRIENDLY_RIGHTS_Y] = (friendly_castle_qs << 1) | friendly_castle_ks;
    meta[PosAuxiliarDim::ENEMY_RIGHTS_Y] = (enemy_castle_qs << 1) | enemy_castle_ks;
    meta[PosAuxiliarDim::EP_FILE_Y] = ep_file;
    meta[PosAuxiliarDim::HM_CLOCK_Y] =
        (data.rule50_count as usize * ChessModel::INPUT_SIZE.z / 150) as u16;
}

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
