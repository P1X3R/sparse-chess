use std::{fs::File, io::Read};

use flate2::read::GzDecoder;
use sparse_chess::{
    chess::{BottomLayerParams, ChessModel, PosAuxiliarDim},
    coder::column_wise_one_hot,
    flat_index,
    sph::LayerParams,
};
use zerocopy::FromBytes;
use zerocopy_derive::*;

const BODY_LEN: usize = 2;

const BOTTOM_PARAMS: BottomLayerParams = BottomLayerParams {
    encoder_lr: 0.2,
    radius: 2,
    learning_radius: 2,
    choice: 0.01,
    vigilance: 0.9,
    active_ratio: 0.1,
    policy_lr: 0.02,
    policy_half_dendrites: 2,
    policy_scale: 8.0,
    value_lr: 0.02,
    value_half_dendrites: 2,
    value_scale: 8.0,
};

const PARAMS: [LayerParams; BODY_LEN] = [
    LayerParams {
        decoder_lr: 0.01,
        encoder_lr: 0.1,
        radius: 1,
        learning_radius: 1,
        choice: 0.01,
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
        choice: 0.01,
        vigilance: 0.9,
        active_ratio: 0.1,
        half_dendrites: 6,
        decoder_scale: 8.0,
    },
];

const PIPELINE_SIZES: [(usize, usize, usize); BODY_LEN + 1] = [(8, 8, 12), (4, 4, 18), (2, 2, 27)];

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
    pub fn get_win(&self) -> f32 {
        (1.0 + self.root_q - self.root_d) * 0.5
    }

    #[inline]
    pub fn get_lose(&self) -> f32 {
        (1.0 - self.root_q - self.root_d) * 0.5
    }

    #[inline]
    pub fn get_wdl(&self) -> [f32; 3] {
        [
            self.get_win(),
            self.root_d, // Draw
            self.get_lose(),
        ]
    }
}

const _: () = assert!(TrainingData::BYTES == 8356);

pub fn load_records_from_bytes(bytes: &[u8]) -> &[TrainingData] {
    let (records, _remainder) =
        <[TrainingData]>::ref_from_prefix(bytes).expect("Byte stream alignment or size issue");

    records
}

pub fn read_chunk_file(path: &str) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut buffer = Vec::new();

    let mut decoder = GzDecoder::new(file);
    decoder.read_to_end(&mut buffer)?;

    Ok(buffer)
}

fn lc0_to_csdr(data: &TrainingData) -> Vec<u16> {
    let mut out = vec![0; ChessModel::INPUT_SIZE.cols];

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

    // Turn: Stored in bit 7 of invariance_info for input type 3
    let turn = (data.invariance_info >> 7) & 1;

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

    meta[PosAuxiliarDim::TURN_Y] = turn as u16;
    meta[PosAuxiliarDim::FRIENDLY_RIGHTS_Y] = (friendly_castle_qs << 1) | friendly_castle_ks;
    meta[PosAuxiliarDim::ENEMY_RIGHTS_Y] = (enemy_castle_qs << 1) | enemy_castle_ks;
    meta[PosAuxiliarDim::EP_FILE_Y] = ep_file;
    meta[PosAuxiliarDim::HM_CLOCK_Y] =
        (data.rule50_count as usize * ChessModel::INPUT_SIZE.z / 150) as u16;

    out
}

fn main() -> std::io::Result<()> {
    let chunk_path = "training-run1--20210605-0516/training.221043087.gz";
    let model_path = "model.bin";

    let bytes = read_chunk_file(chunk_path)?;
    let game = load_records_from_bytes(&bytes);
    let mut model = match ChessModel::load_from_file(model_path) {
        Ok(model) => model,
        Err(err) => {
            println!("error loading model file: {}", err);
            println!("initializing new model instead");

            ChessModel::new(&PIPELINE_SIZES, &PARAMS, &BOTTOM_PARAMS)
        }
    };

    let mut policy_cce = 0.0;
    let mut value_loss = 0.0;

    for training_pos in game {
        let target_value = training_pos.get_wdl();
        let target_policy = training_pos
            .probabilities
            .map(|p| if p == -1.0 { 0.0 } else { p });
        let target_value_idx = column_wise_one_hot(&target_value) as usize;
        let encoded = lc0_to_csdr(training_pos);
        let legality_mask: [bool; 1858] = training_pos.probabilities.map(|p| p != -1.0);
        let output = model.step(&encoded, &legality_mask, None);

        policy_cce += -output.policy[training_pos.played_idx as usize].ln();
        value_loss += -output.value[target_value_idx].ln();
    }

    let num_pos = game.len() as f32;
    println!(
        "Policy CCE: {:.2}, Value CCE: {:.2}",
        policy_cce / num_pos,
        value_loss / num_pos
    );

    Ok(())
}
