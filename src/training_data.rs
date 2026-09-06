use std::{fs::File, io::Read, path::Path};

use flate2::read::GzDecoder;
use crate::{
    chess::{ChessModel, PosAuxiliarDim},
    flat_index,
};
use zerocopy::FromBytes;
use zerocopy_derive::*;

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

pub fn lc0_to_csdr(data: &TrainingData, out: &mut [u16; ChessModel::INPUT_SIZE.cols]) {
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
