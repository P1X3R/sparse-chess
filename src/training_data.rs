use std::{fs::File, io::Read, path::Path};

use crate::{
    chess::{ChessModel, PosAuxiliarDim},
    flat_index,
};
use flate2::read::GzDecoder;
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
    pub fn get_wdl(q: f32, d: f32) -> [f32; 3] {
        [
            Self::q_d_to_win(q, d),
            d, // Draw
            Self::q_d_to_win(-q, d),
        ]
    }

    #[inline]
    pub fn get_blended_wdl(&self, lambda: f32) -> [f32; 3] {
        let [w_search, d_search, l_search] = Self::get_wdl(self.root_q, self.root_d);
        let [w_result, d_result, l_result] = Self::get_wdl(self.result_q, self.result_d);

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

pub fn lc0_to_csdr(data: &TrainingData, out: &mut [&mut [u16]; 2]) {
    let (left, right) = out.split_at_mut(1);
    let planes = &mut left[0];
    let aux = &mut right[0];

    assert_eq!(planes.len(), ChessModel::PLANES_SIZE.cols);
    assert_eq!(aux.len(), ChessModel::AUXILIARY_SIZE.cols);

    // Map Lc0 planes to piece cells:
    // Us:   P=7, N=8, B=9, R=10, Q=11, K=12  (Planes 0..5)
    // Them: P=1, N=2, B=3, R=4, Q=5,  K=6   (Planes 6..11)
    const PLANE_TO_CELL: [u16; 12] = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6];

    planes.fill(0);
    for sq in 0..64 {
        let bit = 1u64 << sq;

        for (p, &piece_cell) in PLANE_TO_CELL.iter().enumerate() {
            if (data.planes[p] & bit) != 0 {
                let rank = sq / 8;
                let file = sq % 8;

                let piece_col = flat_index!(
                    [ChessModel::INPUT_SIZE.x, ChessModel::PLANES_SIZE.y],
                    [rank, file]
                );
                planes[piece_col] = piece_cell;
                break;
            }
        }
    }

    let friendly_castle_ks = (data.castling_us_oo != 0) as u16;
    let friendly_castle_qs = (data.castling_us_ooo != 0) as u16;
    let enemy_castle_ks = (data.castling_them_oo != 0) as u16;
    let enemy_castle_qs = (data.castling_them_ooo != 0) as u16;

    let turn = (data.side_to_move_or_enpassant & 1) as u16;

    aux[PosAuxiliarDim::RIGHTS_US_QS_Y] = friendly_castle_qs;
    aux[PosAuxiliarDim::RIGHTS_US_KS_Y] = friendly_castle_ks;
    aux[PosAuxiliarDim::RIGHTS_THEM_QS_Y] = enemy_castle_qs;
    aux[PosAuxiliarDim::RIGHTS_THEM_KS_Y] = enemy_castle_ks;
    aux[PosAuxiliarDim::TURN_Y] = turn;
    aux[PosAuxiliarDim::HM_CLOCK_Y] =
        (data.rule50_count as usize * ChessModel::PLANES_SIZE.z / 150) as u16;
}
