use shakmaty::{CastlingSide, Chess, Position, Square};

use crate::{
    chess::{ChessModel, PosAuxiliarDim},
    flat_index,
};

pub fn encode_position(pos: &Chess) -> Vec<u16> {
    let mut out = vec![0; ChessModel::INPUT_SIZE.cols];
    let board = pos.board();
    let turn = pos.turn();

    for square in Square::ALL {
        if let Some(piece) = board.piece_at(square) {
            let piece_cell = ((piece.color == turn) as u16 * 6) + piece.role as u16;
            let piece_col = flat_index!(
                [ChessModel::INPUT_SIZE.x, ChessModel::INPUT_SIZE.y],
                [square.rank() as usize, square.file() as usize]
            );

            out[piece_col] = piece_cell;
        }
    }

    let meta_base = flat_index!(
        [ChessModel::INPUT_SIZE.x, ChessModel::INPUT_SIZE.y],
        [PosAuxiliarDim::AUXILIAR_X, 0]
    );
    let meta = &mut out[meta_base..(meta_base + ChessModel::INPUT_SIZE.y)];

    let friendly_castle_ks = pos.castles().has(turn, CastlingSide::KingSide) as u16;
    let friendly_castle_qs = pos.castles().has(turn, CastlingSide::QueenSide) as u16;

    let enemy_castle_ks = pos.castles().has(!turn, CastlingSide::KingSide) as u16;
    let enemy_castle_qs = pos.castles().has(!turn, CastlingSide::QueenSide) as u16;

    meta[PosAuxiliarDim::TURN_Y] = turn as u16;
    meta[PosAuxiliarDim::FRIENDLY_RIGHTS_Y] = (friendly_castle_qs << 1) | friendly_castle_ks;
    meta[PosAuxiliarDim::ENEMY_RIGHTS_Y] = (enemy_castle_qs << 1) | enemy_castle_ks;
    meta[PosAuxiliarDim::EP_FILE_Y] = pos
        .maybe_ep_square()
        .map(|sq| sq.file() as u16 + 1)
        .unwrap_or(0);
    meta[PosAuxiliarDim::HM_CLOCK_Y] =
        (pos.halfmoves() as usize * ChessModel::INPUT_SIZE.z / 150) as u16;

    out
}
