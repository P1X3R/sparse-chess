use shakmaty::{CastlingSide, Chess, Color, Position, Square};

use crate::{
    chess::{ChessModel, PosAuxiliarDim},
    flat_index,
};

const NO_TRANSFORM: u8 = 0;
const FLIP_TRANSFORM: u8 = 1 << 0; // Horizontal flip (File A <-> File H)
const MIRROR_TRANSFORM: u8 = 1 << 1; // Vertical flip (Rank 1 <-> Rank 8)
const TRANSPOSE_TRANSFORM: u8 = 1 << 2; // Diagonal swap (x <-> y)

// Reverse bits inside each 8-bit byte independently (Horizontal flip / File swap)
fn reverse_bits_in_bytes(mut v: u64) -> u64 {
    v = ((v >> 1) & 0x5555_5555_5555_5555) | ((v & 0x5555_5555_5555_5555) << 1);
    v = ((v >> 2) & 0x3333_3333_3333_3333) | ((v & 0x3333_3333_3333_3333) << 2);
    v = ((v >> 4) & 0x0F0F_0F0F_0F0F_0F0F) | ((v & 0x0F0F_0F0F_0F0F_0F0F) << 4);
    v
}

// Reverse byte order in a 64-bit integer (Vertical flip / Rank swap)
fn reverse_bytes_in_bytes(v: u64) -> u64 {
    v.swap_bytes()
}

// Transpose 8x8 matrix encoded as u64
fn transpose_bits_in_bytes(v: u64) -> u64 {
    let mut t;
    let mut x = v;

    t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x0000_0000_FFFF_0000;
    x ^= t ^ (t << 28);

    x
}

fn compare_transposing(board_mask: u64, initial_transform: u8) -> i8 {
    let mut value = board_mask;
    if (initial_transform & FLIP_TRANSFORM) != 0 {
        value = reverse_bits_in_bytes(value);
    }
    if (initial_transform & MIRROR_TRANSFORM) != 0 {
        value = reverse_bytes_in_bytes(value);
    }
    let alternative = transpose_bits_in_bytes(value);

    if value < alternative {
        -1
    } else if value > alternative {
        1
    } else {
        0
    }
}

pub fn choose_transform(pos: &Chess) -> u8 {
    // Castling availability invalidates transformations
    if !pos.castles().is_empty() {
        return NO_TRANSFORM;
    }

    let turn = pos.turn();
    let board = pos.board();

    // Friendly pieces bitboards relative to perspective
    let ours = board.by_color(turn).0;
    let theirs = board.by_color(!turn).0;
    let kings = board.by_role(shakmaty::Role::King).0;
    let pawns = board.by_role(shakmaty::Role::Pawn).0;

    let mut our_king = kings & ours;

    let mut transform = NO_TRANSFORM;

    // Check if king is on the left half of the board
    if (our_king & 0x0F0F_0F0F_0F0F_0F0F) != 0 {
        transform |= FLIP_TRANSFORM;
        our_king = reverse_bits_in_bytes(our_king);
    }

    // Pawns prevent vertical mirroring or diagonal transposition
    if pawns != 0 {
        return transform;
    }

    // Check if king is on top half of the board
    if (our_king & 0xFFFF_FFFF_0000_0000) != 0 {
        transform |= MIRROR_TRANSFORM;
        our_king = reverse_bytes_in_bytes(our_king);
    }

    // Determine diagonal transposition
    if (our_king & 0xE0C0_8000) != 0 {
        transform |= TRANSPOSE_TRANSFORM;
    } else if (our_king & 0x1020_4080) != 0 {
        let bitboards_to_check = [
            ours | theirs,
            ours,
            kings,
            board.by_role(shakmaty::Role::Queen).0,
            board.by_role(shakmaty::Role::Rook).0,
            board.by_role(shakmaty::Role::Knight).0,
            board.by_role(shakmaty::Role::Bishop).0,
        ];

        for &mask in &bitboards_to_check {
            let outcome = compare_transposing(mask, transform);
            if outcome == -1 {
                return transform;
            }
            if outcome == 1 {
                return transform | TRANSPOSE_TRANSFORM;
            }
        }
    }

    transform
}

pub fn encode_position(pos: &Chess) -> Vec<u16> {
    let mut out = vec![0; ChessModel::INPUT_SIZE.cols];
    let board = pos.board();
    let turn = pos.turn();

    // Determine the position transformation mask
    let transform = choose_transform(pos);

    for square in Square::ALL {
        if let Some(piece) = board.piece_at(square) {
            let piece_cell = ((piece.color == turn) as u16 * 6) + piece.role as u16;

            // Perspective-relative coordinates (0..=7)
            let (mut rank, mut file) = match turn {
                Color::White => (square.rank() as usize, square.file() as usize),
                Color::Black => (7 - square.rank() as usize, 7 - square.file() as usize),
            };

            // Apply calculated Lc0 transformations geometrically
            if (transform & FLIP_TRANSFORM) != 0 {
                file = 7 - file;
            }
            if (transform & MIRROR_TRANSFORM) != 0 {
                rank = 7 - rank;
            }
            if (transform & TRANSPOSE_TRANSFORM) != 0 {
                std::mem::swap(&mut rank, &mut file);
            }

            let piece_col = flat_index!(
                [ChessModel::INPUT_SIZE.x, ChessModel::INPUT_SIZE.y],
                [rank, file]
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

    // En-passant file transformation adjustment
    let ep_file = pos
        .maybe_ep_square()
        .map(|sq| {
            let mut file = match turn {
                Color::White => sq.file() as u16,
                Color::Black => 7 - sq.file() as u16,
            };
            if (transform & FLIP_TRANSFORM) != 0 {
                file = 7 - file;
            }
            file + 1
        })
        .unwrap_or(0);

    meta[PosAuxiliarDim::TURN_Y] = turn as u16;
    meta[PosAuxiliarDim::FRIENDLY_RIGHTS_Y] = (friendly_castle_qs << 1) | friendly_castle_ks;
    meta[PosAuxiliarDim::ENEMY_RIGHTS_Y] = (enemy_castle_qs << 1) | enemy_castle_ks;
    meta[PosAuxiliarDim::EP_FILE_Y] = ep_file;
    meta[PosAuxiliarDim::HM_CLOCK_Y] =
        (pos.halfmoves() as usize * ChessModel::INPUT_SIZE.z / 150) as u16;

    out
}
