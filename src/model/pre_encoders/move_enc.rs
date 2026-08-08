use shakmaty::*;
use std::{collections::HashMap, sync::LazyLock};

const DIRECTIONS: [(i8, i8); 8] = [
    (1, 0),   // N
    (1, 1),   // NE
    (0, 1),   // E
    (-1, 1),  // SE
    (-1, 0),  // S
    (-1, -1), // SW
    (0, -1),  // W
    (1, -1),  // NW
];

const KNIGHT_MOVES: [(i8, i8); 8] = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Lc0MoveKey {
    from_sq: u8,
    to_sq: u8,
    promotion: Option<Role>,
}

static LC0_MOVES: LazyLock<Vec<Lc0MoveKey>> = LazyLock::new(|| {
    let mut moves = Vec::with_capacity(1858);

    for from_sq in Square::ALL {
        let from_r = from_sq.rank() as i8;
        let from_c = from_sq.file() as i8;
        let from_u8 = u8::from(from_sq);

        // 1. Queen-like moves (Planes 0..55: 8 directions * 7 steps)
        for &(dr, dc) in &DIRECTIONS {
            for step in 1..=7 {
                let to_r = from_r + dr * step;
                let to_c = from_c + dc * step;

                if (0..8).contains(&to_r) && (0..8).contains(&to_c) {
                    let to_sq = (to_r * 8 + to_c) as u8;
                    moves.push(Lc0MoveKey {
                        from_sq: from_u8,
                        to_sq,
                        promotion: None,
                    });
                }
            }
        }

        // 2. Knight moves (Planes 56..63)
        for &(dr, dc) in &KNIGHT_MOVES {
            let to_r = from_r + dr;
            let to_c = from_c + dc;

            if (0..8).contains(&to_r) && (0..8).contains(&to_c) {
                let to_sq = (to_r * 8 + to_c) as u8;
                moves.push(Lc0MoveKey {
                    from_sq: from_u8,
                    to_sq,
                    promotion: None,
                });
            }
        }

        // 3. Underpromotions (Planes 64..72: 3 directions [left, straight, right] * 3 pieces [N, B, R])
        if from_r == 6 {
            // Direction offsets relative to move direction: Left-capture (-1), Forward (0), Right-capture (1)
            for dc in [-1, 0, 1] {
                let to_c = from_c + dc;
                if (0..8).contains(&to_c) {
                    let to_sq = (7 * 8 + to_c) as u8;
                    // Lc0 underpromotion piece order: Knight, Bishop, Rook
                    for role in [Role::Knight, Role::Bishop, Role::Rook] {
                        moves.push(Lc0MoveKey {
                            from_sq: from_u8,
                            to_sq,
                            promotion: Some(role),
                        });
                    }
                }
            }
        }
    }

    moves
});

static MOVE_TO_INDEX: LazyLock<HashMap<Lc0MoveKey, usize>> = LazyLock::new(|| {
    LC0_MOVES
        .iter()
        .enumerate()
        .map(|(idx, &key)| (key, idx))
        .collect()
});

fn flip_square(sq: Square) -> Square {
    Square::from_coords(sq.file(), sq.rank().flip_vertical())
}

pub fn encode_move(m: &Move, turn: Color) -> Option<usize> {
    let (mut from_sq, mut to_sq) = match *m {
        Move::Normal { from, to, .. } | Move::EnPassant { from, to, .. } => (from, to),
        Move::Castle { king, .. } => {
            let to = match m.castling_side() {
                Some(CastlingSide::KingSide) => Square::from_coords(File::G, king.rank()),
                Some(CastlingSide::QueenSide) => Square::from_coords(File::C, king.rank()),
                None => return None,
            };
            (king, to)
        }
        Move::Put { .. } => return None,
    };

    if turn == Color::Black {
        from_sq = flip_square(from_sq);
        to_sq = flip_square(to_sq);
    }

    let promotion = match m.promotion() {
        Some(Role::Queen) => None,
        other => other,
    };

    let key = Lc0MoveKey {
        from_sq: u8::from(from_sq),
        to_sq: u8::from(to_sq),
        promotion,
    };

    MOVE_TO_INDEX.get(&key).copied()
}

pub fn decode_move(pos: &Chess, index: usize) -> Option<Move> {
    let key = LC0_MOVES.get(index)?;
    let turn = pos.turn();

    let mut target_from = Square::new(key.from_sq as u32);
    let mut target_to = Square::new(key.to_sq as u32);

    if turn == Color::Black {
        target_from = flip_square(target_from);
        target_to = flip_square(target_to);
    }

    pos.legal_moves().into_iter().find(|m| {
        let (m_from, m_to) = match *m {
            Move::Normal { from, to, .. } | Move::EnPassant { from, to, .. } => (from, to),
            Move::Castle { king, .. } => {
                let to = match m.castling_side() {
                    Some(CastlingSide::KingSide) => Square::from_coords(File::G, king.rank()),
                    Some(CastlingSide::QueenSide) => Square::from_coords(File::C, king.rank()),
                    None => return false,
                };
                (king, to)
            }
            Move::Put { .. } => return false,
        };

        if m_from != target_from || m_to != target_to {
            return false;
        }

        let m_prom = match m.promotion() {
            Some(Role::Queen) => None,
            other => other,
        };

        m_prom == key.promotion
    })
}
