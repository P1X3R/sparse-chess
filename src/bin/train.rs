use std::{ops::ControlFlow, str::FromStr};

use pgn_reader::Reader;
use pgn_reader::{SanPlus, Visitor};
use shakmaty::{Chess, Color, KnownOutcome, Outcome, Position};
use sparse_chess::sph::LayerParams;
use sparse_chess::{
    chess::ChessModel,
    pre_encoders::{
        move_enc::{MOVE_STRS, decode_move_idx, encode_move},
        position_enc::encode_position,
    },
};
use std::{fs::File, io::BufReader};

const SAVE_EACH: usize = 25;
const EVAL_EACH: usize = 50;

#[derive(Debug, Default, Clone)]
struct ModelErrors {
    policy_correct_cnt: usize,
    policy_error_sum: f32,
    value_error_sum: f32,
    total_positions: usize,
}

#[derive(Clone)]
struct ValidationPosition {
    encoded_pos: Vec<u16>,
    legality_mask: [bool; MOVE_STRS.len()],
    policy_target: u16,
    value_target: u16,
}

#[derive(Debug)]
struct TrainingState {
    position: Chess,
    model: ChessModel,
    known_outcome: KnownOutcome,
}

impl TrainingState {
    fn get_legality_mask(&self, active_color: Color) -> [bool; MOVE_STRS.len()] {
        let mut legality_mask = [false; MOVE_STRS.len()];

        for lm in self.position.legal_moves() {
            let Some(encoded_legal_move) = encode_move(&lm, active_color) else {
                panic!("failed to encode legal move {}", lm)
            };

            legality_mask[encoded_legal_move] = true;

            assert_eq!(
                decode_move_idx(encoded_legal_move, &self.position).expect(&format!(
                    "failed to decode legal move {}/{}",
                    lm, encoded_legal_move
                )),
                lm
            );
        }

        legality_mask
    }

    fn get_value_target(&self, active_color: Color) -> u16 {
        match self.known_outcome {
            KnownOutcome::Draw => 1,
            KnownOutcome::Decisive { winner } => {
                if winner == active_color {
                    0
                } else {
                    2
                }
            }
        }
    }
}

impl Visitor for TrainingState {
    type Tags = Outcome;
    type Movetext = ModelErrors;
    type Output = Option<ModelErrors>;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        ControlFlow::Continue(Outcome::Unknown)
    }

    fn tag(
        &mut self,
        tags: &mut Self::Tags,
        name: &[u8],
        value: pgn_reader::RawTag<'_>,
    ) -> ControlFlow<Self::Output> {
        let val = value.decode_utf8().expect("error decoding tag value");

        match name {
            b"Result" => match Outcome::from_str(&val) {
                Ok(outcome @ Outcome::Known(_)) => *tags = outcome,
                _ => return ControlFlow::Break(None),
            },
            _ => {}
        }

        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        let Outcome::Known(known) = tags else {
            return ControlFlow::Break(None);
        };

        self.position = Chess::default();
        self.known_outcome = known;
        self.model.clean_learning_state();

        ControlFlow::Continue(ModelErrors::default())
    }

    fn san(
        &mut self,
        movetext: &mut Self::Movetext,
        san_plus: SanPlus,
    ) -> ControlFlow<Self::Output> {
        if self.position.is_game_over() {
            return ControlFlow::Continue(());
        }

        let current_color = self.position.turn();
        let m = san_plus.san.to_move(&self.position).expect("legal move");

        let legality_mask = self.get_legality_mask(current_color);
        let policy_target =
            encode_move(&m, current_color).expect("failed to expected encode move") as u16;
        let value_target = self.get_value_target(current_color);

        // Train step with targets provided
        let output = self.model.step(
            &encode_position(&self.position),
            &legality_mask,
            Some((&[policy_target], &[value_target])),
        );

        let target_move_activation = output.policy[policy_target as usize];
        let (predicted_move_idx, _) = output
            .policy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();

        assert_ne!(target_move_activation, 0.0);
        assert!(
            legality_mask[predicted_move_idx],
            "illegal prediction: {}, activation: {}",
            predicted_move_idx, target_move_activation
        );

        movetext.value_error_sum += -output.value[value_target as usize].ln();
        movetext.policy_error_sum += -target_move_activation.ln();
        movetext.total_positions += 1;
        if predicted_move_idx == policy_target as usize {
            movetext.policy_correct_cnt += 1;
        }

        self.position.play_unchecked(m);
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, movetext: Self::Movetext) -> Self::Output {
        Some(movetext)
    }
}

/// Visitor specifically for collecting static validation data prior to training.
struct ValidationCollector {
    position: Chess,
    known_outcome: KnownOutcome,
    collected: Vec<ValidationPosition>,
    max_positions: usize,
}

impl ValidationCollector {
    fn new(max_positions: usize) -> Self {
        Self {
            position: Chess::default(),
            known_outcome: KnownOutcome::Draw,
            collected: Vec::with_capacity(max_positions),
            max_positions,
        }
    }
}

impl Visitor for ValidationCollector {
    type Tags = Outcome;
    type Movetext = ();
    type Output = Option<()>;

    fn begin_tags(&mut self) -> ControlFlow<Self::Output, Self::Tags> {
        if self.collected.len() >= self.max_positions {
            ControlFlow::Break(None)
        } else {
            ControlFlow::Continue(Outcome::Unknown)
        }
    }

    fn tag(
        &mut self,
        tags: &mut Self::Tags,
        name: &[u8],
        value: pgn_reader::RawTag<'_>,
    ) -> ControlFlow<Self::Output> {
        let val = value.decode_utf8().expect("error decoding tag value");
        if name == b"Result" {
            if let Ok(outcome @ Outcome::Known(_)) = Outcome::from_str(&val) {
                *tags = outcome;
            } else {
                return ControlFlow::Break(None);
            }
        }
        ControlFlow::Continue(())
    }

    fn begin_movetext(&mut self, tags: Self::Tags) -> ControlFlow<Self::Output, Self::Movetext> {
        let Outcome::Known(known) = tags else {
            return ControlFlow::Break(None);
        };
        self.position = Chess::default();
        self.known_outcome = known;
        ControlFlow::Continue(())
    }

    fn san(
        &mut self,
        _movetext: &mut Self::Movetext,
        san_plus: SanPlus,
    ) -> ControlFlow<Self::Output> {
        if self.collected.len() >= self.max_positions || self.position.is_game_over() {
            return ControlFlow::Break(None);
        }

        let current_color = self.position.turn();
        let m = san_plus.san.to_move(&self.position).expect("legal move");

        let mut legality_mask = [false; MOVE_STRS.len()];
        for lm in self.position.legal_moves() {
            if let Some(encoded_legal_move) = encode_move(&lm, current_color) {
                legality_mask[encoded_legal_move] = true;
            }
        }

        let policy_target = encode_move(&m, current_color).unwrap() as u16;
        let value_target = match self.known_outcome {
            KnownOutcome::Draw => 1,
            KnownOutcome::Decisive { winner } => {
                if winner == current_color {
                    0
                } else {
                    2
                }
            }
        };

        self.collected.push(ValidationPosition {
            encoded_pos: encode_position(&self.position).to_vec(),
            legality_mask,
            policy_target,
            value_target,
        });

        self.position.play_unchecked(m);
        ControlFlow::Continue(())
    }

    fn end_game(&mut self, _movetext: Self::Movetext) -> Self::Output {
        Some(())
    }
}

/// Evaluates the model on the static validation set without passing targets (no weight updates).
fn evaluate_validation_set(model: &mut ChessModel, val_set: &[ValidationPosition]) -> ModelErrors {
    let mut errors = ModelErrors::default();

    // Clear learning state before running validation inference pass
    model.clean_learning_state();

    for pos in val_set {
        // Passing target = None ensures model weights are NOT updated
        let output = model.step(&pos.encoded_pos, &pos.legality_mask, None);

        let target_move_activation = output.policy[pos.policy_target as usize].max(1e-7);
        let (predicted_move_idx, _) = output
            .policy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();

        errors.value_error_sum += -output.value[pos.value_target as usize].max(1e-7).ln();
        errors.policy_error_sum += -target_move_activation.ln();
        errors.total_positions += 1;

        if predicted_move_idx == pos.policy_target as usize {
            errors.policy_correct_cnt += 1;
        }
    }

    // Clean state again after evaluation to leave model fresh for next training game
    model.clean_learning_state();
    errors
}

fn default_model() -> ChessModel {
    let pipeline_size = [(8, 8, 16), (4, 4, 24), (2, 2, 36)];

    let params = [
        LayerParams {
            decoder_lr: 0.02,
            encoder_lr: 0.1,
            radius: 2,
            learning_radius: 2,
            choice: 0.01,
            vigilance: 0.9,
            active_ratio: 0.1,
            half_dendrites: 4,
            decoder_scale: 10.0,
        },
        LayerParams {
            decoder_lr: 0.02,
            encoder_lr: 0.1,
            radius: 2,
            learning_radius: 2,
            choice: 0.01,
            vigilance: 0.75,
            active_ratio: 0.1,
            half_dendrites: 4,
            decoder_scale: 8.0,
        },
        LayerParams {
            decoder_lr: 0.01,
            encoder_lr: 0.05,
            radius: 1,
            learning_radius: 1,
            choice: 0.1,
            vigilance: 0.5,
            active_ratio: 0.1,
            half_dendrites: 8,
            decoder_scale: 8.0,
        },
    ];

    ChessModel::new(&pipeline_size, &params)
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("starting...");

    // 1. Load validation benchmark dataset (1,000 positions)
    println!("loading validation set...");
    let val_file = File::open("val-games.pgn").or_else(|_| File::open("games-3.5s.pgn"))?;
    let mut val_collector = ValidationCollector::new(1000);
    let mut val_reader = Reader::new(BufReader::new(val_file));
    while let Ok(Some(_)) = val_reader.read_game(&mut val_collector) {
        if val_collector.collected.len() >= 1000 {
            break;
        }
    }
    let validation_set = val_collector.collected;
    println!(
        "validation set ready with {} positions",
        validation_set.len()
    );

    // 2. Setup training state and load model
    let snapshot_path = "model.bin";
    let train_file = File::open("games-3.5s.pgn")?;
    let reader = BufReader::new(train_file);
    let mut pgn_reader = Reader::new(reader);

    let mut state = TrainingState {
        position: Chess::default(),
        model: {
            println!("loading model snapshot...");
            match ChessModel::load_from_file(snapshot_path) {
                Ok(model) => model,
                Err(err) => {
                    eprintln!("error opening snapshot: {}", err);
                    println!("using random initialization instead...");
                    default_model()
                }
            }
        },
        known_outcome: KnownOutcome::Draw,
    };

    // 3. Main Training Loop
    let mut game_cnt = 0;
    while let Some(Some(movetext)) = pgn_reader.read_game(&mut state)? {
        game_cnt += 1;

        let total = movetext.total_positions as f32;
        let value_cce = movetext.value_error_sum / total;
        let policy_cce = movetext.policy_error_sum / total;
        let policy_accuracy = movetext.policy_correct_cnt as f32 / total;

        println!(
            "{}. [Train Game ({} pos)]: Value CCE: {:.2}, Policy CCE: {:.2}, Top-1 Acc: {:.2}%",
            game_cnt,
            movetext.total_positions,
            value_cce,
            policy_cce,
            policy_accuracy * 100.0
        );

        // Run validation check periodically
        if game_cnt % EVAL_EACH == 0 && !validation_set.is_empty() {
            let val_metrics = evaluate_validation_set(&mut state.model, &validation_set);
            let val_total = val_metrics.total_positions as f32;
            println!(
                "   ==> [VALIDATION ({} pos)]: Value CCE: {:.2}, Policy CCE: {:.2}, Top-1 Acc: {:.2}%",
                val_metrics.total_positions,
                val_metrics.value_error_sum / val_total,
                val_metrics.policy_error_sum / val_total,
                (val_metrics.policy_correct_cnt as f32 / val_total) * 100.0
            );
        }

        if game_cnt % SAVE_EACH == 0 {
            println!("saving snapshot...");
            state.model.save_to_file(snapshot_path)?;
        }
    }

    println!("training finished...");

    Ok(())
}
