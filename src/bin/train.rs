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

const SAVE_EACH: usize = 10;

#[derive(Debug, Default)]
struct ModelErrors {
    policy_correct_cnt: usize,
    policy_error_sum: f32,
    value_error_sum: f32,
    total_positions: usize,
}

#[derive(Debug)]
struct TrainigState {
    position: Chess,
    model: ChessModel,
    outcome: u16,
    model_color: Color,
}

impl TrainigState {
    fn get_legality_mask(&self) -> [bool; MOVE_STRS.len()] {
        let mut legality_mask = [false; MOVE_STRS.len()];

        for lm in self.position.legal_moves() {
            let Some(encoded_legal_move) = encode_move(&lm, self.model_color) else {
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

    fn get_value_target(&self, known: KnownOutcome) -> u16 {
        match known {
            KnownOutcome::Draw => 1,
            KnownOutcome::Decisive { winner } => {
                if winner == self.model_color {
                    0
                } else {
                    2
                }
            }
        }
    }
}

impl Visitor for TrainigState {
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
        self.outcome = self.get_value_target(known);
        self.model_color = [Color::White, Color::Black][fastrand::usize(0..2)];
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

        let m = san_plus.san.to_move(&self.position).expect("legal move");

        if self.position.turn() != self.model_color {
            self.position.play_unchecked(m);
            return ControlFlow::Continue(());
        }

        let legality_mask = self.get_legality_mask();
        let policy_target =
            encode_move(&m, self.model_color).expect("failed to expected encode move") as u16;
        let output = self.model.step(
            &encode_position(&self.position),
            &legality_mask,
            Some((&[policy_target], &[self.outcome])),
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

        movetext.value_error_sum += 1.0 - output.value[self.outcome as usize];
        movetext.policy_error_sum += 1.0 - target_move_activation;
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

fn default_model() -> ChessModel {
    let pipeline_size = [(8, 8, 32), (4, 4, 64)];

    let default_params = LayerParams {
        decoder_lr: 0.25,
        encoder_lr: 0.025,
        radius: 2,
        learning_radius: 2,
        choice: 0.1,
        vigilance: 0.85,
        active_ratio: 0.05,
        half_dendrites: 16,
        decoder_scale: 10.0,
    };

    let params = [default_params, default_params];

    ChessModel::new(&pipeline_size, &params)
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("starting...");

    let snapshot_path = "model.bin";
    let file = File::open("games-3.5s.pgn")?;
    let reader = BufReader::new(file);
    let mut pgn_reader = Reader::new(reader);

    let mut state = TrainigState {
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
        outcome: 3, // Invalid value as placeholder
        model_color: Color::White,
    };

    let mut game_cnt = 0;
    while let Some(Some(movetext)) = pgn_reader.read_game(&mut state)? {
        game_cnt += 1;

        let total = movetext.total_positions as f32;
        let value_mae = movetext.value_error_sum / total;
        let policy_mae = movetext.policy_error_sum / total;
        let policy_accuracy = movetext.policy_correct_cnt as f32 / total;

        println!(
            "{}. [Game loss ({} evaluated positions)]: Value MAE: {:.2}, Policy MAE: {:.2}, Policy Accuracy: {:.2}%",
            game_cnt,
            movetext.total_positions,
            value_mae,
            policy_mae,
            policy_accuracy * 100.0
        );

        if game_cnt % SAVE_EACH == 0 {
            println!("saving snapshot...");
            state.model.save_to_file(snapshot_path)?;
        }
    }

    println!("training finished...");

    Ok(())
}
