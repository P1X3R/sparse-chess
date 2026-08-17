use std::{
    io::{BufWriter, Write},
    path::Path,
};

use crate::{
    coder::{CsdrSize, softmax},
    decoder::{Decoder, DecoderLearningData, DecoderSnapshot},
    encoder::{Encoder, EncoderSnapshot},
    pre_encoders::move_enc::MOVE_STRS,
    sph::{LayerParams, Sph, SphSnapshot},
};
use serde::{Deserialize, Serialize};
use std::fs::File;

#[derive(Debug)]
pub struct PosAuxiliarDim;
impl PosAuxiliarDim {
    pub const AUXILIAR_X: usize = 8;

    pub const TURN_Y: usize = 0;
    pub const FRIENDLY_RIGHTS_Y: usize = 1;
    pub const ENEMY_RIGHTS_Y: usize = 2;
    pub const EP_FILE_Y: usize = 3;
    pub const HM_CLOCK_Y: usize = 4;
}

#[derive(Debug)]
pub struct ModelOutput {
    pub policy: Box<[f32]>,
    pub value: Box<[f32]>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
struct ChessModelSnapshot<'a> {
    policy: DecoderSnapshot<'a>,
    value: DecoderSnapshot<'a>,
    encoder: EncoderSnapshot<'a>,
    body: SphSnapshot<'a>,
    bottom_dendrites: usize,
}

#[derive(Debug)]
pub struct ChessModel {
    body: Sph,

    bottom_encoder: Encoder,
    policy_head: Decoder,
    value_head: Decoder,

    bottom_dendrites: usize,
}

impl<'a> ChessModel {
    pub const INPUT_SIZE: CsdrSize = CsdrSize::new(9, 8, 16);
    pub const POLICY_SIZE: CsdrSize = CsdrSize::new(1, 1, MOVE_STRS.len());
    pub const VALUE_SIZE: CsdrSize = CsdrSize::new(1, 1, 3); // WDL

    pub fn new(pipeline_sizes: &[(usize, usize, usize)], params: &[LayerParams]) -> Self {
        assert_eq!(ChessModel::POLICY_SIZE.flat, ChessModel::POLICY_SIZE.z);
        assert_eq!(ChessModel::VALUE_SIZE.flat, ChessModel::VALUE_SIZE.z);
        assert!(params.len() >= 2);
        assert_eq!(pipeline_sizes.len(), params.len());

        let (bi_x, bi_y, bi_z) = pipeline_sizes[0];
        let bottom_params = params[0];

        let body_input_size = CsdrSize::new(bi_x, bi_y, bi_z);
        let concat_size = CsdrSize::new(bi_x * 2, bi_y, bi_z);

        ChessModel {
            body: Sph::new(pipeline_sizes, &params[1..]),
            bottom_encoder: Encoder::new(
                ChessModel::INPUT_SIZE,
                body_input_size,
                bottom_params.radius,
                bottom_params.learning_radius,
                bottom_params.encoder_lr,
                bottom_params.choice,
                bottom_params.vigilance,
                bottom_params.active_ratio,
            ),
            policy_head: Decoder::new(
                concat_size,
                ChessModel::POLICY_SIZE,
                bottom_params.half_dendrites,
                bottom_params.radius,
                bottom_params.decoder_scale,
                bottom_params.decoder_lr,
            ),
            value_head: Decoder::new(
                concat_size,
                ChessModel::VALUE_SIZE,
                bottom_params.half_dendrites,
                bottom_params.radius,
                bottom_params.decoder_scale,
                bottom_params.decoder_lr,
            ),

            bottom_dendrites: bottom_params.half_dendrites * 2,
        }
    }

    fn step_policy(
        &self,
        concat: &[u16],
        legality_mask: &[bool],
    ) -> (Box<[f32]>, DecoderLearningData) {
        assert_eq!(legality_mask.len(), ChessModel::POLICY_SIZE.flat);

        let mut dendrite_activations: Box<[f32]> =
            vec![0.0; ChessModel::POLICY_SIZE.flat * self.bottom_dendrites].into_boxed_slice();
        let mut policy: Box<[f32]> = vec![0.0; ChessModel::POLICY_SIZE.z].into_boxed_slice();

        self.policy_head.compute_activations(
            concat,
            0,
            &mut dendrite_activations,
            |z, cell_activation| {
                policy[z] = if legality_mask[z] {
                    cell_activation
                } else {
                    f32::NEG_INFINITY
                }
            },
        );

        softmax(&mut policy);

        (
            policy.clone(),
            DecoderLearningData {
                concat: concat.into(),
                dendrite_activations,
                activations: policy,
            },
        )
    }

    pub fn step(
        &mut self,
        input: &[u16],
        legality_mask: &[bool],
        expected: Option<(&[u16], &[u16])>,
    ) -> ModelOutput {
        let learn = expected.is_some();

        let (hidden, enc_data) = self.bottom_encoder.forward(input);
        if learn {
            self.bottom_encoder.learn(input, &hidden, &enc_data);
        }

        let feedback = self.body.step(&hidden, learn);

        let mut concat = Vec::with_capacity(hidden.len() + feedback.len());
        concat.extend_from_slice(&hidden);
        concat.extend_from_slice(&feedback);

        let (policy, policy_data) = self.step_policy(&concat, legality_mask);
        let (_, value_data) = self.value_head.forward(&concat);
        let value = value_data.activations.clone();

        if let Some((policy_target, value_target)) = expected {
            self.policy_head.learn(policy_target, &policy_data);
            self.value_head.learn(value_target, &value_data);
        }

        ModelOutput { policy, value }
    }

    pub fn clean_learning_state(&mut self) {
        self.body.clean_learning_state();
    }

    fn get_snapshot(&'a self) -> ChessModelSnapshot<'a> {
        ChessModelSnapshot {
            policy: self.policy_head.get_snapshot(),
            value: self.value_head.get_snapshot(),
            encoder: self.bottom_encoder.get_snapshot(),
            body: self.body.get_snapshot(),
            bottom_dendrites: self.bottom_dendrites,
        }
    }

    fn from_snapshot(snapshot: ChessModelSnapshot) -> Self {
        Self {
            body: Sph::from_snapshot(snapshot.body),
            bottom_encoder: Encoder::from_snapshot(snapshot.encoder),
            policy_head: Decoder::from_snapshot(snapshot.policy),
            value_head: Decoder::from_snapshot(snapshot.value),
            bottom_dendrites: snapshot.bottom_dendrites,
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&'a self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        postcard::to_io(&self.get_snapshot(), &mut writer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        writer.flush()?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let snapshot: ChessModelSnapshot = postcard::from_bytes(&bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(ChessModel::from_snapshot(snapshot))
    }
}
