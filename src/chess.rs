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
    policy_dendrites: usize,
}

#[derive(Debug)]
pub struct BottomLayerParams {
    pub encoder_lr: f32,
    pub radius: i16,
    pub learning_radius: isize,
    pub choice: f32,
    pub vigilance: f32,
    pub active_ratio: f32,
    pub policy_lr: f32,
    pub policy_half_dendrites: usize,
    pub policy_scale: f32,
    pub value_lr: f32,
    pub value_half_dendrites: usize,
    pub value_scale: f32,
}

#[derive(Debug)]
pub struct ChessModel {
    pub body: Sph,

    bottom_encoder: Encoder,
    policy_head: Decoder,
    value_head: Decoder,

    policy_dendrites: usize,
}

impl<'a> ChessModel {
    pub const INPUT_SIZE: CsdrSize = CsdrSize::new(9, 8, 16);
    pub const POLICY_SIZE: CsdrSize = CsdrSize::new(1, 1, MOVE_STRS.len());
    pub const VALUE_SIZE: CsdrSize = CsdrSize::new(1, 1, 3); // WDL

    pub fn new(
        pipeline_sizes: &[(usize, usize, usize)],
        params: &[LayerParams],
        bottom_params: &BottomLayerParams,
    ) -> Self {
        assert_eq!(ChessModel::POLICY_SIZE.flat, ChessModel::POLICY_SIZE.z);
        assert_eq!(ChessModel::VALUE_SIZE.flat, ChessModel::VALUE_SIZE.z);
        assert!(params.len() >= 1);
        assert_eq!(pipeline_sizes.len(), params.len() + 1);

        let (bi_x, bi_y, bi_z) = pipeline_sizes[0];

        let body_input_size = CsdrSize::new(bi_x, bi_y, bi_z);
        let concat_size = CsdrSize::new(bi_x * 2, bi_y, bi_z);

        ChessModel {
            body: Sph::new(pipeline_sizes, params),
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
                bottom_params.policy_half_dendrites,
                bottom_params.radius,
                bottom_params.policy_scale,
                bottom_params.policy_lr,
            ),
            value_head: Decoder::new(
                concat_size,
                ChessModel::VALUE_SIZE,
                bottom_params.value_half_dendrites,
                bottom_params.radius,
                bottom_params.value_scale,
                bottom_params.value_lr,
            ),

            policy_dendrites: bottom_params.policy_half_dendrites * 2,
        }
    }

    fn step_policy(&self, concat: &[u16], legality_mask: &[bool]) -> DecoderLearningData {
        assert_eq!(legality_mask.len(), ChessModel::POLICY_SIZE.flat);

        let mut dendrite_activations: Box<[f32]> =
            vec![0.0; ChessModel::POLICY_SIZE.flat * self.policy_dendrites].into_boxed_slice();
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

        DecoderLearningData {
            concat: concat.into(),
            dendrite_activations,
            activations: policy,
        }
    }

    pub fn step(
        &mut self,
        input: &[u16],
        legality_mask: &[bool],
        expected: Option<(&[f32], &[f32])>,
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

        let policy_data = self.step_policy(&concat, legality_mask);
        let (_, value_data) = self.value_head.forward(&concat);
        let value = value_data.activations.clone();

        if let Some((policy_target, value_target)) = expected {
            self.policy_head.learn_flat(policy_target, &policy_data);
            self.value_head.learn_flat(value_target, &value_data);
        }

        ModelOutput {
            policy: policy_data.activations,
            value,
        }
    }

    fn get_snapshot(&'a self) -> ChessModelSnapshot<'a> {
        ChessModelSnapshot {
            policy: self.policy_head.get_snapshot(),
            value: self.value_head.get_snapshot(),
            encoder: self.bottom_encoder.get_snapshot(),
            body: self.body.get_snapshot(),
            policy_dendrites: self.policy_dendrites,
        }
    }

    fn from_snapshot(snapshot: ChessModelSnapshot) -> Self {
        Self {
            body: Sph::from_snapshot(snapshot.body),
            bottom_encoder: Encoder::from_snapshot(snapshot.encoder),
            policy_head: Decoder::from_snapshot(snapshot.policy),
            value_head: Decoder::from_snapshot(snapshot.value),
            policy_dendrites: snapshot.policy_dendrites,
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
