use std::{
    io::{BufWriter, Write},
    path::Path,
};

use crate::{
    coder::CsdrSize,
    decoder::DecoderSnapshot,
    encoder::{Encoder, EncoderSnapshot},
    sph::{LayerParams, Sph, SphSnapshot},
};
use crate::{decoder::Head, encoder::EncoderVisibleParams};
use serde::{Deserialize, Serialize};
use std::fs::File;

#[derive(Debug)]
pub struct PosAuxiliarDim;
impl PosAuxiliarDim {
    pub const RIGHTS_US_QS_Y: usize = 0;
    pub const RIGHTS_US_KS_Y: usize = 1;
    pub const RIGHTS_THEM_QS_Y: usize = 2;
    pub const RIGHTS_THEM_KS_Y: usize = 3;
    pub const TURN_Y: usize = 4;
    pub const HM_CLOCK_Y: usize = 5;
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
    pub policy_head: Head,
    pub value_head: Head,
}

impl<'a> ChessModel {
    pub const PLANES_SIZE: CsdrSize = CsdrSize::new(8, 8, 13);
    pub const AUXILIARY_SIZE: CsdrSize = CsdrSize::new(1, 6, 13);
    pub const POLICY_SIZE: CsdrSize = CsdrSize::new(1, 1, 1858);
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

        ChessModel {
            body: Sph::new(pipeline_sizes, params),
            bottom_encoder: Encoder::new(
                body_input_size,
                &[
                    EncoderVisibleParams {
                        visible_size: Self::PLANES_SIZE,
                        radius: bottom_params.radius,
                    },
                    EncoderVisibleParams {
                        visible_size: Self::AUXILIARY_SIZE,
                        radius: 3,
                    },
                ],
                bottom_params.learning_radius,
                bottom_params.encoder_lr,
                bottom_params.choice,
                bottom_params.vigilance,
                bottom_params.active_ratio,
            ),
            policy_head: Head::new(
                body_input_size,
                ChessModel::POLICY_SIZE,
                bottom_params.policy_half_dendrites,
                2,
                bottom_params.policy_scale,
                bottom_params.policy_lr,
            ),
            value_head: Head::new(
                body_input_size,
                ChessModel::VALUE_SIZE,
                2,
                bottom_params.value_half_dendrites,
                bottom_params.value_scale,
                bottom_params.value_lr,
            ),
        }
    }

    pub fn step(
        &mut self,
        input: &[&[u16]],
        legality_mask: &[bool],
        expected: Option<(&[f32], &[f32])>,
    ) -> ModelOutput {
        let learn = expected.is_some();

        let (hidden, enc_data) = self.bottom_encoder.forward(input);
        if learn {
            self.bottom_encoder.learn(input, &hidden, &enc_data);
        }

        let feedback = self.body.step(&hidden, learn);

        let concat = &[&hidden, &feedback[..]];

        let (_, policy_data) = self.policy_head.forward(concat, legality_mask);
        let (_, value_data) = self.value_head.forward(concat, &[true, true, true]);

        if let Some((policy_target, value_target)) = expected {
            self.policy_head.learn(policy_target, &policy_data);
            self.value_head.learn(value_target, &value_data);
        }

        ModelOutput {
            policy: policy_data.activations,
            value: value_data.activations,
        }
    }

    pub fn get_commited_rates(&self) -> (f32, Vec<f32>) {
        (
            self.bottom_encoder.get_commited_rate(),
            self.body.get_committed_rates(),
        )
    }

    fn get_snapshot(&'a self) -> ChessModelSnapshot<'a> {
        ChessModelSnapshot {
            policy: self.policy_head.get_snapshot(),
            value: self.value_head.get_snapshot(),
            encoder: self.bottom_encoder.get_snapshot(),
            body: self.body.get_snapshot(),
        }
    }

    fn from_snapshot(snapshot: ChessModelSnapshot) -> Self {
        Self {
            body: Sph::from_snapshot(snapshot.body),
            bottom_encoder: Encoder::from_snapshot(snapshot.encoder),
            policy_head: Head::from_snapshot(snapshot.policy),
            value_head: Head::from_snapshot(snapshot.value),
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
