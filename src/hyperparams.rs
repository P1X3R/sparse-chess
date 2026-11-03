use crate::{chess::BottomLayerParams, sph::LayerParams};

pub const BODY_LEN: usize = 2;

pub const BOTTOM_PARAMS: BottomLayerParams = BottomLayerParams {
    encoder_lr: 0.2,
    radius: 0,
    learning_radius: 1,
    choice: 1e-4,
    vigilance: 0.8703,
    active_ratio: 1.0,
    policy_lr: 0.02,
    policy_half_dendrites: 2,
    policy_scale: 1.0,
    value_lr: 0.02,
    value_half_dendrites: 6,
    value_scale: 1.0,
};

pub const PARAMS: [LayerParams; BODY_LEN] = [
    LayerParams {
        decoder_lr: 0.02,
        encoder_lr: 0.1,
        radius: 1,
        learning_radius: 1,
        choice: 1e-4,
        vigilance: 0.8,
        active_ratio: 1.0,
        half_dendrites: 2,
        decoder_scale: 8.0,
    },
    LayerParams {
        decoder_lr: 0.01,
        encoder_lr: 0.1,
        radius: 1,
        learning_radius: 1,
        choice: 1e-4,
        vigilance: 0.8,
        active_ratio: 1.0,
        half_dendrites: 3,
        decoder_scale: 8.0,
    },
];

pub const PIPELINE_SIZES: [(usize, usize, usize); BODY_LEN + 1] =
    [(8, 8, 32), (8, 8, 32), (8, 8, 32)];

pub const VALIDATION_SAMPLES: usize = 200;
pub const VALIDATE_EVERY: usize = 1000;
pub const LOG_EVERY: usize = 100;
pub const LAMBDA: f32 = 0.25;
