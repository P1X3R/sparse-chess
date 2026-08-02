use crate::model::{
    coder::CsdrSize,
    decoder::{Decoder, DecoderLearningData},
    encoder::Encoder,
    sph::{LayerParams, Sph},
};

#[derive(Debug)]
pub struct ModelOutput {
    pub policy_max: Box<[u16]>,
    pub policy: Box<[f32]>,
    pub value: Box<[u16]>,
}

#[derive(Debug)]
pub struct ChessModel {
    body: Sph,

    bottom_encoder: Encoder,
    policy_head: Decoder,
    value_head: Decoder,

    prev_policy_data: Option<DecoderLearningData>,
    prev_value_data: Option<DecoderLearningData>,
}

impl ChessModel {
    pub fn new(pipeline_sizes: &[(usize, usize, usize)], params: &[LayerParams]) -> Self {
        assert!(params.len() >= 2);
        assert_eq!(pipeline_sizes.len(), params.len());

        let (bi_x, bi_y, bi_z) = pipeline_sizes[0];
        let bottom_params = params[0];

        let input_size = CsdrSize::new(9, 8, 16);
        let body_input_size = CsdrSize::new(bi_x, bi_y, bi_z);
        let concat_size = CsdrSize::new(bi_x * 2, bi_y, bi_z);
        let policy_size = CsdrSize::new(1, 1, 4672);
        let value_size = CsdrSize::new(1, 1, 256);

        ChessModel {
            body: Sph::new(pipeline_sizes, &params[1..]),
            bottom_encoder: Encoder::new(
                input_size,
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
                policy_size,
                bottom_params.half_dendrites,
                bottom_params.radius,
                bottom_params.decoder_scale,
                bottom_params.decoder_lr,
            ),
            value_head: Decoder::new(
                concat_size,
                value_size,
                bottom_params.half_dendrites,
                bottom_params.radius,
                bottom_params.decoder_scale,
                bottom_params.decoder_lr,
            ),

            prev_policy_data: None,
            prev_value_data: None,
        }
    }

    pub fn step(&mut self, input: &[u16], expected: Option<(&[u16], &[u16])>) -> ModelOutput {
        let learn = expected.is_some();

        let (hidden, enc_data) = self.bottom_encoder.forward(input);
        if learn {
            self.bottom_encoder.learn(input, &hidden, &enc_data);
        }

        let feedback = self.body.step(&hidden, learn);

        let mut concat = Vec::with_capacity(hidden.len() + feedback.len());
        concat.extend_from_slice(&hidden);
        concat.extend_from_slice(&feedback);

        if let Some((policy_target, value_target)) = expected {
            if let Some(prev_data) = self.prev_policy_data.take() {
                self.policy_head.learn(policy_target, &prev_data);
            }
            if let Some(prev_data) = self.prev_value_data.take() {
                self.value_head.learn(value_target, &prev_data);
            }
        }

        let (policy, policy_data) = self.policy_head.forward(&concat);
        let (value, value_data) = self.value_head.forward(&concat);

        let policy_activations = policy_data.2.clone();

        self.prev_policy_data = Some(policy_data);
        self.prev_value_data = Some(value_data);

        ModelOutput {
            policy_max: policy,
            policy: policy_activations,
            value: value,
        }
    }
}
