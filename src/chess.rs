use crate::{
    coder::{CsdrSize, SoftmaxState},
    decoder::{Decoder, DecoderLearningData},
    encoder::Encoder,
    pre_encoders::move_enc::MOVE_STRS,
    sph::{LayerParams, Sph},
};

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

#[derive(Debug)]
pub struct ChessModel {
    body: Sph,

    bottom_encoder: Encoder,
    policy_head: Decoder,
    value_head: Decoder,

    prev_policy_data: Option<DecoderLearningData>,
    prev_value_data: Option<DecoderLearningData>,

    bottom_dendrites: usize,
}

impl ChessModel {
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

            prev_policy_data: None,
            prev_value_data: None,

            bottom_dendrites: bottom_params.half_dendrites * 2,
        }
    }

    fn step_policy(
        &self,
        concat: &[u16],
        legality_mask: &[bool],
    ) -> (Box<[f32]>, DecoderLearningData) {
        assert_eq!(legality_mask.len(), ChessModel::POLICY_SIZE.flat);

        let mut dendrite_activations: Box<[i16]> =
            vec![0; ChessModel::POLICY_SIZE.flat * self.bottom_dendrites].into_boxed_slice();

        let mut policy: Box<[f32]> = vec![0.0; ChessModel::POLICY_SIZE.z].into_boxed_slice();
        let mut policy_softmax = SoftmaxState::new();

        self.policy_head.compute_activations(
            concat,
            0,
            &mut dendrite_activations,
            |z, cell_activation| {
                if !legality_mask[z] {
                    return;
                }

                policy_softmax.update(cell_activation);
                policy[z] = cell_activation;
            },
        );

        policy_softmax.normalize(&mut policy);

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
        let learn =
            expected.is_some() && self.prev_policy_data.is_some() && self.prev_value_data.is_some();

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

        let (policy, policy_data) = self.step_policy(&concat, legality_mask);
        let (_, value_data) = self.value_head.forward(&concat);
        let value = value_data.activations.clone();

        self.prev_policy_data = Some(policy_data);
        self.prev_value_data = Some(value_data);

        ModelOutput { policy, value }
    }

    pub fn clean_learning_state(&mut self) {
        self.prev_policy_data = None;
        self.prev_value_data = None;
        self.body.clean_learning_state();
    }
}
