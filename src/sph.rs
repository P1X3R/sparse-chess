use serde::{Deserialize, Serialize};

use crate::{
    coder::CsdrSize,
    decoder::{Decoder, DecoderLearningData, DecoderSnapshot},
    encoder::{Encoder, EncoderSnapshot, EncoderVisibleParams},
};

#[derive(Serialize, Deserialize)]
#[serde(bound(deserialize = "'de: 'a"))]
pub struct SphSnapshot<'a> {
    layers: Vec<(EncoderSnapshot<'a>, DecoderSnapshot<'a>)>,
    input_cols: usize,
}

#[derive(Debug, Default)]
struct LayerState {
    hidden_state: Vec<u16>,
    prediction: Vec<u16>,
    prev_decoder_data: Option<DecoderLearningData>,
}

#[derive(Debug)]
pub struct SphLayer {
    encoder: Encoder,
    decoder: Decoder,
    state: LayerState,
}

#[derive(Debug)]
pub struct Sph {
    layers: Vec<SphLayer>,
    input_cols: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerParams {
    pub decoder_lr: f32,
    pub encoder_lr: f32,
    pub radius: i16,
    pub learning_radius: isize,
    pub choice: f32,
    pub vigilance: f32,
    pub active_ratio: f32,
    pub half_dendrites: usize,
    pub decoder_scale: f32,
}

impl SphLayer {
    pub fn new(encoder: Encoder, decoder: Decoder) -> Self {
        let hidden_cols = encoder.hidden_size.cols;
        let visible_cols = encoder.visible_layers[0].visible_size.cols;

        Self {
            encoder,
            decoder,
            state: LayerState {
                hidden_state: vec![0; hidden_cols],
                prediction: vec![0; visible_cols],
                prev_decoder_data: None,
            },
        }
    }
}

impl<'a> Sph {
    pub fn new(pipeline_sizes: &[(usize, usize, usize)], params: &[LayerParams]) -> Self {
        assert!(
            pipeline_sizes.len() > 1,
            "The model must have at least an input and hidden state size"
        );
        assert_eq!(
            pipeline_sizes.len(),
            params.len() + 1,
            "Params count must equal pipeline_sizes count minus 1"
        );

        let layers = pipeline_sizes
            .windows(2)
            .zip(params)
            .map(|(pair, layer_params)| {
                let (prev_x, prev_y, prev_z) = pair[0];
                let (size_x, size_y, size_z) = pair[1];

                let visible_size = CsdrSize::new(prev_x, prev_y, prev_z);
                let hidden_size = CsdrSize::new(size_x, size_y, size_z);

                assert_ne!(layer_params.decoder_scale, 0.0);

                SphLayer::new(
                    Encoder::new(
                        hidden_size,
                        &[EncoderVisibleParams {
                            visible_size,
                            radius: layer_params.radius,
                        }],
                        layer_params.learning_radius,
                        layer_params.encoder_lr,
                        layer_params.choice,
                        layer_params.vigilance,
                        layer_params.active_ratio,
                    ),
                    Decoder::new(
                        hidden_size,
                        visible_size,
                        2,
                        layer_params.half_dendrites,
                        layer_params.radius,
                        layer_params.decoder_scale,
                        layer_params.decoder_lr,
                    ),
                )
            })
            .collect();

        Self {
            layers,
            input_cols: pipeline_sizes[0].0 * pipeline_sizes[0].1,
        }
    }

    /// Processes a single input state and returns the predicted next state
    pub fn step(&mut self, input: &[u16], learn: bool) -> Vec<u16> {
        assert_eq!(input.len(), self.input_cols, "Input dimension mismatch");

        // --- Bottom -> Top Pass ---
        let mut current_input = input;

        for layer in &mut self.layers {
            let (hidden, enc_data) = layer.encoder.forward(&[current_input]);

            if learn {
                layer.encoder.learn(&[current_input], &hidden, &enc_data);
            }

            layer.state.hidden_state = hidden.into();
            current_input = &layer.state.hidden_state;
        }

        // --- Top -> Bottom Pass ---
        let mut feedback: Option<Vec<u16>> = None;
        let num_layers = self.layers.len();

        for idx in (0..num_layers).rev() {
            let (left, right) = self.layers.split_at_mut(idx);
            let layer = &mut right[0];

            let target_data = if idx == 0 {
                input
            } else {
                &left[idx - 1].state.hidden_state
            };

            let hidden_slice = &layer.state.hidden_state[..];
            let decoder_input = match &feedback {
                Some(fb) => &[hidden_slice, fb][..],
                None => &[hidden_slice][..],
            };

            if learn && let Some(prev_data) = layer.state.prev_decoder_data.take() {
                layer.decoder.learn(target_data, &prev_data);
            }

            let (prediction, learning_data) = layer.decoder.forward(decoder_input);
            let pred_vec: Vec<u16> = prediction.into();

            layer.state.prediction = pred_vec.clone();
            layer.state.prev_decoder_data = Some(learning_data);

            feedback = Some(pred_vec);
        }

        self.layers[0].state.prediction.clone()
    }

    pub fn get_committed_rates(&self) -> Vec<f32> {
        self.layers
            .iter()
            .map(|l| l.encoder.get_commited_rate())
            .collect()
    }

    pub fn clean_learning_state(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.state.prev_decoder_data = None;
        }
    }

    pub fn get_snapshot(&'a self) -> SphSnapshot<'a> {
        SphSnapshot {
            layers: self
                .layers
                .iter()
                .map(|l| (l.encoder.get_snapshot(), l.decoder.get_snapshot()))
                .collect(),
            input_cols: self.input_cols,
        }
    }

    pub fn from_snapshot(snapshot: SphSnapshot) -> Self {
        Sph {
            layers: snapshot
                .layers
                .into_iter()
                .map(|(enc, dec)| {
                    SphLayer::new(
                        Encoder::from_snapshot(enc),
                        Decoder::from_snapshot(dec).expect(
                            "couldn't find a radius in a body's layer while loading a snapshot",
                        ),
                    )
                })
                .collect(),
            input_cols: snapshot.input_cols,
        }
    }
}
