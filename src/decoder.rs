use std::borrow::Cow;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    coder::{
        column_wise_one_hot, rand_round, softmax, CsdrSize, FieldBounds, FieldEntry,
        ReceptiveField,
    },
    flat_index,
};

type LocalField = ReceptiveField<FieldEntry>;

const LRELU_FACTOR: f32 = 0.01;

fn unit_step(x: f32) -> f32 {
    if x < 0.0 {
        LRELU_FACTOR
    } else {
        1.0
    }
}

#[derive(Debug, Clone, Default)]
pub struct DecoderLearningData {
    pub(crate) concat: Vec<Box<[u16]>>,
    pub(crate) dendrite_activations: Box<[f32]>,
    pub(crate) activations: Box<[f32]>,
}

#[derive(Serialize, Deserialize)]
pub struct DecoderSnapshot<'a> {
    visible_size: CsdrSize,
    hidden_size: CsdrSize,
    num_visible_layers: usize,
    half_dendrites: usize,
    radius: Option<i16>,
    scale: f32,
    lr: f32,
    #[serde(borrow)]
    weights: Cow<'a, [i8]>,
}

#[derive(Debug)]
pub struct Decoder {
    pub(crate) visible_size: CsdrSize,
    pub(crate) hidden_size: CsdrSize,
    pub(crate) num_visible_layers: usize,
    area: usize,
    receptive_field: LocalField,
    half_dendrites: usize,
    dendrites: usize,
    lr: f32,
    weights: Box<[i8]>,
    scale: f32,
}

impl<'a> Decoder {
    pub fn new(
        visible_size: CsdrSize,
        hidden_size: CsdrSize,
        num_visible_layers: usize,
        half_dendrites: usize,
        radius: i16,
        scale: f32,
        lr: f32,
    ) -> Self {
        let mut rng = fastrand::Rng::new();
        let dendrites = half_dendrites * 2;
        let diameter = radius * 2 + 1;
        let area = (diameter * diameter) as usize;

        let receptive_field = Decoder::init_receptive_field(
            &hidden_size,
            &visible_size,
            num_visible_layers,
            dendrites,
            radius,
        );

        let total_weights = hidden_size.cols
            * num_visible_layers
            * visible_size.z
            * area
            * hidden_size.z
            * dendrites;

        Self {
            visible_size,
            hidden_size,
            num_visible_layers,
            area,
            receptive_field,
            half_dendrites,
            dendrites,
            lr,
            weights: std::iter::repeat_with(|| rng.i8(-13..=12))
                .take(total_weights)
                .collect(),
            scale: scale / 127.0,
        }
    }

    fn init_receptive_field(
        hidden_size: &CsdrSize,
        visible_size: &CsdrSize,
        num_visible_layers: usize,
        dendrites: usize,
        radius: i16,
    ) -> LocalField {
        let diameter = (2 * radius + 1) as usize;
        let area = diameter * diameter;
        let ratio = (
            visible_size.x as f32 / hidden_size.x as f32,
            visible_size.y as f32 / hidden_size.y as f32,
        );

        let mut local_field_lut = Vec::with_capacity(hidden_size.cols * area);
        let mut local_field_offsets = Vec::with_capacity(hidden_size.cols);

        for hidden_col in 0..hidden_size.cols {
            let bounds = FieldBounds::new(
                hidden_col % hidden_size.x,
                hidden_col / hidden_size.x,
                radius,
                ratio,
                visible_size,
            );
            let start_idx = local_field_lut.len() as u32;

            for visible_y in bounds.clamped_start_y..=bounds.clamped_end_y {
                let in_field_y = visible_y - bounds.field_start_y;

                for visible_x in bounds.clamped_start_x..=bounds.clamped_end_x {
                    let in_field_x = visible_x - bounds.field_start_x;

                    let in_field_idx = flat_index!(
                        [diameter, diameter],
                        [in_field_y as usize, in_field_x as usize]
                    );

                    local_field_lut.push(FieldEntry {
                        input_cell_idx: flat_index!(
                            [visible_size.y, visible_size.x],
                            [visible_y as usize, visible_x as usize]
                        ) as u32,
                        weights_base: flat_index!(
                            [
                                hidden_size.cols,
                                num_visible_layers,
                                visible_size.z,
                                area,
                                hidden_size.z,
                                dendrites
                            ],
                            [hidden_col, 0, 0, in_field_idx, 0, 0]
                        ) as u32,
                    });
                }
            }

            let end = local_field_lut.len() as u32;
            local_field_offsets.push((start_idx, end));
        }

        ReceptiveField {
            lut: local_field_lut.into_boxed_slice(),
            offsets: local_field_offsets.into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn get_weight_idx(
        &self,
        v_layer: usize,
        concat_cell: usize,
        hidden_col: usize,
        area_idx: usize,
        hidden_z: usize,
        dendrite: usize,
    ) -> usize {
        flat_index!(
            [
                self.hidden_size.cols,
                self.num_visible_layers,
                self.visible_size.z,
                self.area,
                self.hidden_size.z,
                self.dendrites
            ],
            [
                hidden_col,
                v_layer,
                concat_cell,
                area_idx,
                hidden_z,
                dendrite
            ]
        )
    }

    #[inline(always)]
    fn accumulate_dendrite_activations(
        &self,
        visible_layers: &[&[u16]],
        local_field: &[FieldEntry],
        dendrite_col: &mut [f32],
    ) {
        for field in local_field {
            let input_col = field.input_cell_idx as usize;

            for (v_layer, layer_data) in visible_layers.iter().enumerate() {
                let concat_cell = layer_data[input_col] as usize;

                let concat_cell_base = field.weights_base as usize
                    + self.get_weight_idx(v_layer, concat_cell, 0, 0, 0, 0);

                for hidden_z in 0..self.hidden_size.z {
                    let weights_start =
                        concat_cell_base + self.get_weight_idx(0, 0, 0, 0, hidden_z, 0);
                    let weights_end = weights_start + self.dendrites;
                    let weights_cell = &self.weights[weights_start..weights_end];

                    let dendritic_start =
                        flat_index!([self.hidden_size.z, self.dendrites], [hidden_z, 0]);
                    let dendritic_end = dendritic_start + self.dendrites;
                    let dendritic_activations_cell =
                        &mut dendrite_col[dendritic_start..dendritic_end];

                    for dendrite in 0..self.dendrites {
                        dendritic_activations_cell[dendrite] += weights_cell[dendrite] as f32;
                    }
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn compute_activations<F>(
        &self,
        visible_layers: &[&[u16]],
        hidden_col: usize,
        dendrite_col: &mut [f32],
        mut yield_activation: F,
    ) where
        F: FnMut(usize, f32),
    {
        let local_field = self.receptive_field.get_col(hidden_col);
        self.accumulate_dendrite_activations(visible_layers, local_field, dendrite_col);

        let field_count = (local_field.len() * self.num_visible_layers) as f32;
        let dendrite_scale = (self.scale / field_count).sqrt();
        let activation_scale = 1.0 / self.dendrites as f32;

        for hidden_z in 0..self.hidden_size.z {
            let dendritic_start =
                flat_index!([self.hidden_size.z, self.dendrites], [hidden_z, 0]);
            let dendritic_end = dendritic_start + self.dendrites;
            let dendritic_cell = &mut dendrite_col[dendritic_start..dendritic_end];

            let mut cell_activation_raw = 0.0;
            for d in 0..self.dendrites {
                let da = &mut dendritic_cell[d];
                let non_linear = (*da).max(*da * LRELU_FACTOR);
                *da = non_linear * dendrite_scale;

                let val = if d >= self.half_dendrites {
                    *da
                } else {
                    -*da
                };
                cell_activation_raw += val;
            }

            let cell_activation = cell_activation_raw * activation_scale;

            yield_activation(hidden_z, cell_activation);
        }
    }

    pub fn forward(&self, visible_layers: &[&[u16]]) -> (Box<[u16]>, DecoderLearningData) {
        assert!(!visible_layers.is_empty());
        assert!(visible_layers.len() <= self.num_visible_layers);

        for layer in visible_layers {
            assert_eq!(layer.len(), self.visible_size.cols);
        }

        let mut hidden: Box<[u16]> = vec![0; self.hidden_size.cols].into_boxed_slice();
        let mut dendrite_activations: Box<[f32]> =
            vec![0.0; self.hidden_size.flat * self.dendrites].into_boxed_slice();
        let mut activations: Box<[f32]> = vec![0.0; self.hidden_size.flat].into_boxed_slice();

        let dendrite_activations_stride = self.hidden_size.z * self.dendrites;

        hidden
            .par_iter_mut()
            .zip(dendrite_activations.par_chunks_mut(dendrite_activations_stride))
            .zip(activations.par_chunks_mut(self.hidden_size.z))
            .enumerate()
            .for_each(
                |(hidden_col, ((hidden_cell, dendrite_col), activation_col))| {
                    self.compute_activations(
                        visible_layers,
                        hidden_col,
                        dendrite_col,
                        |hidden_z, cell_activation| activation_col[hidden_z] = cell_activation,
                    );

                    softmax(activation_col);
                    *hidden_cell = column_wise_one_hot(activation_col);
                },
            );

        (
            hidden,
            DecoderLearningData {
                concat: visible_layers.iter().map(|&l| l.into()).collect(),
                dendrite_activations,
                activations,
            },
        )
    }

    #[inline(always)]
    fn calc_weight_delta(
        half_dendrites: usize,
        lr: f32,
        dendrite: usize,
        dendritic_cell: &[f32],
        error: f32,
    ) -> i8 {
        let sign = if dendrite >= half_dendrites {
            1.0
        } else {
            -1.0
        };
        let delta = lr * sign * unit_step(dendritic_cell[dendrite]) * error;

        rand_round(delta) as i8
    }

    pub fn learn_flat(&mut self, expected: &[f32], data: &DecoderLearningData) {
        assert_eq!(expected.len(), data.activations.len());

        let expected_chunks = expected.par_chunks(self.hidden_size.z);
        self.learn_core(data, expected_chunks, |exp_chunk, hidden_z| {
            exp_chunk[hidden_z]
        });
    }

    pub fn learn(&mut self, expected: &[u16], data: &DecoderLearningData) {
        assert_eq!(expected.len(), self.hidden_size.cols);

        let expected_items = expected.par_iter();
        self.learn_core(data, expected_items, |&exp_z, hidden_z| {
            f32::from(hidden_z == exp_z as usize)
        });
    }

    fn learn_core<T, F>(
        &mut self,
        data: &'a DecoderLearningData,
        expected_iter: impl IndexedParallelIterator<Item = T>,
        get_target: F,
    ) where
        T: Send + Sync + Copy,
        F: Fn(T, usize) -> f32 + Sync + Send,
    {
        assert!(data.concat.len() <= self.num_visible_layers);
        assert!(!data.concat.is_empty());
        for layer in &data.concat {
            assert_eq!(layer.len(), self.visible_size.cols);
        }
        assert_eq!(data.activations.len(), self.hidden_size.flat);
        assert_eq!(
            data.dendrite_activations.len(),
            self.hidden_size.flat * self.dendrites
        );

        let dendrite_chunk_size = self.hidden_size.z * self.dendrites;
        let col_weights_len = self.num_visible_layers
            * self.visible_size.z
            * self.area
            * self.hidden_size.z
            * self.dendrites;

        self.weights
            .par_chunks_mut(col_weights_len)
            .zip(data.activations.par_chunks(self.hidden_size.z))
            .zip(data.dendrite_activations.par_chunks(dendrite_chunk_size))
            .zip(expected_iter)
            .enumerate()
            .for_each(
                |(hidden_col, (((col_weights, col_acts), col_dendrite_acts), exp_item))| {
                    let local_field = self.receptive_field.get_col(hidden_col);

                    let mut weight_deltas = vec![0; self.dendrites];

                    for hidden_z in 0..self.hidden_size.z {
                        let dendritic_start = hidden_z * self.dendrites;
                        let dendritic_end = dendritic_start + self.dendrites;
                        let dendritic_cell = &col_dendrite_acts[dendritic_start..dendritic_end];

                        let target = get_target(exp_item, hidden_z);
                        let error = target - col_acts[hidden_z];

                        for (dendrite, delta_slot) in weight_deltas.iter_mut().enumerate() {
                            *delta_slot = Decoder::calc_weight_delta(
                                self.half_dendrites,
                                self.lr,
                                dendrite,
                                dendritic_cell,
                                error,
                            );
                        }

                        for field in local_field {
                            let input_col = field.input_cell_idx as usize;

                            for (v_layer, layer_data) in data.concat.iter().enumerate() {
                                let concat_cell = layer_data[input_col] as usize;

                                let rel_weights_base = field.weights_base as usize
                                    - (hidden_col * col_weights_len);

                                let weights_start = rel_weights_base
                                    + flat_index!(
                                        [
                                            self.hidden_size.cols,
                                            self.num_visible_layers,
                                            self.visible_size.z,
                                            self.area,
                                            self.hidden_size.z,
                                            self.dendrites
                                        ],
                                        [0, v_layer, concat_cell, 0, hidden_z, 0]
                                    );

                                let weights_end = weights_start + self.dendrites;
                                let weights_cell = &mut col_weights[weights_start..weights_end];

                                for dendrite in 0..self.dendrites {
                                    weights_cell[dendrite] = weights_cell[dendrite]
                                        .saturating_add(weight_deltas[dendrite]);
                                }
                            }
                        }
                    }
                },
            );
    }

    pub fn get_snapshot(&'a self) -> DecoderSnapshot<'a> {
        DecoderSnapshot {
            visible_size: self.visible_size.clone(),
            hidden_size: self.hidden_size.clone(),
            num_visible_layers: self.num_visible_layers,
            half_dendrites: self.half_dendrites,
            radius: Some((self.area.isqrt() as i16 - 1) / 2),
            scale: self.scale,
            lr: self.lr,
            weights: Cow::Borrowed(&self.weights),
        }
    }

    pub fn from_snapshot(snapshot: DecoderSnapshot) -> Option<Self> {
        let dendrites = snapshot.half_dendrites * 2;
        let diameter = snapshot.radius? * 2 + 1;
        let area = (diameter * diameter) as usize;

        let receptive_field = Decoder::init_receptive_field(
            &snapshot.hidden_size,
            &snapshot.visible_size,
            snapshot.num_visible_layers,
            dendrites,
            snapshot.radius?,
        );

        Some(Self {
            visible_size: snapshot.visible_size,
            hidden_size: snapshot.hidden_size,
            num_visible_layers: snapshot.num_visible_layers,
            area,
            receptive_field,
            half_dendrites: snapshot.half_dendrites,
            dendrites,
            lr: snapshot.lr,
            weights: snapshot.weights.into_owned().into_boxed_slice(),
            scale: snapshot.scale,
        })
    }
}

#[derive(Debug)]
pub struct Head {
    pub(crate) visible_size: CsdrSize,
    pub(crate) hidden_size: CsdrSize,
    pub(crate) num_visible_layers: usize,
    half_dendrites: usize,
    dendrites: usize,
    pub lr: f32,
    weights: Box<[i8]>,
    scale: f32,
}

impl<'a> Head {
    pub fn new(
        visible_size: CsdrSize,
        hidden_size: CsdrSize,
        num_visible_layers: usize,
        half_dendrites: usize,
        scale: f32,
        lr: f32,
    ) -> Self {
        let mut rng = fastrand::Rng::new();
        let dendrites = half_dendrites * 2;

        let total_weights = hidden_size.flat
            * num_visible_layers
            * visible_size.cols
            * visible_size.z
            * dendrites;

        Self {
            visible_size,
            hidden_size,
            num_visible_layers,
            half_dendrites,
            dendrites,
            lr,
            weights: std::iter::repeat_with(|| rng.i8(-13..=12))
                .take(total_weights)
                .collect(),
            scale: scale / 127.0,
        }
    }

    #[inline(always)]
    fn get_weight_idx(
        &self,
        hidden_col: usize,
        hidden_z: usize,
        v_layer: usize,
        concat_col: usize,
        concat_z: usize,
        dendrite: usize,
    ) -> usize {
        flat_index!(
            [
                self.hidden_size.cols,
                self.hidden_size.z,
                self.num_visible_layers,
                self.visible_size.cols,
                self.visible_size.z,
                self.dendrites
            ],
            [
                hidden_col,
                hidden_z,
                v_layer,
                concat_col,
                concat_z,
                dendrite
            ]
        )
    }

    #[inline(always)]
    fn accumulate_dendrite_activations(
        &self,
        hidden_col: usize,
        visible_layers: &[&[u16]],
        mask_col: &[bool],
        hidden_dendritic_col_base: usize,
        dendrite_activations: &mut [f32],
    ) {
        let base_start = hidden_dendritic_col_base;
        let base_end = base_start + self.hidden_size.z * self.dendrites;
        let target_slice = &mut dendrite_activations[base_start..base_end];

        target_slice
            .par_chunks_mut(self.dendrites)
            .enumerate()
            .filter(|&(hidden_z, _)| mask_col[hidden_z])
            .for_each(|(hidden_z, dendritic_activations_cell)| {
                for (v_layer, layer_data) in visible_layers.iter().enumerate() {
                    for concat_col in 0..self.visible_size.cols {
                        let concat_z = layer_data[concat_col] as usize;

                        debug_assert!(concat_z < self.visible_size.z);

                        let weights_start = self.get_weight_idx(
                            hidden_col,
                            hidden_z,
                            v_layer,
                            concat_col,
                            concat_z,
                            0,
                        );
                        let weights_end = weights_start + self.dendrites;
                        let weights_cell = &self.weights[weights_start..weights_end];

                        for dendrite in 0..self.dendrites {
                            dendritic_activations_cell[dendrite] += weights_cell[dendrite] as f32;
                        }
                    }
                }
            });
    }

    pub(crate) fn compute_activations(
        &self,
        visible_layers: &[&[u16]],
        hidden_col: usize,
        mask_col: &[bool],
        dendrite_activations: &mut [f32],
        activation_col: &mut [f32],
    ) {
        let hidden_activation_col_base =
            flat_index!([self.hidden_size.cols, self.hidden_size.z], [hidden_col, 0]);
        let hidden_dendritic_col_base = flat_index!(
            [self.hidden_size.flat, self.dendrites],
            [hidden_activation_col_base, 0]
        );

        self.accumulate_dendrite_activations(
            hidden_col,
            visible_layers,
            mask_col,
            hidden_dendritic_col_base,
            dendrite_activations,
        );

        let fan_in = (self.visible_size.cols * self.num_visible_layers) as f32;
        let dendrite_scale = (self.scale / fan_in).sqrt();
        let activation_scale = 1.0 / self.dendrites as f32;

        let base_start = hidden_dendritic_col_base;
        let base_end = base_start + self.hidden_size.z * self.dendrites;
        let col_dendrites = &mut dendrite_activations[base_start..base_end];

        col_dendrites
            .par_chunks_mut(self.dendrites)
            .zip(activation_col.par_iter_mut())
            .zip(mask_col)
            .for_each(|((dendritic_cell, cell_activation), is_active)| {
                if !is_active {
                    *cell_activation = f32::NEG_INFINITY;
                    return;
                }

                let mut cell_activation_raw = 0.0;

                for d in 0..self.dendrites {
                    let da = &mut dendritic_cell[d];
                    let non_linear = (*da).max(*da * LRELU_FACTOR);
                    *da = non_linear * dendrite_scale;

                    let val = if d >= self.half_dendrites {
                        *da
                    } else {
                        -*da
                    };
                    cell_activation_raw += val;
                }

                *cell_activation = cell_activation_raw * activation_scale;
            });
    }

    pub fn forward(
        &self,
        visible_layers: &[&[u16]],
        mask: &[bool],
    ) -> (Box<[u16]>, DecoderLearningData) {
        assert_eq!(visible_layers.len(), self.num_visible_layers);
        for layer in visible_layers {
            assert_eq!(layer.len(), self.visible_size.cols);
        }
        assert_eq!(mask.len(), self.hidden_size.flat);

        let mut hidden: Box<[u16]> = vec![0; self.hidden_size.cols].into_boxed_slice();
        let mut dendrite_activations: Box<[f32]> =
            vec![0.0; self.hidden_size.flat * self.dendrites].into_boxed_slice();
        let mut activations: Box<[f32]> = vec![0.0; self.hidden_size.flat].into_boxed_slice();

        for hidden_col in 0..self.hidden_size.cols {
            let hidden_activation_col_base =
                flat_index!([self.hidden_size.cols, self.hidden_size.z], [hidden_col, 0]);
            let hidden_activation_col_end = hidden_activation_col_base + self.hidden_size.z;

            let mask_col = &mask[hidden_activation_col_base..hidden_activation_col_end];
            let activation_col =
                &mut activations[hidden_activation_col_base..hidden_activation_col_end];

            self.compute_activations(
                visible_layers,
                hidden_col,
                mask_col,
                &mut dendrite_activations,
                activation_col,
            );

            softmax(activation_col);
            hidden[hidden_col] = column_wise_one_hot(activation_col);
        }

        (
            hidden,
            DecoderLearningData {
                concat: visible_layers.iter().map(|l| (*l).into()).collect(),
                dendrite_activations,
                activations,
            },
        )
    }

    pub fn learn(&mut self, expected: &[f32], data: &'a DecoderLearningData) {
        assert_eq!(expected.len(), data.activations.len());
        assert_eq!(data.concat.len(), self.num_visible_layers);
        for layer in &data.concat {
            assert_eq!(layer.len(), self.visible_size.cols);
        }
        assert_eq!(data.activations.len(), self.hidden_size.flat);
        assert_eq!(
            data.dendrite_activations.len(),
            self.hidden_size.flat * self.dendrites
        );

        let z_weights_len =
            self.num_visible_layers * self.visible_size.flat * self.dendrites;

        self.weights
            .par_chunks_mut(z_weights_len)
            .zip(data.activations.par_iter())
            .zip(data.dendrite_activations.par_chunks(self.dendrites))
            .zip(expected.par_iter())
            .for_each(|(((z_weights, &act), dendritic_cell), target)| {
                let error = target - act;

                if act == f32::NEG_INFINITY || error.abs() < f32::EPSILON {
                    return;
                }

                let mut weight_deltas = vec![0; self.dendrites];
                for (dendrite, delta_slot) in weight_deltas.iter_mut().enumerate() {
                    *delta_slot = Decoder::calc_weight_delta(
                        self.half_dendrites,
                        self.lr,
                        dendrite,
                        dendritic_cell,
                        error,
                    );
                }

                for (v_layer, layer_data) in data.concat.iter().enumerate() {
                    for concat_col in 0..self.visible_size.cols {
                        let concat_z = layer_data[concat_col] as usize;

                        let rel_start = flat_index!(
                            [
                                self.num_visible_layers,
                                self.visible_size.cols,
                                self.visible_size.z,
                                self.dendrites
                            ],
                            [v_layer, concat_col, concat_z, 0]
                        );

                        let weights_start = rel_start;
                        let weights_end = weights_start + self.dendrites;
                        let weights_cell = &mut z_weights[weights_start..weights_end];

                        for dendrite in 0..self.dendrites {
                            weights_cell[dendrite] = weights_cell[dendrite]
                                .saturating_add(weight_deltas[dendrite]);
                        }
                    }
                }
            });
    }

    pub fn get_snapshot(&'a self) -> DecoderSnapshot<'a> {
        DecoderSnapshot {
            visible_size: self.visible_size.clone(),
            hidden_size: self.hidden_size.clone(),
            num_visible_layers: self.num_visible_layers,
            half_dendrites: self.half_dendrites,
            radius: None,
            scale: self.scale,
            lr: self.lr,
            weights: Cow::Borrowed(&self.weights),
        }
    }

    pub fn from_snapshot(snapshot: DecoderSnapshot) -> Self {
        let dendrites = snapshot.half_dendrites * 2;

        Self {
            visible_size: snapshot.visible_size,
            hidden_size: snapshot.hidden_size,
            num_visible_layers: snapshot.num_visible_layers,
            half_dendrites: snapshot.half_dendrites,
            dendrites,
            lr: snapshot.lr,
            weights: snapshot.weights.into_owned().into_boxed_slice(),
            scale: snapshot.scale,
        }
    }
}
