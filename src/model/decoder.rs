use crate::{
    flat_index,
    model::coder::{CsdrSize, FieldBounds, rand_round},
};

fn heavystep(x: i16) -> f32 {
    if x <= 0 { 0.0 } else { 1.0 }
}

#[derive(Debug)]
struct LocalField {
    concat_cell_idx: u32,
    weight_base: u32,
}

#[derive(Debug, Clone, Default)]
pub struct DecoderLearningData(pub Box<[u16]>, Box<[i16]>, Box<[f32]>);

#[derive(Debug)]
pub struct Decoder {
    pub(crate) visible_size: CsdrSize,
    pub(crate) hidden_size: CsdrSize,

    area: usize,

    local_field_lut: Box<[LocalField]>,
    local_field_offsets: Box<[(u32, u32)]>,

    half_dendrites: usize,
    dendrites: usize,

    lr: f32,

    weight_deltas: Box<[i8]>,
    weights: Box<[i8]>,

    scale: f32,

    rng: fastrand::Rng,
}

impl Decoder {
    pub fn new(
        visible_size: CsdrSize,
        hidden_size: CsdrSize,
        half_dendrites: usize,
        radius: i16,
        scale: f32,
        lr: f32,
    ) -> Self {
        let mut rng = fastrand::Rng::new();
        let dendrites = half_dendrites * 2;
        let diameter = radius * 2 + 1;
        let area = (diameter * diameter) as usize;

        let (local_field_lut, local_field_offsets) =
            Decoder::init_local_field_lut(&hidden_size, &visible_size, dendrites, radius);

        Self {
            visible_size,
            hidden_size,

            area,

            local_field_lut,
            local_field_offsets,

            half_dendrites: half_dendrites,
            dendrites: half_dendrites * 2,

            lr,

            weight_deltas: vec![0; dendrites].into_boxed_slice(),
            weights: std::iter::repeat_with(|| rng.i8(-12..=12))
                .take(visible_size.z * hidden_size.cols * area * hidden_size.z * dendrites)
                .collect(),

            scale: scale / 127.0,

            rng,
        }
    }

    fn init_local_field_lut(
        hidden_size: &CsdrSize,
        visible_size: &CsdrSize,
        dendrites: usize,
        radius: i16,
    ) -> (Box<[LocalField]>, Box<[(u32, u32)]>) {
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

                    local_field_lut.push(LocalField {
                        concat_cell_idx: flat_index!(
                            [visible_size.y, visible_size.x],
                            [visible_y as usize, visible_x as usize]
                        ) as u32,
                        weight_base: flat_index!(
                            [
                                visible_size.z,
                                hidden_size.cols,
                                area,
                                hidden_size.z,
                                dendrites
                            ],
                            [0, hidden_col, in_field_idx, 0, 0]
                        ) as u32,
                    });
                }
            }

            let end = local_field_lut.len() as u32;
            local_field_offsets.push((start_idx, end));
        }

        (
            local_field_lut.into_boxed_slice(),
            local_field_offsets.into_boxed_slice(),
        )
    }

    pub fn forward(&mut self, concat: &[u16]) -> (Box<[u16]>, DecoderLearningData) {
        assert_eq!(concat.len(), self.visible_size.cols);

        let mut hidden: Box<[u16]> = vec![0; self.hidden_size.cols].into_boxed_slice();
        let mut dendrite_activations: Box<[i16]> =
            vec![0; self.hidden_size.flat * self.dendrites].into_boxed_slice();
        let mut activations: Box<[f32]> = vec![0.0; self.hidden_size.flat].into_boxed_slice();

        for hidden_col in 0..self.hidden_size.cols {
            let (start, end) = self.local_field_offsets[hidden_col];
            let (start, end) = (start as usize, end as usize);
            let hidden_activation_col_base =
                flat_index!([self.hidden_size.cols, self.hidden_size.z], [hidden_col, 0]);
            let hidden_dendritic_col_base = flat_index!(
                [self.hidden_size.flat, self.dendrites],
                [hidden_activation_col_base, 0]
            );

            for field in &self.local_field_lut[start..end] {
                let concat_cell = concat[field.concat_cell_idx as usize] as usize;

                let concat_cell_base = field.weight_base as usize
                    + flat_index!(
                        [
                            self.visible_size.z,
                            self.hidden_size.cols,
                            self.area,
                            self.hidden_size.z,
                            self.dendrites
                        ],
                        [concat_cell, 0, 0, 0, 0]
                    );

                for hidden_z in 0..self.hidden_size.z {
                    let weights_start = concat_cell_base
                        + flat_index!(
                            [
                                self.visible_size.z,
                                self.hidden_size.cols,
                                self.area,
                                self.hidden_size.z,
                                self.dendrites
                            ],
                            [0, 0, 0, hidden_z, 0]
                        );
                    let weights_end = weights_start + self.dendrites;
                    let weights_cell = &self.weights[weights_start..weights_end];

                    let dendritic_start = hidden_dendritic_col_base
                        + flat_index!(
                            [self.hidden_size.cols, self.hidden_size.z, self.dendrites],
                            [0, hidden_z, 0]
                        );
                    let dendritic_end = dendritic_start + self.dendrites;
                    let dendritic_activations_cell =
                        &mut dendrite_activations[dendritic_start..dendritic_end];

                    for dendrite in 0..self.dendrites {
                        dendritic_activations_cell[dendrite] += weights_cell[dendrite] as i16;
                    }
                }
            }

            let mut max_activation = f32::MIN;
            let mut activation_sum = 0.0;
            let field_count = (end - start) as f32;
            let dendrite_scale = (1.0 / field_count) * self.scale;
            let activation_scale = 1.0 / self.dendrites as f32;
            let activation_col = &mut activations
                [hidden_activation_col_base..(hidden_activation_col_base + self.hidden_size.z)];

            for hidden_z in 0..self.hidden_size.z {
                let dendritic_start = hidden_dendritic_col_base
                    + flat_index!(
                        [self.hidden_size.cols, self.hidden_size.z, self.dendrites],
                        [0, hidden_z, 0]
                    );
                let dendritic_end = dendritic_start + self.dendrites;
                let dendritic_activations_cell =
                    &mut dendrite_activations[dendritic_start..dendritic_end];

                let cell_activation_raw: i16 = (0..self.dendrites)
                    .map(|dendrite| {
                        let da = &mut dendritic_activations_cell[dendrite];

                        let non_linear = (*da).max(0) as f32; // ReLU
                        *da = rand_round(non_linear * dendrite_scale, &mut self.rng) as i16;

                        if dendrite >= self.half_dendrites {
                            *da
                        } else {
                            -*da
                        }
                    })
                    .sum();
                let cell_activation = cell_activation_raw as f32 * activation_scale;

                if cell_activation > max_activation {
                    let shift = max_activation - cell_activation;
                    activation_sum = activation_sum * fast_math::exp(shift) + 1.0;
                    max_activation = cell_activation;
                } else {
                    let shift = cell_activation - max_activation;
                    activation_sum += fast_math::exp(shift);
                }

                activation_col[hidden_z] = cell_activation;
            }

            let activation_sum_inv = 1.0 / activation_sum;
            for hidden_z in 0..self.hidden_size.z {
                let shift = activation_col[hidden_z] - max_activation;
                activation_col[hidden_z] = fast_math::exp(shift) * activation_sum_inv;
            }

            let (max_idx, _) = activation_col
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();

            hidden[hidden_col] = max_idx as u16;
        }

        (
            hidden,
            DecoderLearningData(concat.into(), dendrite_activations, activations),
        )
    }

    pub fn learn(
        &mut self,
        expected: &[u16], // Must be at current time step
        DecoderLearningData(concat, dendrite_activations, activations): &DecoderLearningData, // Must be at previous time step
    ) {
        assert_eq!(expected.len(), self.hidden_size.cols);
        assert_eq!(concat.len(), self.visible_size.cols);
        assert_eq!(activations.len(), self.hidden_size.flat);
        assert_eq!(
            dendrite_activations.len(),
            self.hidden_size.flat * self.dendrites
        );

        for hidden_col in 0..self.hidden_size.cols {
            let (start, end) = self.local_field_offsets[hidden_col];
            let (start, end) = (start as usize, end as usize);

            for hidden_z in 0..self.hidden_size.z {
                let hidden_idx = flat_index!(
                    [self.hidden_size.cols, self.hidden_size.z],
                    [hidden_col, hidden_z]
                );
                let dendritic_start =
                    flat_index!([self.hidden_size.flat, self.dendrites], [hidden_idx, 0]);
                let dendritic_end = dendritic_start + self.dendrites;
                let dendritic_cell = &dendrite_activations[dendritic_start..dendritic_end];

                let activation = activations[hidden_idx];
                let input_cell_val = f32::from(hidden_z == expected[hidden_col] as usize);
                let error = input_cell_val - activation;

                for dendrite in 0..self.dendrites {
                    let sign = if dendrite >= self.half_dendrites {
                        1.0
                    } else {
                        -1.0
                    };
                    let delta = self.lr * sign * heavystep(dendritic_cell[dendrite]) * error;

                    self.weight_deltas[dendrite] =
                        rand_round(delta, &mut self.rng).max(-128.0).min(127.0) as i8;
                }

                for field in &self.local_field_lut[start..end] {
                    let concat_col = field.concat_cell_idx as usize;
                    let concat_cell = concat[concat_col] as usize;

                    let weights_start = field.weight_base as usize
                        + flat_index!(
                            [
                                self.visible_size.z,
                                self.hidden_size.cols,
                                self.area,
                                self.hidden_size.z,
                                self.dendrites
                            ],
                            [concat_cell, 0, 0, hidden_z, 0]
                        );
                    let weights_end = weights_start + self.dendrites;
                    let weights_cell = &mut self.weights[weights_start..weights_end];

                    for dendrite in 0..self.dendrites {
                        weights_cell[dendrite] += self.weight_deltas[dendrite];
                    }
                }
            }
        }
    }
}
