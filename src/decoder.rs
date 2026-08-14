use crate::{
    coder::{
        CsdrSize, FieldBounds, FieldEntry, ReceptiveField, column_wise_one_hot, rand_round, softmax,
    },
    flat_index,
};

type LocalField = ReceptiveField<FieldEntry>;

fn unit_step(x: f32) -> f32 {
    if x <= 0.0 { 0.0 } else { 1.0 }
}

#[derive(Debug, Clone, Default)]
pub struct DecoderLearningData {
    pub(crate) concat: Box<[u16]>,
    pub(crate) dendrite_activations: Box<[f32]>,
    pub(crate) activations: Box<[f32]>,
}

#[derive(Debug)]
pub struct Decoder {
    pub(crate) visible_size: CsdrSize,
    pub(crate) hidden_size: CsdrSize,
    area: usize,
    receptive_field: LocalField,
    half_dendrites: usize,
    dendrites: usize,
    lr: f32,
    weights: Box<[i8]>,
    scale: f32,
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

        let receptive_field =
            Decoder::init_receptive_field(&hidden_size, &visible_size, dendrites, radius);

        Self {
            visible_size,
            hidden_size,
            area,
            receptive_field,
            half_dendrites,
            dendrites,
            lr,
            weights: std::iter::repeat_with(|| rng.i8(-12..=12))
                .take(visible_size.z * hidden_size.cols * area * hidden_size.z * dendrites)
                .collect(),
            scale: scale / 127.0,
        }
    }

    fn init_receptive_field(
        hidden_size: &CsdrSize,
        visible_size: &CsdrSize,
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

        ReceptiveField {
            lut: local_field_lut.into_boxed_slice(),
            offsets: local_field_offsets.into_boxed_slice(),
        }
    }

    #[inline(always)]
    fn get_weight_idx(
        &self,
        concat_cell: usize,
        hidden_col: usize,
        area_idx: usize,
        hidden_z: usize,
        dendrite: usize,
    ) -> usize {
        flat_index!(
            [
                self.visible_size.z,
                self.hidden_size.cols,
                self.area,
                self.hidden_size.z,
                self.dendrites
            ],
            [concat_cell, hidden_col, area_idx, hidden_z, dendrite]
        )
    }

    #[inline(always)]
    fn accumulate_dendrite_activations(
        &self,
        concat: &[u16],
        hidden_dendritic_col_base: usize,
        local_field: &[FieldEntry],
        dendrite_activations: &mut [f32],
    ) {
        for field in local_field {
            let concat_cell = concat[field.input_cell_idx as usize] as usize;

            let concat_cell_base =
                field.weights_base as usize + self.get_weight_idx(concat_cell, 0, 0, 0, 0);

            for hidden_z in 0..self.hidden_size.z {
                let weights_start = concat_cell_base + self.get_weight_idx(0, 0, 0, hidden_z, 0);
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
                    dendritic_activations_cell[dendrite] += weights_cell[dendrite] as f32;
                }
            }
        }
    }

    #[inline(always)]
    pub(crate) fn compute_activations<F>(
        &self,
        concat: &[u16],
        hidden_col: usize,
        dendrite_activations: &mut [f32],
        mut yield_activation: F,
    ) where
        F: FnMut(usize, f32), // (hidden_z, cell_activation)
    {
        let local_field = self.receptive_field.get_col(hidden_col);
        let hidden_activation_col_base =
            flat_index!([self.hidden_size.cols, self.hidden_size.z], [hidden_col, 0]);
        let hidden_dendritic_col_base = flat_index!(
            [self.hidden_size.flat, self.dendrites],
            [hidden_activation_col_base, 0]
        );

        self.accumulate_dendrite_activations(
            concat,
            hidden_dendritic_col_base,
            local_field,
            dendrite_activations,
        );

        let field_count = local_field.len() as f32;
        let dendrite_scale = (self.scale / field_count).sqrt();
        let activation_scale = 1.0 / self.dendrites as f32;

        for hidden_z in 0..self.hidden_size.z {
            let dendritic_start = hidden_dendritic_col_base
                + flat_index!(
                    [self.hidden_size.cols, self.hidden_size.z, self.dendrites],
                    [0, hidden_z, 0]
                );
            let dendritic_end = dendritic_start + self.dendrites;
            let dendritic_cell = &mut dendrite_activations[dendritic_start..dendritic_end];

            let mut cell_activation_raw = 0.0;
            for d in 0..self.dendrites {
                let da = &mut dendritic_cell[d];
                let non_linear = (*da).max(0.0); // ReLU
                *da = non_linear * dendrite_scale;

                let val = if d >= self.half_dendrites { *da } else { -*da };
                cell_activation_raw += val;
            }

            let cell_activation = cell_activation_raw * activation_scale;

            yield_activation(hidden_z, cell_activation);
        }
    }

    pub fn forward(&self, concat: &[u16]) -> (Box<[u16]>, DecoderLearningData) {
        assert_eq!(concat.len(), self.visible_size.cols);

        let mut hidden: Box<[u16]> = vec![0; self.hidden_size.cols].into_boxed_slice();
        let mut dendrite_activations: Box<[f32]> =
            vec![0.0; self.hidden_size.flat * self.dendrites].into_boxed_slice();
        let mut activations: Box<[f32]> = vec![0.0; self.hidden_size.flat].into_boxed_slice();

        for hidden_col in 0..self.hidden_size.cols {
            let hidden_activation_col_base =
                flat_index!([self.hidden_size.cols, self.hidden_size.z], [hidden_col, 0]);

            let activation_col = &mut activations
                [hidden_activation_col_base..(hidden_activation_col_base + self.hidden_size.z)];

            self.compute_activations(
                concat,
                hidden_col,
                &mut dendrite_activations,
                |hidden_z, cell_activation| activation_col[hidden_z] = cell_activation,
            );

            softmax(activation_col);
            hidden[hidden_col] = column_wise_one_hot(activation_col);
        }

        (
            hidden,
            DecoderLearningData {
                concat: concat.into(),
                dendrite_activations,
                activations,
            },
        )
    }

    #[inline(always)]
    fn calc_weight_delta(&self, dendrite: usize, dendritic_cell: &[f32], error: f32) -> i8 {
        let sign = if dendrite >= self.half_dendrites {
            1.0
        } else {
            -1.0
        };
        let delta = self.lr * sign * unit_step(dendritic_cell[dendrite]) * error;

        rand_round(delta) as i8
    }

    pub fn learn(
        &mut self,
        expected: &[u16], // Must be at current time step
        DecoderLearningData {
            concat,
            dendrite_activations,
            activations,
        }: &DecoderLearningData, // Must be at previous time step
    ) {
        assert_eq!(expected.len(), self.hidden_size.cols);
        assert_eq!(concat.len(), self.visible_size.cols);
        assert_eq!(activations.len(), self.hidden_size.flat);
        assert_eq!(
            dendrite_activations.len(),
            self.hidden_size.flat * self.dendrites
        );

        let mut weight_deltas = vec![0; self.dendrites].into_boxed_slice();

        for hidden_col in 0..self.hidden_size.cols {
            let local_field = self.receptive_field.get_col(hidden_col);

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
                    weight_deltas[dendrite] =
                        self.calc_weight_delta(dendrite, dendritic_cell, error);
                }

                for field in local_field {
                    let concat_col = field.input_cell_idx as usize;
                    let concat_cell = concat[concat_col] as usize;

                    let weights_start = field.weights_base as usize
                        + self.get_weight_idx(concat_cell, 0, 0, hidden_z, 0);
                    let weights_end = weights_start + self.dendrites;
                    let weights_cell = &mut self.weights[weights_start..weights_end];

                    for dendrite in 0..self.dendrites {
                        weights_cell[dendrite] =
                            weights_cell[dendrite].saturating_add(weight_deltas[dendrite]);
                    }
                }
            }
        }
    }
}
