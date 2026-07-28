use crate::{
    flat_index,
    model::coder::{CsdrSize, FieldBounds},
};

const BYTE_INV: f32 = 1.0 / 255.0;

#[derive(Debug)]
struct LocalField {
    input_cell_idx: u32,
    weights_base: u32,
}

#[derive(Debug, Clone, Default)]
pub struct EncoderLearningData {
    max_activations: Box<[f32]>,
    learn_flags: Box<[bool]>,
}

#[derive(Debug)]
#[repr(align(64))]
pub struct Encoder {
    pub(crate) visible_size: CsdrSize,
    pub(crate) hidden_size: CsdrSize,

    area: usize,

    local_field_lut: Box<[LocalField]>,
    local_field_offsets: Box<[(u32, u32)]>,

    learning_field_lut: Box<[u32]>,
    learning_field_offsets: Box<[(u32, u32)]>,

    weight_deltas: [u8; 256],

    hidden_sum: Box<[u32]>,
    hidden_totals: Box<[u16]>,
    is_commited: Box<[bool]>,

    choice: f32,
    vigilance: f32,
    active_ratio: f32,

    weights: Box<[u8]>,
}

impl Encoder {
    pub fn new(
        visible_size: CsdrSize,
        hidden_size: CsdrSize,
        radius: i16,
        learning_radius: isize,
        lr: f32,
        choice: f32,
        vigilance: f32,
        active_ratio: f32,
    ) -> Self {
        let diameter = radius * 2 + 1;
        let area = (diameter * diameter) as usize;
        let (learning_field_lut, learning_field_offsets) =
            Encoder::init_learning_field_lut(&hidden_size, learning_radius);
        let (local_field_lut, local_field_offsets) =
            Encoder::init_local_field_lut(&hidden_size, &visible_size, radius);

        let mut rng = fastrand::Rng::new();

        Self {
            visible_size,
            hidden_size,

            area,

            local_field_lut,
            local_field_offsets,

            learning_field_lut,
            learning_field_offsets,

            weight_deltas: std::array::from_fn(|w| {
                (w as u8).saturating_add((lr * (255.0 - w as f32)).ceil() as u8)
            }),

            hidden_sum: vec![0; hidden_size.flat].into_boxed_slice(),
            hidden_totals: vec![0; hidden_size.flat].into_boxed_slice(),
            is_commited: vec![false; hidden_size.flat].into_boxed_slice(),

            choice,
            vigilance,
            active_ratio,

            weights: std::iter::repeat_with(|| rng.u8(0..=8))
                .take(visible_size.z * hidden_size.cols * area * hidden_size.z)
                .collect(),
        }
    }

    fn init_learning_field_lut(
        hidden_size: &CsdrSize,
        learning_radius: isize,
    ) -> (Box<[u32]>, Box<[(u32, u32)]>) {
        let diameter = (2 * learning_radius + 1) as usize;
        let area = diameter * diameter;
        let mut learning_field_lut = Vec::with_capacity(hidden_size.cols * area);
        let mut learning_field_offsets = Vec::with_capacity(hidden_size.cols);

        for hidden_col in 0..hidden_size.cols {
            let hidden_x = hidden_col % hidden_size.x;
            let hidden_y = hidden_col / hidden_size.x;

            let start_idx = learning_field_lut.len() as u32;

            for delta_y in -learning_radius..=learning_radius {
                let neighbor_y = hidden_y as isize + delta_y;
                if neighbor_y < 0 || neighbor_y >= hidden_size.y as isize {
                    continue;
                }

                for delta_x in -learning_radius..=learning_radius {
                    let neighbor_x = hidden_x as isize + delta_x;
                    if neighbor_x < 0
                        || neighbor_x >= hidden_size.x as isize
                        || (delta_x == 0 && delta_y == 0)
                    {
                        continue;
                    }

                    let neighbor_idx = flat_index!(
                        [hidden_size.y, hidden_size.x],
                        [neighbor_y as usize, neighbor_x as usize]
                    );

                    learning_field_lut.push(neighbor_idx as u32);
                }
            }

            let end = learning_field_lut.len() as u32;
            learning_field_offsets.push((start_idx, end));
        }

        (
            learning_field_lut.into_boxed_slice(),
            learning_field_offsets.into_boxed_slice(),
        )
    }

    fn init_local_field_lut(
        hidden_size: &CsdrSize,
        visible_size: &CsdrSize,
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
                        input_cell_idx: flat_index!(
                            [visible_size.y, visible_size.x],
                            [visible_y as usize, visible_x as usize]
                        ) as u32,
                        weights_base: flat_index!(
                            [visible_size.z, hidden_size.cols, area, hidden_size.z],
                            [0, hidden_col, in_field_idx, 0]
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

    pub fn forward(&mut self, input: &[u16]) -> (Box<[u16]>, EncoderLearningData) {
        assert_eq!(input.len() % self.visible_size.cols, 0);

        self.hidden_sum.fill(0);
        let mut hidden = vec![0; self.hidden_size.cols].into_boxed_slice();
        let mut cols_max_activation = vec![0.0; self.hidden_size.cols].into_boxed_slice();
        let mut cols_learn_flag = vec![false; self.hidden_size.cols].into_boxed_slice();

        for hidden_col in 0..self.hidden_size.cols {
            let sum_col_start =
                flat_index!([self.hidden_size.cols, self.hidden_size.z], [hidden_col, 0]);
            let sum_col_end = sum_col_start + self.hidden_size.z;
            let sum_col_range = sum_col_start..sum_col_end;
            let sum_col = &mut self.hidden_sum[sum_col_range.clone()];
            let total_col = &self.hidden_totals[sum_col_range.clone()];
            let commited_col = &self.is_commited[sum_col_range];

            let (start, end) = self.local_field_offsets[hidden_col];
            let (start, end) = (start as usize, end as usize);

            for field in &self.local_field_lut[start..end] {
                let input_cell = input[field.input_cell_idx as usize] as usize;
                assert!(input_cell < self.visible_size.z);

                let weights_start = field.weights_base as usize
                    + flat_index!(
                        [
                            self.visible_size.z,
                            self.hidden_size.cols,
                            self.area,
                            self.hidden_size.z
                        ],
                        [input_cell, 0, 0, 0]
                    ) as usize;
                let weights_end = weights_start + self.hidden_size.z;

                let weights_col = &self.weights[weights_start..weights_end];

                for cell in 0..self.hidden_size.z {
                    sum_col[cell] += weights_col[cell] as u32;
                }
            }

            let clamped_area = end - start;
            let count_all = clamped_area as f32 * self.visible_size.z as f32;
            let count_except = clamped_area as f32 * (self.visible_size.z - 1) as f32;
            let beta = self.choice + count_all;

            let mut max_activation = 0.0;
            let mut max_complete_activation = 0.0;

            let mut max_activation_cell = None;
            let mut max_complete_activation_cell = 0;

            for cell in 0..self.hidden_size.z {
                let sum = sum_col[cell] as f32 * BYTE_INV;
                let total = total_col[cell] as f32 * BYTE_INV;
                let complemented = sum - total + count_except;
                let match_score = complemented / count_except;
                let activation = complemented / (beta - total);

                if (!commited_col[cell] || match_score >= self.vigilance)
                    && activation > max_activation
                {
                    max_activation = activation;
                    max_activation_cell = Some(cell);
                }

                if activation > max_complete_activation {
                    max_complete_activation = activation;
                    max_complete_activation_cell = cell;
                }
            }

            match max_activation_cell {
                None => {
                    hidden[hidden_col] = max_complete_activation_cell as u16;
                    cols_max_activation[hidden_col] = max_complete_activation;
                    cols_learn_flag[hidden_col] = false;
                }
                Some(cell) => {
                    hidden[hidden_col] = cell as u16;
                    cols_max_activation[hidden_col] = max_activation;
                    cols_learn_flag[hidden_col] = true;
                }
            }
        }

        (
            hidden,
            EncoderLearningData {
                max_activations: cols_max_activation,
                learn_flags: cols_learn_flag,
            },
        )
    }

    #[inline]
    fn can_col_learn(&self, hidden_col: usize, learning_data: &EncoderLearningData) -> bool {
        if !learning_data.learn_flags[hidden_col] {
            return false;
        }

        let (start, end) = self.learning_field_offsets[hidden_col];
        let (start, end) = (start as usize, end as usize);
        let learning_field = &self.learning_field_lut[start..end];

        let center_activation = learning_data.max_activations[hidden_col];

        let field_cnt = end - start;
        let mut higher_neighbors = 0;

        for &neighbor_idx in learning_field {
            higher_neighbors +=
                (learning_data.max_activations[neighbor_idx as usize] > center_activation) as usize;
        }

        higher_neighbors as f32 <= self.active_ratio * field_cnt as f32
    }

    pub fn learn(&mut self, input: &[u16], hidden: &[u16], learning_data: &EncoderLearningData) {
        assert_eq!(input.len() % self.visible_size.cols, 0);

        for hidden_col in 0..self.hidden_size.cols {
            if !self.can_col_learn(hidden_col, learning_data) {
                continue;
            }

            let hidden_z = hidden[hidden_col] as usize;
            let hidden_idx = flat_index!(
                [self.hidden_size.cols, self.hidden_size.z],
                [hidden_col, hidden_z]
            );

            let is_commited = self.is_commited[hidden_idx];

            let (start, end) = self.local_field_offsets[hidden_col];
            let (start, end) = (start as usize, end as usize);

            for field in &self.local_field_lut[start..end] {
                let input_cell = input[field.input_cell_idx as usize] as usize;
                let weights_idx = field.weights_base as usize
                    + flat_index!(
                        [
                            self.visible_size.z,
                            self.hidden_size.cols,
                            self.area,
                            self.hidden_size.z
                        ],
                        [input_cell, 0, 0, hidden_z]
                    ) as usize;

                let old = self.weights[weights_idx];
                self.weights[weights_idx] = if is_commited {
                    self.weight_deltas[old as usize]
                } else {
                    255
                };
                self.hidden_totals[hidden_idx] += (self.weights[weights_idx] - old) as u16;
            }

            self.is_commited[hidden_idx] = true;
        }
    }
}
