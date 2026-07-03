use crate::{flat_index, model::coder::CsdrSize};
use rand::{Rng, rngs::SmallRng};

#[derive(Debug)]
struct FieldBounds {
    field_start_x: i16,
    field_start_y: i16,

    clamped_start_x: i16,
    clamped_start_y: i16,

    clamped_end_x: i16,
    clamped_end_y: i16,
}

#[derive(Debug)]
struct LocalField {
    input_cell_idx: u32,
    dictionary_base: u32,
}

impl FieldBounds {
    #[inline]
    fn new(
        x: usize,
        y: usize,
        radius: i16,
        ratio: (f32, f32),
        projected_csdr_size: &CsdrSize,
    ) -> Self {
        let visible_center_x = ((x as f32 + 0.5f32) * ratio.0) as i16;
        let visible_center_y = ((y as f32 + 0.5f32) * ratio.1) as i16;

        let field_start_x = visible_center_x - radius;
        let field_start_y = visible_center_y - radius;

        let field_end_x = visible_center_x + radius;
        let field_end_y = visible_center_y + radius;

        let clamped_start_x = field_start_x.max(0);
        let clamped_start_y = field_start_y.max(0);

        let clamped_end_x = field_end_x.min(projected_csdr_size.x as i16 - 1);
        let clamped_end_y = field_end_y.min(projected_csdr_size.y as i16 - 1);

        Self {
            field_start_x,
            field_start_y,
            clamped_start_x,
            clamped_start_y,
            clamped_end_x,
            clamped_end_y,
        }
    }
}

#[derive(Debug)]
pub struct Encoder {
    visible_size: CsdrSize,
    hidden_size: CsdrSize,

    local_field_lut: Box<[LocalField]>,
    local_field_offsets: Box<[(u32, u32)]>,

    learning_field_lut: Box<[u32]>,
    learning_field_offsets: Box<[(u32, u32)]>,

    hidden_sum: Box<[u16]>,
    hidden_totals: Box<[u16]>,
    hidden: Box<[u16]>,
    is_commited: Box<[bool]>,

    hidden_max_activation: Box<[f32]>,
    hidden_learn_flag: Box<[bool]>,

    choice: f32,
    vigilance: f32,
    active_ratio: f32,

    lr: f32,

    dictionary: Box<[u8]>,
}

impl Encoder {
    pub fn new(
        visible_size: CsdrSize,
        hidden_size: CsdrSize,
        radius: i16,
        learning_radius: isize,
        lr: f32,
        rng: &mut SmallRng,
    ) -> Self {
        let diameter = radius * 2 + 1;
        let area = (diameter * diameter) as usize;
        let (learning_field_lut, learning_field_offsets) =
            Encoder::init_learning_field_lut(&hidden_size, learning_radius);
        let (local_field_lut, local_field_offsets) =
            Encoder::init_local_field_lut(&hidden_size, &visible_size, radius);

        let mut dictionary: Box<[u8]> = unsafe {
            Box::new_uninit_slice(hidden_size.cols * area * visible_size.z * hidden_size.z)
                .assume_init()
        };
        rng.fill_bytes(&mut dictionary);

        Self {
            visible_size,
            hidden_size,

            local_field_lut,
            local_field_offsets,

            learning_field_lut,
            learning_field_offsets,

            hidden_sum: vec![0; hidden_size.flat].into_boxed_slice(),
            hidden_totals: vec![0; hidden_size.flat].into_boxed_slice(),
            hidden: vec![0; hidden_size.cols].into_boxed_slice(),
            is_commited: vec![false; hidden_size.flat].into_boxed_slice(),

            hidden_max_activation: vec![0.0; hidden_size.cols].into_boxed_slice(),
            hidden_learn_flag: vec![false; hidden_size.cols].into_boxed_slice(),

            choice: 0.01,
            vigilance: 0.9,
            active_ratio: 0.5,

            lr,
            dictionary,
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
        let mut local_field_lut = Vec::with_capacity(hidden_size.cols * area);
        let mut local_field_offsets = Vec::with_capacity(hidden_size.cols);

        let ratio = (
            visible_size.x as f32 / hidden_size.x as f32,
            visible_size.y as f32 / hidden_size.y as f32,
        );
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
                        dictionary_base: flat_index!(
                            [hidden_size.cols, area, visible_size.z, hidden_size.z],
                            [hidden_col, in_field_idx, 0, 0]
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

    pub fn forward(&mut self, input: &[u16]) -> &[u16] {
        debug_assert_eq!(input.len(), self.visible_size.cols);

        self.hidden_sum.fill(0);
        self.hidden.fill(0);

        for hidden_col in 0..self.hidden_size.cols {
            let sum_col_start =
                flat_index!([self.hidden_size.cols, self.hidden_size.z], [hidden_col, 0]);
            let sum_col = &mut self.hidden_sum[sum_col_start..(sum_col_start + self.hidden_size.z)];
            let total_col =
                &self.hidden_totals[sum_col_start..(sum_col_start + self.hidden_size.z)];
            let commited_col =
                &self.is_commited[sum_col_start..(sum_col_start + self.hidden_size.z)];

            let (start, end) = self.local_field_offsets[hidden_col];
            let (start, end) = (start as usize, end as usize);

            for field in &self.local_field_lut[start..end] {
                let input_cell = input[field.input_cell_idx as usize] as usize;

                let dictionary_start = field.dictionary_base as usize
                    + flat_index!([self.visible_size.z, self.hidden_size.z], [input_cell, 0]);

                let dictionary_col =
                    &self.dictionary[dictionary_start..(dictionary_start + self.hidden_size.z)];

                for (sum, &weight) in sum_col.iter_mut().zip(dictionary_col) {
                    *sum += weight as u16;
                }
            }

            let byte_inv = 1.0 / 255.0;
            let clamped_area = end - start + 1;
            let count_all = clamped_area as f32 * self.visible_size.z as f32;
            let count_except = clamped_area as f32 * (self.visible_size.z - 1) as f32;
            let beta = self.choice + count_all;

            let mut max_activation = 0.0;
            let mut max_complete_activation = 0.0;

            let mut max_activation_cell = None;
            let mut max_complete_activation_cell = 0;

            for (cell, ((&sum_raw, &total_raw), is_commited)) in
                sum_col.iter().zip(total_col).zip(commited_col).enumerate()
            {
                let sum = sum_raw as f32 * byte_inv;
                let total = total_raw as f32 * byte_inv;
                let complemented = sum - total + count_except;
                let match_score = complemented / count_except;
                let activation = complemented / (beta - total);

                if (!is_commited || match_score >= self.vigilance) && activation > max_activation {
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
                    self.hidden[hidden_col] = max_complete_activation_cell as u16;
                    self.hidden_max_activation[hidden_col] = max_complete_activation;
                    self.hidden_learn_flag[hidden_col] = false;
                }
                Some(cell) => {
                    self.hidden[hidden_col] = cell as u16;
                    self.hidden_max_activation[hidden_col] = max_activation;
                    self.hidden_learn_flag[hidden_col] = true;
                }
            }
        }

        &self.hidden
    }

    fn can_col_learn(&self, hidden_col: usize) -> bool {
        if !self.hidden_learn_flag[hidden_col] {
            return false;
        }

        let (start, end) = self.learning_field_offsets[hidden_col];
        let (start, end) = (start as usize, end as usize);
        let learning_field = &self.learning_field_lut[start..end];

        let higher_neighbors = learning_field
            .iter()
            .filter(|&&neighbor_idx| {
                self.hidden_max_activation[neighbor_idx as usize]
                    > self.hidden_max_activation[hidden_col]
            })
            .count();
        let field_cnt = end - start + 1;

        higher_neighbors as f32 <= self.active_ratio * field_cnt as f32
    }

    pub fn learn(&mut self, input: &[u16]) {
        for hidden_col in 0..self.hidden_size.cols {
            if !self.can_col_learn(hidden_col) {
                continue;
            }

            let hidden_z = self.hidden[hidden_col] as usize;
            let hidden_idx = flat_index!(
                [self.hidden_size.cols, self.hidden_size.z],
                [hidden_col, hidden_z]
            );
            let learning_rate = if self.is_commited[hidden_idx] {
                self.lr
            } else {
                1.0
            };

            let (start, end) = self.local_field_offsets[hidden_col];
            let (start, end) = (start as usize, end as usize);

            for field in &self.local_field_lut[start..end] {
                let input_cell = input[field.input_cell_idx as usize] as usize;

                let dictionary_idx = field.dictionary_base as usize
                    + flat_index!(
                        [self.visible_size.z, self.hidden_size.z],
                        [input_cell, self.hidden[hidden_col] as usize]
                    );

                let old = self.dictionary[dictionary_idx];
                self.dictionary[dictionary_idx] =
                    old.saturating_add((learning_rate * (255.0 - old as f32)).ceil() as u8);
                self.hidden_totals[hidden_idx] += (self.dictionary[dictionary_idx] - old) as u16;
            }

            self.is_commited[hidden_idx] = true;
        }
    }
}
