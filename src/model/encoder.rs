use crate::{flat_index, model::coder::CsdrSize};
use rand::{RngExt, rngs::SmallRng};

#[derive(Debug)]
struct FieldBounds {
    field_start_x: i16,
    field_start_y: i16,

    clamped_start_x: i16,
    clamped_start_y: i16,

    clamped_end_x: i16,
    clamped_end_y: i16,
}

impl FieldBounds {
    #[inline]
    fn new(
        hidden_x: usize,
        hidden_y: usize,
        radius: i16,
        ratio: (f32, f32),
        visible_size: &CsdrSize,
    ) -> Self {
        let visible_center_x = ((hidden_x as f32 + 0.5f32) * ratio.0) as i16;
        let visible_center_y = ((hidden_y as f32 + 0.5f32) * ratio.1) as i16;

        let field_start_x = visible_center_x - radius;
        let field_start_y = visible_center_y - radius;

        let field_end_x = visible_center_x + radius;
        let field_end_y = visible_center_y + radius;

        let clamped_start_x = field_start_x.max(0);
        let clamped_start_y = field_start_y.max(0);

        let clamped_end_x = field_end_x.min(visible_size.x as i16 - 1);
        let clamped_end_y = field_end_y.min(visible_size.y as i16 - 1);

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

    area: usize,
    diameter: usize,
    field_bounds: Box<[FieldBounds]>,

    importances: Box<[f32]>,
    hidden_sum: Box<[u16]>,
    hidden_totals: Box<[u16]>,
    hidden: Box<[u16]>,
    is_commited: Box<[bool]>,

    hidden_max_activation: Box<[f32]>,
    hidden_learn_flag: Box<[bool]>,

    choice: f32,
    vigilance: f32,

    lr: f32,
    dictionary: Vec<u8>,
}

impl Encoder {
    pub fn new(
        visible_size: CsdrSize,
        hidden_size: CsdrSize,
        radius: usize,
        lr: f32,
        rng: &mut SmallRng,
    ) -> Self {
        let diameter = radius * 2 + 1;
        let area = diameter * diameter;
        let ratio = (
            visible_size.x as f32 / hidden_size.x as f32,
            visible_size.y as f32 / hidden_size.y as f32,
        );

        Self {
            visible_size,
            hidden_size,

            area,
            diameter,
            field_bounds: (0..hidden_size.cols)
                .map(|col| {
                    FieldBounds::new(
                        col % hidden_size.x,
                        col / hidden_size.x,
                        radius as i16,
                        ratio,
                        &visible_size,
                    )
                })
                .collect(),

            importances: vec![1.0; hidden_size.cols].into_boxed_slice(),
            hidden_sum: vec![0; hidden_size.flat].into_boxed_slice(),
            hidden_totals: vec![0; hidden_size.flat].into_boxed_slice(),
            hidden: vec![0; hidden_size.cols].into_boxed_slice(),
            is_commited: vec![false; hidden_size.flat].into_boxed_slice(),

            hidden_max_activation: vec![0.0; hidden_size.cols].into_boxed_slice(),
            hidden_learn_flag: vec![false; hidden_size.cols].into_boxed_slice(),

            choice: 0.01,
            vigilance: 0.9,

            lr,
            dictionary: (0..hidden_size.cols * area * visible_size.z * hidden_size.z)
                .map(|_| rng.random())
                .collect(),
        }
    }

    pub fn forward(&mut self, input: &[usize]) -> &[u16] {
        debug_assert_eq!(input.len(), self.visible_size.cols);

        self.hidden_sum.fill(0);
        self.hidden.fill(0);

        for hidden_col in 0..self.hidden_size.cols {
            let FieldBounds {
                field_start_x,
                field_start_y,

                clamped_start_x,
                clamped_start_y,

                clamped_end_x,
                clamped_end_y,
            } = self.field_bounds[hidden_col];

            let sum_col_start =
                flat_index!([self.hidden_size.cols, self.hidden_size.z], [hidden_col, 0]);
            let sum_col = &mut self.hidden_sum[sum_col_start..(sum_col_start + self.hidden_size.z)];
            let total_col =
                &self.hidden_totals[sum_col_start..(sum_col_start + self.hidden_size.z)];
            let commited_col =
                &self.is_commited[sum_col_start..(sum_col_start + self.hidden_size.z)];

            for visible_y in clamped_start_y..=clamped_end_y {
                let in_field_y = visible_y - field_start_y;

                for visible_x in clamped_start_x..=clamped_end_x {
                    let in_field_x = visible_x - field_start_x;

                    let in_field_idx = flat_index!(
                        [self.diameter, self.diameter],
                        [in_field_y as usize, in_field_x as usize]
                    );

                    let input_cell_idx = flat_index!(
                        [self.visible_size.y, self.visible_size.x],
                        [visible_y as usize, visible_x as usize]
                    );
                    let input_cell = input[input_cell_idx];

                    let dictionary_start = flat_index!(
                        [
                            self.hidden_size.cols,
                            self.area,
                            self.visible_size.z,
                            self.hidden_size.z
                        ],
                        [hidden_col, in_field_idx, input_cell, 0]
                    );

                    let dictionary_col =
                        &self.dictionary[dictionary_start..(dictionary_start + self.hidden_size.z)];

                    for (sum, &weight) in sum_col.iter_mut().zip(dictionary_col) {
                        *sum += weight as u16;
                    }
                }
            }

            let importance = self.importances[hidden_col];
            let scale = importance / 255.0;
            let clamped_area =
                (clamped_end_x - clamped_start_x + 1) * (clamped_end_y - clamped_start_y + 1);
            let count_all = importance * clamped_area as f32 * self.visible_size.z as f32;
            let count_except = importance * clamped_area as f32 * (self.visible_size.z - 1) as f32;
            let beta = self.choice + count_all;

            let mut max_activation = 0.0;
            let mut max_complete_activation = 0.0;

            let mut max_activation_cell = None;
            let mut max_complete_activation_cell = 0;

            for (cell, ((&sum_raw, &total_raw), is_commited)) in
                sum_col.iter().zip(total_col).zip(commited_col).enumerate()
            {
                let sum = sum_raw as f32 * scale;
                let total = total_raw as f32 * scale;
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

    pub fn learn(&mut self, expected: &[usize], hidden: &[usize]) {
        todo!()
    }
}
