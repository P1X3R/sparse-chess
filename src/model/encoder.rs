use crate::model::coder::{Coder, CsdrSize, column_wise_one_hot};
use rand::{RngExt, rngs::SmallRng};

#[derive(Debug)]
pub struct Encoder {
    visible_size: CsdrSize,
    hidden_size: CsdrSize,
    ratio: (f32, f32),

    area: usize,
    radius: usize,

    lr: u8,
    dictionary: Vec<i8>,
}

impl Encoder {
    pub fn new(
        visible_size: CsdrSize,
        hidden_size: CsdrSize,
        radius: usize,
        lr: u8,
        rng: &mut SmallRng,
    ) -> Self {
        let diameter = radius * 2 + 1;
        let area = diameter * diameter;

        Self {
            visible_size,
            hidden_size,
            ratio: (
                visible_size.x as f32 / hidden_size.x as f32,
                visible_size.y as f32 / hidden_size.y as f32,
            ),

            area,
            radius,

            lr,
            dictionary: (0..hidden_size.flat * area * visible_size.z)
                .map(|_| rng.random_range(-2..=-1))
                .collect(),
        }
    }
}

impl Coder for Encoder {
    fn forward(&self, input: Vec<usize>) -> Vec<usize> {
        assert_eq!(input.len(), self.visible_size.cols);

        let mut activations: Vec<i8> = vec![0; self.hidden_size.flat];

        for hidden_col in 0..self.hidden_size.cols {
            let hidden_col_x = hidden_col % self.hidden_size.x;
            let hidden_col_y = hidden_col / self.hidden_size.x;

            let visible_center = (
                ((hidden_col_x as f32 + 0.5f32) * self.ratio.0) as isize,
                ((hidden_col_y as f32 + 0.5f32) * self.ratio.1) as isize,
            );

            let field_start = (
                visible_center.0 - self.radius as isize,
                visible_center.1 - self.radius as isize,
            );

            let field_end = (
                visible_center.0 as isize + self.radius as isize,
                visible_center.1 as isize + self.radius as isize,
            );

            let clamped_start = (field_start.0.max(0), field_start.1.max(0));
            let clamped_end = (
                field_end.0.min(self.visible_size.x as isize - 1),
                field_end.1.min(self.visible_size.y as isize - 1),
            );

            for visible_x in clamped_start.0..=clamped_end.0 {
                for visible_y in clamped_start.1..=clamped_end.1 {
                    let in_field_x = visible_x - field_start.0 as isize;
                    let in_field_y = visible_y - field_start.1 as isize;

                    let diameter = 2 * self.radius as isize + 1;
                    let in_field_idx = (in_field_x + (diameter * in_field_y)) as usize;
                    let input_cell =
                        input[(visible_x + (self.visible_size.x as isize * visible_y)) as usize];

                    for hidden_cell in 0..self.hidden_size.z {
                        let hidden_idx = (hidden_col * self.hidden_size.z) + hidden_cell;

                        let dictionary_idx = input_cell
                            + (in_field_idx * self.visible_size.z)
                            + (hidden_idx * self.visible_size.z * self.area);

                        activations[hidden_idx] =
                            activations[hidden_idx].saturating_add(self.dictionary[dictionary_idx]);
                    }
                }
            }
        }

        let hidden = column_wise_one_hot(activations, self.hidden_size.z);
        assert_eq!(hidden.len(), self.hidden_size.cols);

        hidden
    }

    fn learn(&mut self) {}
}
