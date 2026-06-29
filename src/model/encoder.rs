use crate::model::coder::{CsdrSize, column_wise_one_hot};
use fast_math::exp_raw;
use rand::{
    distr::{Distribution, Uniform},
    rngs::SmallRng,
};

fn gen_field_bounds(
    radius: usize,
    ratio: (f32, f32),
    hidden_size: &CsdrSize,
    visible_size: &CsdrSize,
) -> (
    Vec<(isize, isize)>,
    Vec<(isize, isize)>,
    Vec<(isize, isize)>,
) {
    let mut field_start_lut: Vec<(isize, isize)> = Vec::with_capacity(hidden_size.cols);
    let mut clamped_start_lut: Vec<(isize, isize)> = Vec::with_capacity(hidden_size.cols);
    let mut clamped_end_lut: Vec<(isize, isize)> = Vec::with_capacity(hidden_size.cols);

    for hidden_col in 0..hidden_size.cols {
        let hidden_col_x = hidden_col % hidden_size.x;
        let hidden_col_y = hidden_col / hidden_size.x;

        let visible_center = (
            ((hidden_col_x as f32 + 0.5f32) * ratio.0) as isize,
            ((hidden_col_y as f32 + 0.5f32) * ratio.1) as isize,
        );

        let field_start = (
            visible_center.0 - radius as isize,
            visible_center.1 - radius as isize,
        );

        let field_end = (
            visible_center.0 as isize + radius as isize,
            visible_center.1 as isize + radius as isize,
        );

        let clamped_start = (field_start.0.max(0), field_start.1.max(0));
        let clamped_end = (
            field_end.0.min(visible_size.x as isize - 1),
            field_end.1.min(visible_size.y as isize - 1),
        );

        field_start_lut.push(field_start);
        clamped_start_lut.push(clamped_start);
        clamped_end_lut.push(clamped_end);
    }

    return (field_start_lut, clamped_start_lut, clamped_end_lut);
}

#[derive(Debug)]
pub struct Encoder {
    visible_size: CsdrSize,
    hidden_size: CsdrSize,

    area: usize,
    diameter: usize,

    field_start_lut: Vec<(isize, isize)>,
    clamped_start_lut: Vec<(isize, isize)>,
    clamped_end_lut: Vec<(isize, isize)>,

    lr: f32,
    dictionary: Vec<f32>,
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
        let range = Uniform::new(-0.1, -0.01).unwrap();
        let ratio = (
            visible_size.x as f32 / hidden_size.x as f32,
            visible_size.y as f32 / hidden_size.y as f32,
        );
        let (field_start_lut, clamped_start_lut, clamped_end_lut) =
            gen_field_bounds(radius, ratio, &hidden_size, &visible_size);

        Self {
            visible_size,
            hidden_size,

            area,
            diameter,

            field_start_lut,
            clamped_start_lut,
            clamped_end_lut,

            lr,
            dictionary: (0..visible_size.z * area * hidden_size.flat)
                .map(|_| range.sample(rng))
                .collect(),
        }
    }

    pub fn forward(&self, input: &[usize]) -> Vec<usize> {
        assert_eq!(input.len(), self.visible_size.cols);

        // NOTE: accumulator/activations columns are center-biased. Inside each, cells are normalized, meaning `column_wise_one_hot` works.
        let mut activations: Vec<f32> = vec![0.0; self.hidden_size.flat];

        for hidden_col in 0..self.hidden_size.cols {
            let field_start = self.field_start_lut[hidden_col];
            let clamped_start = self.clamped_start_lut[hidden_col];
            let clamped_end = self.clamped_end_lut[hidden_col];

            let activation_col_start = hidden_col * self.hidden_size.z;
            let activation_col =
                &mut activations[activation_col_start..(activation_col_start + self.hidden_size.z)];

            for visible_x in clamped_start.0..=clamped_end.0 {
                for visible_y in clamped_start.1..=clamped_end.1 {
                    let in_field_x = visible_x - field_start.0 as isize;
                    let in_field_y = visible_y - field_start.1 as isize;

                    let in_field_idx =
                        (in_field_x + (self.diameter as isize * in_field_y)) as usize;
                    let input_cell =
                        input[(visible_x + (self.visible_size.x as isize * visible_y)) as usize];

                    let dictionary_start = (hidden_col * self.hidden_size.z)
                        + (in_field_idx * self.visible_size.z)
                        + (input_cell * self.visible_size.z * self.area);

                    let dictionary_col =
                        &self.dictionary[dictionary_start..(dictionary_start + self.hidden_size.z)];

                    for hidden_cell in 0..self.hidden_size.z {
                        activation_col[hidden_cell] += dictionary_col[hidden_cell];
                    }
                }
            }
        }

        let hidden = column_wise_one_hot(&activations, self.hidden_size.z);
        assert_eq!(hidden.len(), self.hidden_size.cols);

        hidden
    }

    pub fn learn(&mut self, expected: &[usize], hidden: &[usize]) {
        let reconstruction = self.reconstruct(hidden);
    }

    fn reconstruct(&self, hidden: &[usize]) -> Vec<f32> {
        // NOTE: accumulator/reconstruction columns are center-biased. Inside each, cells are normalized, meaning `column_wise_one_hot` works.
        let mut reconstruction_acc: Vec<f32> = vec![0.0; self.visible_size.flat];

        for hidden_col in 0..self.hidden_size.cols {
            let field_start = self.field_start_lut[hidden_col];
            let clamped_start = self.clamped_start_lut[hidden_col];
            let clamped_end = self.clamped_end_lut[hidden_col];

            let hidden_idx = hidden[hidden_col] + (hidden_col * self.hidden_size.z);

            for visible_x in clamped_start.0..=clamped_end.0 {
                for visible_y in clamped_start.1..=clamped_end.1 {
                    let in_field_x = visible_x - field_start.0 as isize;
                    let in_field_y = visible_y - field_start.1 as isize;

                    let in_field_idx =
                        (in_field_x + (self.diameter as isize * in_field_y)) as usize;

                    let reconstruction_start = (visible_y * self.visible_size.z as isize) as usize
                        + (visible_x * self.visible_size.z as isize * self.visible_size.y as isize)
                            as usize;

                    let reconstruction_col = &mut reconstruction_acc
                        [reconstruction_start..(reconstruction_start + self.visible_size.z)];
                    let idxs: Vec<usize> = (0..self.visible_size.z)
                        .map(|z| {
                            hidden_idx
                                + (in_field_idx * self.visible_size.z)
                                + (z * self.visible_size.z * self.area)
                        })
                        .collect();

                    for visible_z in 0..self.visible_size.z {
                        reconstruction_col[visible_z] += self.dictionary[idxs[visible_z]];
                    }
                }
            }
        }

        for cell in reconstruction_acc.iter_mut() {
            *cell = exp_raw(cell.min(0.0));
        }

        return reconstruction_acc;
    }
}
