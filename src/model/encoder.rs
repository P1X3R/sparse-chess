use crate::model::coder::{CsdrSize, column_wise_one_hot};
use rand::{
    distr::{Distribution, Uniform},
    rngs::SmallRng,
};

#[derive(Debug)]
struct FieldBounds {
    field_start: (i16, i16),
    clamped_start: (i16, i16),
    clamped_end: (i16, i16),
}

impl FieldBounds {
    fn new(
        hidden_x: usize,
        hidden_y: usize,
        radius: usize,
        ratio: (f32, f32),
        visible_size: &CsdrSize,
    ) -> Self {
        let visible_center = (
            ((hidden_x as f32 + 0.5f32) * ratio.0) as i16,
            ((hidden_y as f32 + 0.5f32) * ratio.1) as i16,
        );

        let field_start = (
            visible_center.0 - radius as i16,
            visible_center.1 - radius as i16,
        );

        let field_end = (
            visible_center.0 as i16 + radius as i16,
            visible_center.1 as i16 + radius as i16,
        );

        let clamped_start = (field_start.0.max(0), field_start.1.max(0));
        let clamped_end = (
            field_end.0.min(visible_size.x as i16 - 1),
            field_end.1.min(visible_size.y as i16 - 1),
        );

        Self {
            field_start,
            clamped_start,
            clamped_end,
        }
    }
}

#[derive(Debug)]
pub struct Encoder {
    visible_size: CsdrSize,
    hidden_size: CsdrSize,

    area: usize,
    diameter: usize,
    field_bounds: Vec<FieldBounds>,

    activations: Vec<f32>,

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

        Self {
            visible_size,
            hidden_size,

            area,
            diameter,
            field_bounds: (0..hidden_size.cols)
                .map(|col| {
                    FieldBounds::new(
                        col % hidden_size.x,
                        col / hidden_size.y,
                        radius,
                        ratio,
                        &visible_size,
                    )
                })
                .collect(),

            activations: vec![0.0; hidden_size.flat],

            lr,
            dictionary: (0..area * visible_size.z * hidden_size.z)
                .map(|_| range.sample(rng))
                .collect(),
        }
    }

    pub fn forward(&mut self, input: &[usize]) -> Vec<usize> {
        assert_eq!(input.len(), self.visible_size.cols);

        self.activations.fill(0.0);

        // NOTE: accumulator/activations columns are center-biased. Inside each, cells are normalized, meaning `column_wise_one_hot` works.
        for hidden_col in 0..self.hidden_size.cols {
            let FieldBounds {
                field_start: (field_start_x, field_start_y),
                clamped_start: (clamped_start_x, clamped_start_y),
                clamped_end: (clamped_end_x, clamped_end_y),
            } = self.field_bounds[hidden_col];

            let (field_start_x, field_start_y) = (field_start_x as isize, field_start_y as isize);
            let (clamped_start_x, clamped_start_y) =
                (clamped_start_x as isize, clamped_start_y as isize);
            let (clamped_end_x, clamped_end_y) = (clamped_end_x as isize, clamped_end_y as isize);

            let activation_col_start = hidden_col * self.hidden_size.z;
            let activation_col = &mut self.activations
                [activation_col_start..(activation_col_start + self.hidden_size.z)];

            for visible_y in clamped_start_y..=clamped_end_y {
                let visible_y_offset = (self.visible_size.x as isize * visible_y) as usize;
                let in_field_y = visible_y - field_start_y;
                let in_field_y_offset = (self.diameter as isize * in_field_y) as usize;

                for visible_x in clamped_start_x..=clamped_end_x {
                    let in_field_x = visible_x - field_start_x;
                    let in_field_idx = (in_field_x as usize) + in_field_y_offset;

                    let input_cell = input[(visible_x as usize) + visible_y_offset];

                    let dictionary_start =
                        self.hidden_size.z * (input_cell + (in_field_idx * self.visible_size.z));

                    let dictionary_col =
                        &self.dictionary[dictionary_start..(dictionary_start + self.hidden_size.z)];

                    for (activation, weight) in activation_col.iter_mut().zip(dictionary_col) {
                        *activation += weight;
                    }
                }
            }
        }

        let hidden = column_wise_one_hot(&self.activations, self.hidden_size.z);
        assert_eq!(hidden.len(), self.hidden_size.cols);

        hidden
    }

    pub fn learn(&mut self, expected: &[usize], hidden: &[usize]) {
        todo!()
    }
}
