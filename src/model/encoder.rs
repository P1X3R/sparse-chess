use rand::{RngExt, rngs::SmallRng};

#[derive(Debug)]
pub struct Encoder {
    input_cols: usize,
    input_cells: usize,
    input_size: usize,

    hidden_cols: usize,
    hidden_cells: usize,
    hidden_size: usize,

    lr: u8,

    dictionary: Vec<i8>,
}

impl Encoder {
    pub fn new(
        input_cols: usize,
        input_cells: usize,
        hidden_cols: usize,
        hidden_cells: usize,
        lr: u8,
        rng: &mut SmallRng,
    ) -> Self {
        let hidden_size = input_cols * input_cells;
        let input_size = input_cols * input_cells;
        return Self {
            input_cols,
            input_cells,
            input_size,
            hidden_cols,
            hidden_cells,
            hidden_size,
            lr,
            dictionary: (0..input_size * hidden_size)
                .map(|_| rng.random_range(-2..=-1))
                .collect(),
        };
    }

    pub fn forward(&mut self, input: Vec<i8>) -> Vec<i8> {
        return vec![];
    }
}
