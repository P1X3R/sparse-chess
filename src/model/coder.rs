pub trait Coder {
    fn forward(&self, input: Vec<usize>) -> Vec<usize>;
    fn learn(&mut self);
}

#[derive(Debug, Copy, Clone)]
pub struct CsdrSize {
    pub x: usize,
    pub y: usize,
    pub z: usize,
    pub cols: usize,
    pub flat: usize,
}

impl CsdrSize {
    pub fn new(x: usize, y: usize, z: usize) -> Self {
        return Self {
            x,
            y,
            z,
            cols: x * y,
            flat: x * y * z,
        };
    }
}

pub fn column_wise_one_hot(csdr: Vec<i8>, z: usize) -> Vec<usize> {
    return csdr
        .chunks_exact(z)
        .map(|col| {
            col.iter()
                .enumerate()
                .max_by_key(|&(_, cell)| cell)
                .map(|(idx, _)| idx)
                .unwrap()
        })
        .collect();
}
