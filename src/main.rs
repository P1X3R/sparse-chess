mod model;

use model::coder::CsdrSize;
use model::encoder;
use rand::{RngExt, rngs::SmallRng};

fn main() {
    let mut rng: SmallRng = rand::make_rng();
    let input_size = CsdrSize::new(4, 4, 4);
    let mut e = encoder::Encoder::new(input_size, CsdrSize::new(8, 8, 8), 1, 0.01, &mut rng);
    let input: Vec<usize> = (0..input_size.cols)
        .map(|_| rng.random_range(0..input_size.z))
        .collect();
    let hidden = e.forward(&input);
    println!("Input:          {:?}", input);
    println!("Hidden:         {:?}", hidden);
    e.learn(&input, &hidden);
}
