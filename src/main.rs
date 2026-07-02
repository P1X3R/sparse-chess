mod model;

use model::coder::CsdrSize;
use model::encoder;
use rand::{
    distr::{Distribution, Uniform},
    rngs::SmallRng,
};

fn main() {
    let mut rng: SmallRng = rand::make_rng();
    let input_size = CsdrSize::new(8, 8, 12);
    let range = Uniform::new(0, input_size.z).unwrap();
    let mut e = encoder::Encoder::new(input_size, CsdrSize::new(32, 32, 32), 1, 0.01, &mut rng);
    let samples = 10_000;

    let mut input_buffer = vec![0; input_size.cols];

    for _ in 0..samples {
        for cell in input_buffer.iter_mut() {
            *cell = range.sample(&mut rng);
        }

        e.forward(&input_buffer);
    }

    println!("Samples: {}", samples);
}
