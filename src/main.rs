mod model;

use std::time::Instant;

use model::coder::CsdrSize;
use model::encoder::Encoder;
use rand::{
    distr::{Distribution, Uniform},
    rngs::SmallRng,
};

fn main() {
    let start = Instant::now();
    let mut rng: SmallRng = rand::make_rng();
    let input_size = CsdrSize::new(8, 8, 13);
    let range = Uniform::new(0, input_size.z as u16).unwrap();
    let mut encoder = Encoder::new(input_size, CsdrSize::new(32, 32, 32), 1, 1, 0.01, &mut rng);
    let samples = 1_000;

    let mut input_buffer = vec![0u16; input_size.cols];
    for cell in input_buffer.iter_mut() {
        *cell = range.sample(&mut rng);
    }
    let init_time = Instant::now() - start;

    let start = Instant::now();
    for _ in 0..samples {
        encoder.forward(&input_buffer);
        encoder.learn(&input_buffer);
    }
    let elapsed = Instant::now() - start;

    println!("Initialization time: {:.2?}", init_time);
    println!("Samples: {}", samples);
    println!("Elapsed: {:.2?}", elapsed);
    println!(
        "Performance: {:.2} Hz | {:.2?} per forward",
        samples as f64 / elapsed.as_secs_f64(),
        elapsed / samples,
    );
}
