mod model;

use std::time::Instant;

use model::coder::CsdrSize;
use model::encoder::Encoder;

use crate::model::decoder::Decoder;

fn main() {
    let start = Instant::now();
    let input_size = CsdrSize::new(8, 8, 13);
    let hidden_size = CsdrSize::new(32, 32, 32);
    let mut encoder = Encoder::new(input_size, hidden_size, 1, 1, 0.01);
    let mut decoder = Decoder::new(hidden_size, input_size, 2, 1, 0.01);
    let samples = 1_000;

    let mut input_buffer = vec![0u16; input_size.cols];
    let input_start = Instant::now();
    for cell in input_buffer.iter_mut() {
        *cell = fastrand::u16(0u16..(input_size.z as u16));
    }
    let input_time = Instant::now() - input_start;
    let init_time = Instant::now() - start;

    let start = Instant::now();

    // 1. Create 3 distinct "prototype" patterns
    let mut prototype_a = vec![0u16; input_size.cols];
    let mut prototype_b = vec![0u16; input_size.cols];
    let mut prototype_c = vec![0u16; input_size.cols];

    for i in 0..input_size.cols {
        prototype_a[i] = (i % input_size.z) as u16;
        prototype_b[i] = ((i * 2) % input_size.z) as u16;
        prototype_c[i] = (input_size.z - 1 - (i % input_size.z)) as u16;
    }

    for _ in 0..samples {
        let mut current_input = match fastrand::u8(0..3) {
            0 => prototype_a.clone(),
            1 => prototype_b.clone(),
            _ => prototype_c.clone(),
        };

        for cell in current_input.iter_mut() {
            if fastrand::f32() < 0.05 {
                *cell = fastrand::u16(0u16..(input_size.z as u16));
            }
        }

        let hidden = encoder.forward(&current_input);
        let (_, learning_data) = decoder.forward(&hidden);
        encoder.learn(&current_input, &hidden);
        decoder.learn(&hidden, &hidden, &learning_data);
        std::hint::black_box(learning_data);
    }
    let elapsed = Instant::now() - start - (input_time * samples);

    println!("Initialization time: {:.2?}", init_time);
    println!("Samples: {}", samples);
    println!("Elapsed: {:.2?}", elapsed);
    println!(
        "Performance: {:.2} Hz | {:.2?} per forward",
        samples as f64 / elapsed.as_secs_f64(),
        elapsed / samples,
    );
}
