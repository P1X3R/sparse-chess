mod model;

use model::{
    coder::CsdrSize,
    sph::{LayerParams, Sph},
};
use std::time::Instant;

const LOG_ERR: bool = true;

fn main() {
    let start = Instant::now();

    // 1. Setup Layer Pipeline Geometry
    let pipeline_sizes = [
        (8, 8, 13), // Input:  8x8 board, 12 feature planes (e.g., piece types)
        (8, 8, 8),  // Layer 1: 8x8 cols, 8 cells/col (reduced from 16)
        (4, 4, 12), // Layer 2: 4x4 cols, 12 cells/col (reduced from 32)
    ];

    // 2. Setup Layer Hyperparameters
    let params = vec![
        LayerParams {
            decoder_lr: 0.08, // Slightly faster learning rate
            encoder_lr: 0.05,
            radius: 1,
            learning_radius: 1,
            choice: 0.1,
            vigilance: 0.85, // Higher vigilance forces more distinct SDR representations
            active_ratio: 0.1,
            half_dendrites: 4, // INCREASED: 4 -> 8 total dendrites per cell
        },
        LayerParams {
            decoder_lr: 0.08,
            encoder_lr: 0.05,
            radius: 1,
            learning_radius: 1,
            choice: 0.1,
            vigilance: 0.85,
            active_ratio: 0.1,
            half_dendrites: 4, // INCREASED: 4 -> 8 total dendrites per cell
        },
    ];

    // Initialize the hierarchy
    let mut sph = Sph::new(&pipeline_sizes, &params);

    let input_size = CsdrSize::new(8, 8, 13);
    let samples = 1_000;

    let mut input_buffer = vec![0u16; input_size.cols];
    for cell in input_buffer.iter_mut() {
        *cell = fastrand::u16(0u16..(input_size.z as u16));
    }
    let init_time = Instant::now() - start;

    let mut prototype_a = vec![0u16; input_size.cols].into_boxed_slice();
    let mut prototype_b = vec![0u16; input_size.cols].into_boxed_slice();
    let mut prototype_c = vec![0u16; input_size.cols].into_boxed_slice();

    for i in 0..input_size.cols {
        prototype_a[i] = (i % input_size.z) as u16;
        prototype_b[i] = ((i * 2) % input_size.z) as u16;
        prototype_c[i] = (input_size.z - 1 - (i % input_size.z)) as u16;
    }

    // Measure baseline input noise generation time per iteration
    let noise_start = Instant::now();
    for i in 0..samples {
        let prototype_index = i % 3;
        let mut current_input = match prototype_index {
            0 => prototype_a.clone(),
            1 => prototype_b.clone(),
            _ => prototype_c.clone(),
        };

        for cell in current_input.iter_mut() {
            if fastrand::f32() < 0.05 {
                *cell = fastrand::u16(0u16..(input_size.z as u16));
            }
        }
    }
    let noise_overhead = Instant::now() - noise_start;

    // 3. Execution Loop with MSE Error Logging
    let sph_start = Instant::now();
    let mut previous_prediction: Option<Box<[u16]>> = None;

    let mut running_accuracy = 0.0;
    let alpha = 0.05;

    for i in 0..samples {
        let prototype_index = i % 3;
        let mut current_input = match prototype_index {
            0 => prototype_a.clone(),
            1 => prototype_b.clone(),
            _ => prototype_c.clone(),
        };

        // Add 5% noise to simulate probabilistic input transitions
        for cell in current_input.iter_mut() {
            if fastrand::f32() < 0.05 {
                *cell = fastrand::u16(0u16..(input_size.z as u16));
            }
        }

        // --- Measure Prediction Accuracy & Categorical Error ---
        if LOG_ERR
            && i > 30
            && let Some(pred) = &previous_prediction
        {
            let mut incorrect_cells = 0;

            for (&actual, &predicted) in current_input.iter().zip(pred.iter()) {
                if actual != predicted {
                    incorrect_cells += 1;
                }
            }

            let accuracy = 100.0 * (1.0 - (incorrect_cells as f32 / current_input.len() as f32));
            let current_acc = 100.0 * (1.0 - (incorrect_cells as f32 / current_input.len() as f32));
            running_accuracy = if running_accuracy == 0.0 {
                current_acc
            } else {
                (1.0 - alpha) * running_accuracy + alpha * current_acc
            };

            if i % 50 == 0 {
                println!(
                    "Step {:4}: Instant Acc = {:5.1}% | Smoothed Acc = {:5.1}%",
                    i, accuracy, running_accuracy
                );
            }
        }

        // Run step with learning enabled
        let output = sph.step(&current_input, true);

        // Save prediction for evaluation in step i + 1
        previous_prediction = Some(output.into_boxed_slice());
    }

    let raw_elapsed = Instant::now() - sph_start;
    let elapsed = raw_elapsed.saturating_sub(noise_overhead);

    println!();
    println!("Initialization time: {:.2?}", init_time);
    println!("Samples: {}", samples);
    println!("Elapsed: {:.2?}", elapsed);
    println!(
        "Performance: {:.2} Hz | {:.2?} per step",
        samples as f64 / elapsed.as_secs_f64(),
        elapsed / samples as u32,
    );
}
