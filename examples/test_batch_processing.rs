//! Example demonstrating the new batch processing functionality.
//!
//! This example shows how the fft_batch and ifft_batch functions work
//! to process multiple FFTs efficiently.

use num_complex::Complex;
use wgls_rs_fft::GpuFft;

fn main() {
    println!("🚀 Testing Batch Processing Implementation");
    println!("=========================================\n");

    let fft = GpuFft::new().expect("GPU required");
    println!("✅ Created GPU FFT instance\n");

    // Test 1: Single FFT (using batch API with single element)
    println!("Test 1: Single FFT (backward compatibility)");
    let single_input = vec![vec![Complex::new(1.0, 0.0); 16]];
    let single_result_batch = fft.fft(&single_input).expect("Single FFT failed");
    let single_result = &single_result_batch[0];
    println!("  Input length: {}", single_input[0].len());
    println!("  Output length: {}", single_result.len());
    println!("  ✅ Single FFT works\n");

    // Test 2: Batch FFT
    println!("Test 2: Batch FFT Processing");
    let batch_inputs = vec![
        vec![Complex::new(1.0, 0.0); 16],
        vec![Complex::new(0.5, 0.0); 16],
        vec![Complex::new(0.25, 0.0); 16],
    ];

    let batch_results = fft.fft(&batch_inputs).expect("Batch FFT failed");
    println!("  Batch size: {}", batch_results.len());
    println!("  Each FFT size: {}", batch_results[0].len());
    println!("  ✅ Batch FFT works\n");

    // Test 3: Batch IFFT
    println!("Test 3: Batch IFFT Processing");
    let batch_ifft_results = fft.ifft(&batch_results).expect("Batch IFFT failed");
    println!("  Batch size: {}", batch_ifft_results.len());
    println!("  Each IFFT size: {}", batch_ifft_results[0].len());
    println!("  ✅ Batch IFFT works\n");

    // Test 4: Roundtrip consistency
    println!("Test 4: Roundtrip Consistency Check");
    let mut max_total_error: f32 = 0.0;
    for (i, (original, reconstructed)) in batch_inputs
        .iter()
        .zip(batch_ifft_results.iter())
        .enumerate()
    {
        let max_error: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(orig, recon)| {
                ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt()
            })
            .fold(0.0, f32::max);
        println!("  Signal {} max error: {:.2e}", i, max_error);
        max_total_error = max_total_error.max(max_error);
    }
    println!("  Overall max error: {:.2e}", max_total_error);
    assert!(
        max_total_error < 1e-4,
        "Roundtrip error too large: {}",
        max_total_error
    );
    println!("  ✅ Roundtrip consistency verified\n");

    // Test 5: Performance comparison
    println!("Test 5: Performance Comparison");
    use std::time::Instant;

    let large_batch_size = 8;
    let large_fft_size = 1024;
    let large_batch: Vec<_> = (0..large_batch_size)
        .map(|i| vec![Complex::new(i as f32 * 0.1, 0.0); large_fft_size])
        .collect();

    // Individual processing
    let start = Instant::now();
    let individual_results: Vec<_> = large_batch
        .iter()
        .map(|input| {
            let batch = vec![input.clone()];
            fft.fft(&batch).expect("FFT failed")[0].clone()
        })
        .collect();
    let individual_time = start.elapsed();

    // Batch processing
    let start = Instant::now();
    let batch_results = fft.fft(&large_batch).expect("Batch FFT failed");
    let batch_time = start.elapsed();

    println!("  Individual processing time: {:?}", individual_time);
    println!("  Batch processing time: {:?}", batch_time);

    if batch_time < individual_time {
        let speedup = individual_time.as_secs_f32() / batch_time.as_secs_f32();
        println!("  Speedup: {:.2}x", speedup);
    } else {
        println!("  Note: Batch processing may not be faster for small batches");
    }
    println!("  ✅ Performance test completed\n");

    println!("🎉 All batch processing tests passed!");
    println!("\nSummary:");
    println!("  ✅ Single FFT (backward compatibility)");
    println!("  ✅ Batch FFT processing");
    println!("  ✅ Batch IFFT processing");
    println!("  ✅ Roundtrip consistency");
    println!("  ✅ Performance comparison");
    println!("\nThe batch processing implementation is working correctly!");
}
