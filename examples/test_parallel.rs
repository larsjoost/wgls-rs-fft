use num_complex::Complex;
use std::time::Instant;
use wgls_rs_fft::GpuFft;

fn main() {
    let fft = GpuFft::new().expect("Failed to create FFT instance");

    println!("Testing Parallel Batch Processing");
    println!("==================================");

    // Test different batch sizes
    let fft_size = 1024;
    let batch_sizes = vec![1, 4, 16, 64, 256];

    for &batch_size in &batch_sizes {
        // Generate test data
        let inputs: Vec<Vec<Complex<f32>>> = (0..batch_size)
            .map(|i| {
                (0..fft_size)
                    .map(|j| Complex::new((i * fft_size + j) as f32 * 0.1, 0.0))
                    .collect()
            })
            .collect();

        // Warm-up
        let _ = fft.fft(&inputs).expect("FFT failed");

        // Timed run
        let start = Instant::now();
        let results = fft.fft(&inputs).expect("FFT failed");
        let duration = start.elapsed();

        let total_samples = batch_size * fft_size;
        let samples_per_second = total_samples as f64 / duration.as_secs_f64();
        let mega_samples_per_second = samples_per_second / 1_000_000.0;

        println!(
            "Batch Size {}: {:.2} MSa/s ({} FFTs, {} samples)",
            batch_size, mega_samples_per_second, batch_size, total_samples
        );

        // Verify results are correct (basic sanity check)
        for result in &results {
            assert_eq!(result.len(), fft_size);
        }
    }

    println!("\nParallel processing test completed successfully!");
}
