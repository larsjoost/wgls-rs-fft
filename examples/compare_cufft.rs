use num_complex::Complex;
use std::time::Instant;
use wgls_rs_fft::GpuFft;

#[cfg(feature = "cuda")]
use wgls_rs_fft::CuFft;
#[cfg(feature = "rocm")]
use wgls_rs_fft::RocFft;

fn main() {
    println!("FFT Implementation Comparison");
    println!("==============================");

    // Test parameters
    let fft_size = 1024;
    let batch_sizes = vec![1, 4, 16, 64, 256];

    // Generate test data
    let inputs: Vec<Vec<Complex<f32>>> = (0..256)
        .map(|i| {
            (0..fft_size)
                .map(|j| Complex::new((i * fft_size + j) as f32 * 0.1, 0.0))
                .collect()
        })
        .collect();

    // Test WebGPU implementation
    println!("\nTesting WebGPU Implementation:");
    println!("------------------------------");

    let wgpu_fft = GpuFft::new().expect("Failed to create WebGPU FFT instance");

    for &batch_size in &batch_sizes {
        let batch_inputs: Vec<Vec<Complex<f32>>> = inputs[..batch_size].to_vec();

        // Warm-up
        let _ = wgpu_fft.fft(&batch_inputs).expect("FFT failed");

        // Timed run
        let start = Instant::now();
        let _results = wgpu_fft.fft(&batch_inputs).expect("FFT failed");
        let duration = start.elapsed();

        let total_samples = batch_size * fft_size;
        let samples_per_second = total_samples as f64 / duration.as_secs_f64();
        let mega_samples_per_second = samples_per_second / 1_000_000.0;

        println!(
            "Batch Size {}: {:.2} MSa/s",
            batch_size, mega_samples_per_second
        );
    }

    // Test cuFFT implementation (if available)
    #[cfg(feature = "cuda")]
    {
        match CuFft::new(fft_size) {
            Ok(cufft) => {
                println!("\nTesting cuFFT Implementation:");
                println!("------------------------------");

                for &batch_size in &batch_sizes {
                    let batch_inputs: Vec<Vec<Complex<f32>>> = inputs[..batch_size].to_vec();

                    // Warm-up
                    let _ = cufft.batch_fft(&batch_inputs).expect("FFT failed");

                    // Timed run
                    let start = Instant::now();
                    let _results = cufft.batch_fft(&batch_inputs).expect("FFT failed");
                    let duration = start.elapsed();

                    let total_samples = batch_size * fft_size;
                    let samples_per_second = total_samples as f64 / duration.as_secs_f64();
                    let mega_samples_per_second = samples_per_second / 1_000_000.0;

                    println!(
                        "Batch Size {}: {:.2} MSa/s",
                        batch_size, mega_samples_per_second
                    );
                }
            }
            Err(e) => {
                println!("\ncuFFT not available ({e})");
                if e.to_string().contains("CUFFT_NOT_SUPPORTED") {
                    println!(
                        "Hint: try launching with LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
                    );
                }
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        println!("\ncuFFT support not compiled (add --features cuda to enable)");
    }

    // Test ROCm implementation (AMD GPUs only)
    #[cfg(feature = "rocm")]
    {
        println!("\nTesting ROCm FFT Implementation:");
        println!("----------------------------------");

        match RocFft::new() {
            Ok(_) => {
                println!("ROCm FFT initialized successfully");
                // ROCm FFT implementation would go here
                // Note: This won't work on NVIDIA GTX 1070
            }
            Err(e) => {
                println!("ROCm FFT error: {}", e);
            }
        }
    }

    #[cfg(not(feature = "rocm"))]
    {
        println!("\nROCm support not compiled (add --features rocm to enable, AMD GPUs only)");
    }
}
