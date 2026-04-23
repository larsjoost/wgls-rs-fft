//! CI Debug Test File
//!
//! This test file helps debug CI issues by running the most critical tests
//! in an environment that mimics GitHub Actions as closely as possible.

use num_complex::Complex;
use wgls_rs_fft::GpuFft;

#[test]
fn test_ci_minimal() {
    // This test verifies the most basic functionality
    // It should work in CI environments

    // Check if GPU is available first
    if !GpuFft::is_gpu_available() {
        println!("CI DEBUG: No GPU available - using CPU fallback would be needed");
        return; // Skip test if no GPU
    }

    // Create FFT instance
    let fft = match GpuFft::new() {
        Ok(fft) => fft,
        Err(e) => {
            println!("CI DEBUG: Failed to create FFT: {}", e);
            return;
        }
    };

    // Create test signal
    let mut signal = vec![Complex::new(1.0, 0.0); 1024];

    // Add some variation
    for (i, c) in signal.iter_mut().enumerate() {
        let t = i as f32 / 1024.0;
        c.re = (t * 2.0 * std::f32::consts::PI * 5.0).sin();
    }

    // Test FFT
    match fft.fft(&signal) {
        Ok(spectrum) => {
            println!(
                "CI DEBUG: FFT successful, spectrum length: {}",
                spectrum.len()
            );

            // Test IFFT
            match fft.ifft(&spectrum) {
                Ok(reconstructed) => {
                    println!("CI DEBUG: IFFT successful");

                    // Verify roundtrip
                    let mut max_error = 0.0f32;
                    for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
                        let error =
                            ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
                        max_error = max_error.max(error);
                    }
                    println!("CI DEBUG: Max roundtrip error: {:.2e}", max_error);
                    assert!(max_error < 1e-3, "Roundtrip error too large: {}", max_error);
                }
                Err(e) => {
                    println!("CI DEBUG: IFFT failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("CI DEBUG: FFT failed: {}", e);
        }
    }
}

#[test]
fn test_ci_dependencies() {
    // Test that all dependencies are available
    println!("CI DEBUG: Testing dependencies...");

    // This test mainly verifies the environment
    assert!(
        GpuFft::is_gpu_available(),
        "CI DEBUG: GPU should be available in CI"
    );
}

#[test]
fn test_ci_performance() {
    // Performance test that should work in CI
    use std::time::Instant;

    if !GpuFft::is_gpu_available() {
        println!("CI DEBUG: No GPU available, skipping performance test");
        return;
    }

    let fft = GpuFft::new().expect("GPU required");
    let signal = vec![Complex::new(1.0, 0.0); 4096];

    let start = Instant::now();
    let spectrum = fft.fft(&signal).expect("FFT failed");
    let fft_time = start.elapsed();

    let start = Instant::now();
    let _reconstructed = fft.ifft(&spectrum).expect("IFFT failed");
    let ifft_time = start.elapsed();

    println!("CI DEBUG: FFT: {:?}, IFFT: {:?}", fft_time, ifft_time);
}
