//! Example demonstrating FFT with window functions and IFFT.
//!
//! This example shows how to:
//! 1. Apply a window function to time-domain data
//! 2. Compute FFT to get frequency spectrum
//! 3. Compute IFFT to reconstruct the windowed signal
//! 4. Measure performance of the operations

use num_complex::Complex;
use std::time::Instant;
use wgls_rs_fft::GpuFft;

/// Apply a Hann window function to the input signal
fn apply_hann_window(signal: &mut [Complex<f32>]) {
    let n = signal.len();
    for (i, sample) in signal.iter_mut().enumerate() {
        let window = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
        *sample = Complex {
            re: sample.re * window,
            im: sample.im * window,
        };
    }
}

/// Generate a test signal with multiple frequencies
fn generate_test_signal(n: usize) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            // Mix of sine waves: 5Hz, 20Hz, and 50Hz components
            let signal = 0.5 * (2.0 * std::f32::consts::PI * 5.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 20.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 50.0 * t).sin();
            Complex {
                re: signal,
                im: 0.0,
            }
        })
        .collect()
}

/// Compute power spectrum from FFT result
fn compute_power_spectrum(fft_result: &[Complex<f32>]) -> Vec<f32> {
    fft_result
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .collect()
}

fn main() {
    println!("Windowed FFT Example");
    println!("====================\n");

    // Configuration
    let n = 4096;
    println!("Signal length: {} samples", n);

    // Generate test signal
    let mut signal = generate_test_signal(n);

    // Apply window function
    apply_hann_window(&mut signal);
    println!("Applied Hann window to signal");

    // Create FFT instance
    let fft = GpuFft::new().expect("GPU required");
    println!("Using GPU backend");

    // Performance measurement
    let start = Instant::now();

    // Compute FFT
    let spectrum = fft.fft(&signal).expect("FFT failed");
    let fft_time = start.elapsed();

    // Compute power spectrum
    let power_spectrum = compute_power_spectrum(&spectrum);
    let power_time = Instant::now() - (start + fft_time);

    // Compute IFFT to reconstruct windowed signal
    let start_ifft = Instant::now();
    let reconstructed = fft.ifft(&spectrum).expect("IFFT failed");
    let ifft_time = start_ifft.elapsed();

    // Performance results
    println!("\nPerformance Results:");
    println!("  FFT time:    {:?}", fft_time);
    println!("  Power time:  {:?}", power_time);
    println!("  IFFT time:   {:?}", ifft_time);
    println!("  Total time:  {:?}", fft_time + power_time + ifft_time);

    // Find peak frequencies
    let mut peak_frequencies = vec![];
    let sample_rate = n as f32; // Assume sample rate equals signal length for demo

    // Look for peaks in the first half of the spectrum (avoid mirroring)
    for i in 1..n / 2 {
        if power_spectrum[i] > 100.0
            && power_spectrum[i] > power_spectrum[i - 1]
            && power_spectrum[i] > power_spectrum[i + 1]
        {
            let freq = i as f32 * sample_rate / n as f32;
            let power = power_spectrum[i];
            peak_frequencies.push((freq, power));
        }
    }

    println!("\nDetected Peak Frequencies:");
    for (freq, power) in peak_frequencies {
        println!("  {:.1} Hz (power: {:.0})", freq, power);
    }

    // Verify roundtrip accuracy
    let mut max_error: f32 = 0.0;
    let mut total_error: f32 = 0.0;
    for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
        let error: f32 = ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
        max_error = max_error.max(error);
        total_error += error;
    }
    let avg_error = total_error / n as f32;

    println!("\nRoundtrip Accuracy:");
    println!("  Max error:  {:.2e}", max_error);
    println!("  Avg error:  {:.2e}", avg_error);
    println!(
        "  SNR:       {:.1} dB",
        20.0 * (max_error + 1e-10).log10().abs()
    );

    println!("\nExample completed successfully!");
}
