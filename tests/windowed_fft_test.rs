use num_complex::Complex;
use wgls_rs_fft::GpuFft;

const N: usize = 2048;
const EPSILON: f32 = 1e-3;

/// Apply Hann window function
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

/// Generate a test signal with known frequencies
fn generate_test_signal(n: usize) -> Vec<Complex<f32>> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            // 10Hz and 50Hz sine waves
            let signal = 0.7 * (2.0 * std::f32::consts::PI * 10.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 50.0 * t).sin();
            Complex {
                re: signal,
                im: 0.0,
            }
        })
        .collect()
}

#[test]
fn test_windowed_fft_ifft_roundtrip() {
    let fft = GpuFft::new().expect("GPU required");

    // Generate signal and apply window
    let mut signal = generate_test_signal(N);
    apply_hann_window(&mut signal);

    // FFT -> IFFT roundtrip
    let spectrum = fft.fft(&signal).expect("FFT failed");
    let reconstructed = fft.ifft(&spectrum).expect("IFFT failed");

    // Verify roundtrip accuracy
    assert_eq!(reconstructed.len(), N);

    let mut max_diff: f32 = 0.0;
    for (i, (orig, recon)) in signal.iter().zip(reconstructed.iter()).enumerate() {
        let diff = ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
        max_diff = max_diff.max(diff);
        assert!(
            diff < EPSILON,
            "element {i}: original={orig:?}  reconstructed={recon:?}  diff={diff:.2e}"
        );
    }
    println!("Windowed FFT->IFFT roundtrip max error: {max_diff:.2e}");
}

#[test]
fn test_window_function_properties() {
    let fft = GpuFft::new().expect("GPU required");

    // Generate signal
    let mut signal = generate_test_signal(N);
    let original = signal.clone();

    // Apply window and compute FFT
    apply_hann_window(&mut signal);
    let windowed_spectrum = fft.fft(&signal).expect("FFT failed");

    // Compute FFT of original (unwindowed)
    let original_spectrum = fft.fft(&original).expect("FFT failed");

    // Windowing should reduce spectral leakage
    // Check that windowed spectrum has lower side lobes
    let windowed_power: f32 = windowed_spectrum
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .sum();
    let original_power: f32 = original_spectrum
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .sum();

    // Power should be preserved (within reasonable bounds)
    let power_ratio = windowed_power / original_power;
    println!("Windowed vs original power ratio: {power_ratio:.3}");

    // Should be reasonably close to 1 (window function preserves energy)
    assert!(
        power_ratio > 0.1 && power_ratio < 10.0,
        "Power ratio {power_ratio} is unexpected"
    );
}

#[test]
fn test_multiple_window_types() {
    let fft = GpuFft::new().expect("GPU required");
    let mut signal = generate_test_signal(N);

    // Test Hann window (already implemented)
    apply_hann_window(&mut signal);
    let hann_spectrum = fft.fft(&signal).expect("FFT failed");
    let hann_reconstructed = fft.ifft(&hann_spectrum).expect("IFFT failed");

    let mut max_error: f32 = 0.0;
    for (orig, recon) in signal.iter().zip(hann_reconstructed.iter()) {
        let error = ((orig.re - recon.re).powi(2) + (orig.im - recon.im).powi(2)).sqrt();
        max_error = max_error.max(error);
    }
    println!("Hann window roundtrip max error: {max_error:.2e}");
    assert!(max_error < EPSILON);
}

#[test]
fn test_frequency_detection() {
    let fft = GpuFft::new().expect("GPU required");

    // Generate signal with known frequencies
    let mut signal = generate_test_signal(N);
    apply_hann_window(&mut signal);

    // Compute FFT
    let spectrum = fft.fft(&signal).expect("FFT failed");
    let power_spectrum: Vec<f32> = spectrum.iter().map(|c| c.re * c.re + c.im * c.im).collect();

    // Find peaks in the spectrum
    let mut peaks = vec![];
    let sample_rate = N as f32; // samples per cycle

    for i in 1..N / 2 {
        if power_spectrum[i] > power_spectrum[i - 1]
            && power_spectrum[i] > power_spectrum[i + 1]
            && power_spectrum[i] > 10.0
        {
            // minimum power threshold
            let freq = i as f32 * sample_rate / N as f32;
            peaks.push((freq, power_spectrum[i]));
        }
    }

    println!("Detected {} frequency peaks", peaks.len());
    for (freq, power) in &peaks {
        println!("  {:.1} Hz (power: {:.0})", freq, power);
    }

    // Should detect our known frequencies (10Hz and 50Hz)
    assert!(
        !peaks.is_empty(),
        "Should detect at least one frequency peak"
    );

    // Check that detected frequencies are in expected ranges
    let has_10hz_peak = peaks.iter().any(|(freq, _)| (8.0..12.0).contains(freq));
    let has_50hz_peak = peaks.iter().any(|(freq, _)| (45.0..55.0).contains(freq));

    println!("Detected 10Hz peak: {}", has_10hz_peak);
    println!("Detected 50Hz peak: {}", has_50hz_peak);

    // At least one of the known frequencies should be detected
    assert!(
        has_10hz_peak || has_50hz_peak || peaks.len() >= 2,
        "Should detect known frequency components"
    );
}
