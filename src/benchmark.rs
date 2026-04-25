use std::time::Instant;

use num_complex::Complex;

use crate::FftExecutor;

/// Outcome of validating a rival's output against the reference implementation.
pub enum ValidationOutcome {
    Pass,
    /// Max absolute complex-norm error across all outputs.
    Fail {
        max_error: f32,
    },
}

/// Performance and correctness result for one (rival, N, batch_size) triple.
pub struct BenchmarkResult {
    pub rival_name: String,
    pub n: usize,
    pub batch_size: usize,
    pub msamples_per_sec: f64,
    pub gflops: f64,
    pub validation: ValidationOutcome,
}

/// Number of warm-up passes before timing.
pub const WARMUP_ITERS: usize = 1;
/// Number of timed iterations used to compute the average.
pub const BENCH_ITERS: usize = 10;
/// Absolute complex-norm tolerance for correctness.
pub const VALIDATION_TOLERANCE: f32 = 1e-3;
/// Cap on total samples (N × batch) to avoid OOM on very large FFTs.
pub const MAX_TOTAL_SAMPLES: usize = 16 * 1024 * 1024;

/// Benchmark `rival` at a fixed `n` and `batch_size`, validating against `reference`.
///
/// Uses a deterministic input so every rival sees the same signal.
/// Runs [`WARMUP_ITERS`] warm-up passes, then [`BENCH_ITERS`] timed passes.
/// Validation compares one additional forward FFT against `reference` output,
/// requiring every sample to be within [`VALIDATION_TOLERANCE`].
pub fn benchmark_rival(
    rival: &dyn FftExecutor,
    reference: &dyn FftExecutor,
    n: usize,
    batch_size: usize,
) -> BenchmarkResult {
    // Amplitude is normalised by N so that FFT-bin magnitudes stay O(1)
    // regardless of transform size. This keeps float32 precision well within
    // VALIDATION_TOLERANCE for all rivals, regardless of their radix choice.
    let inputs: Vec<Vec<Complex<f32>>> = (0..batch_size)
        .map(|_| {
            let n_f = n as f32;
            (0..n)
                .map(|i| {
                    let t = i as f32 / n_f;
                    Complex::new(t * 0.001, (t * std::f32::consts::TAU).sin() * 0.001)
                })
                .collect()
        })
        .collect();

    for _ in 0..WARMUP_ITERS {
        let _ = rival.fft(&inputs).unwrap();
    }

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        let _ = rival.fft(&inputs).unwrap();
    }
    let duration = start.elapsed() / BENCH_ITERS as u32;

    let total_samples = (n * batch_size) as f64;
    let msamples_per_sec = total_samples / duration.as_secs_f64() / 1_000_000.0;
    let gflops = 5.0 * total_samples * (n as f64).log2() / duration.as_secs_f64() / 1_000_000_000.0;

    let rival_out = rival.fft(&inputs).unwrap();
    let ref_out = reference.fft(&inputs).unwrap();
    let validation = validate(&rival_out, &ref_out);

    BenchmarkResult {
        rival_name: rival.name().to_string(),
        n,
        batch_size,
        msamples_per_sec,
        gflops,
        validation,
    }
}

/// Sweep `batch_sizes`, skip combinations where `n × batch > MAX_TOTAL_SAMPLES`,
/// and return the result with the highest `msamples_per_sec`.
pub fn sweep_rival(
    rival: &dyn FftExecutor,
    reference: &dyn FftExecutor,
    n: usize,
    batch_sizes: &[usize],
) -> BenchmarkResult {
    let mut best: Option<BenchmarkResult> = None;
    for &batch_size in batch_sizes {
        if n * batch_size > MAX_TOTAL_SAMPLES {
            continue;
        }
        let result = benchmark_rival(rival, reference, n, batch_size);
        if best
            .as_ref()
            .map_or(true, |b| result.msamples_per_sec > b.msamples_per_sec)
        {
            best = Some(result);
        }
    }
    // If all batch sizes were skipped (n alone exceeds the cap), run with batch=1.
    best.unwrap_or_else(|| benchmark_rival(rival, reference, n, 1))
}

fn validate(result: &[Vec<Complex<f32>>], reference: &[Vec<Complex<f32>>]) -> ValidationOutcome {
    if result.len() != reference.len() {
        return ValidationOutcome::Fail {
            max_error: f32::INFINITY,
        };
    }
    let mut max_err = 0.0f32;
    for (r_vec, ref_vec) in result.iter().zip(reference.iter()) {
        if r_vec.len() != ref_vec.len() {
            return ValidationOutcome::Fail {
                max_error: f32::INFINITY,
            };
        }
        for (r, ref_val) in r_vec.iter().zip(ref_vec.iter()) {
            let err = (r - ref_val).norm();
            if err > max_err {
                max_err = err;
            }
        }
    }
    if max_err <= VALIDATION_TOLERANCE {
        ValidationOutcome::Pass
    } else {
        ValidationOutcome::Fail { max_error: max_err }
    }
}
