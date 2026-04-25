use wgls_rs_fft::{
    benchmark::{benchmark_rival, ValidationOutcome, MAX_TOTAL_SAMPLES},
    FftExecutor, GpuFft,
};

#[cfg(feature = "cuda")]
use wgls_rs_fft::CuFft;

fn main() {
    println!("\n=== WGSL-RS FFT RIVALRY LEADERBOARD ===");

    // Baseline is the single source of truth for validation.
    let reference = GpuFft::new().expect("Failed to init WebGPU reference");

    let mut rivals: Vec<Box<dyn FftExecutor>> = Vec::new();
    rivals.push(Box::new(GpuFft::new().expect("Failed to init WebGPU")));
    rivals.push(Box::new(wgls_rs_fft::rivals::radix4::Radix4Rival::new()));
    rivals.push(Box::new(wgls_rs_fft::rivals::claude::ClaudeFft::new()));
    rivals.push(Box::new(wgls_rs_fft::rivals::codex::CodexFft::new()));
    rivals.push(Box::new(wgls_rs_fft::rivals::gemini::GeminiFft::new()));
    rivals.push(Box::new(
        wgls_rs_fft::rivals::mistral_vibe::MistralVibeFft::new(),
    ));

    #[cfg(feature = "cuda")]
    if let Ok(cufft) = CuFft::new(1024) {
        rivals.push(Box::new(cufft));
    }

    let fft_sizes = [256, 1024, 16384, 65536, 1048576];

    for &n in &fft_sizes {
        let batch_size = choose_batch_size(n);
        println!("\n--- N = {} ---", n);
        println!(
            "{:>32} | {:>8} | {:>14} | {:>10} | {:>8}",
            "Implementation", "Batch", "MSamples/s", "GFLOPS", "Status"
        );
        println!("{}", "-".repeat(84));

        for rival in &rivals {
            let result = benchmark_rival(rival.as_ref(), &reference, n, batch_size);
            let status = match &result.validation {
                ValidationOutcome::Pass => "PASS".to_string(),
                ValidationOutcome::Fail { max_error } => format!("FAIL({:.2e})", max_error),
            };
            println!(
                "{:>32} | {:>8} | {:>14.2} | {:>10.2} | {:>8}",
                result.rival_name,
                result.batch_size,
                result.msamples_per_sec,
                result.gflops,
                status,
            );
        }
    }

    println!("\nAdd your implementation to src/rivals/ and register it above to compete.");
}

fn choose_batch_size(n: usize) -> usize {
    let preferred = if n <= 1024 {
        1024
    } else if n <= 16384 {
        256
    } else if n <= 65536 {
        64
    } else if n <= 262_144 {
        16
    } else {
        1
    };

    let mut batch = preferred;
    while n.saturating_mul(batch) > MAX_TOTAL_SAMPLES && batch > 1 {
        batch /= 2;
    }
    batch.max(1)
}
