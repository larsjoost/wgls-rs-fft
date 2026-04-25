use wgls_rs_fft::{
    benchmark::{sweep_rival, ValidationOutcome},
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

    #[cfg(feature = "cuda")]
    if let Ok(cufft) = CuFft::new(1024) {
        rivals.push(Box::new(cufft));
    }

    let fft_sizes = [256, 1024, 16384, 65536, 1048576];
    let batch_sizes = [1, 16, 64, 256, 1024];

    for &n in &fft_sizes {
        println!("\n--- N = {} ---", n);
        println!(
            "{:>32} | {:>8} | {:>14} | {:>10} | {:>8}",
            "Implementation", "Batch", "MSamples/s", "GFLOPS", "Status"
        );
        println!("{}", "-".repeat(84));

        for rival in &rivals {
            let result = sweep_rival(rival.as_ref(), &reference, n, &batch_sizes);
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
