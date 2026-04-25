use std::env;
use wgls_rs_fft::{
    benchmark::{benchmark_rival, ValidationOutcome},
    FftExecutor, GpuFft,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run --example find_optimal_batch <N> [rival_name]");
        println!("Example: cargo run --example find_optimal_batch 1024");
        return;
    }

    let n: usize = args[1].parse().expect("N must be a positive integer");
    let target_rival = args.get(2);

    println!("\n=== OPTIMAL BATCH SIZE FINDER (N = {}) ===", n);

    let reference = GpuFft::new().expect("Failed to init WebGPU reference");

    let mut rivals: Vec<Box<dyn FftExecutor>> = Vec::new();
    rivals.push(Box::new(GpuFft::new().expect("Failed to init WebGPU")));
    rivals.push(Box::new(wgls_rs_fft::rivals::radix4::Radix4Rival::new()));
    rivals.push(Box::new(wgls_rs_fft::rivals::claude::ClaudeFft::new()));
    rivals.push(Box::new(wgls_rs_fft::rivals::gemini::GeminiFft::new()));

    let batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

    for rival in &rivals {
        let name = rival.name();
        if let Some(target) = target_rival {
            if !name.to_lowercase().contains(&target.to_lowercase()) {
                continue;
            }
        }

        println!("\n--- Rival: {} ---", name);
        println!(
            "{:>8} | {:>14} | {:>10} | {:>8}",
            "Batch", "MSamples/s", "GFLOPS", "Status"
        );
        println!("{}", "-".repeat(48));

        let mut best_msps = 0.0;
        let mut best_batch = 0;

        for &batch in &batch_sizes {
            if n * batch > wgls_rs_fft::benchmark::MAX_TOTAL_SAMPLES {
                continue;
            }

            let result = benchmark_rival(rival.as_ref(), &reference, n, batch);
            let status = match &result.validation {
                ValidationOutcome::Pass => "PASS".to_string(),
                ValidationOutcome::Fail { max_error } => format!("FAIL({:.2e})", max_error),
            };

            if result.msamples_per_sec > best_msps {
                best_msps = result.msamples_per_sec;
                best_batch = batch;
            }

            println!(
                "{:>8} | {:>14.2} | {:>10.2} | {:>8}",
                batch, result.msamples_per_sec, result.gflops, status,
            );
        }

        println!(
            "Best batch size for {}: {} ({:.2} MSamples/s)",
            name, best_batch, best_msps
        );
    }
}
