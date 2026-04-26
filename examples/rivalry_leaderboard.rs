use std::process::Command;
use wgls_rs_fft::{
    benchmark::{benchmark_rival, ValidationOutcome, MAX_TOTAL_SAMPLES},
    FftExecutor, GpuFft,
};

#[cfg(feature = "cuda")]
use wgls_rs_fft::CuFft;

#[cfg(feature = "hipfft")]
use wgls_rs_fft::HipFft;

/// Detect if NVIDIA GPU is available for cuFFT
#[cfg(feature = "cuda")]
fn probe_cufft_runtime() -> Result<(), String> {
    use std::panic;

    match panic::catch_unwind(|| CuFft::new(256)) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("panic during cuFFT initialization".to_string()),
    }
}

#[cfg(not(feature = "cuda"))]
fn probe_cufft_runtime() -> Result<(), String> {
    Err("binary was built without --features cuda".to_string())
}

fn is_host_nvidia_present() -> bool {
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=name")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let name = String::from_utf8_lossy(&output.stdout);
            if !name.trim().is_empty() {
                return true;
            }
        }
    }

    if let Ok(output) = Command::new("lspci").output() {
        if output.status.success() {
            let listing = String::from_utf8_lossy(&output.stdout);
            return listing.to_lowercase().contains("nvidia");
        }
    }

    false
}

fn main() {
    println!("\n=== WGSL-RS FFT RIVALRY LEADERBOARD ===");

    // Distinguish host GPU detection from cuFFT runtime availability.
    let host_nvidia_present = is_host_nvidia_present();
    let cufft_probe = probe_cufft_runtime();
    let cufft_available = cufft_probe.is_ok();
    if cufft_available {
        println!("🟢 NVIDIA GPU detected - cuFFT benchmarks will be included");
    } else if host_nvidia_present {
        if cfg!(feature = "cuda") {
            println!("🟡 NVIDIA GPU detected, but cuFFT runtime init failed");
            if let Err(err) = &cufft_probe {
                println!("   cuFFT init error: {err}");
                if err.contains("CUFFT_NOT_SUPPORTED") {
                    println!("   Hint: try launching with LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH");
                }
            }
            println!("   cuFFT benchmarks will be skipped for this run");
        } else {
            println!("🟡 NVIDIA GPU detected, but this binary was built without --features cuda");
            println!("   cuFFT benchmarks will be skipped");
        }
    } else {
        println!("🔴 No NVIDIA GPU detected - cuFFT benchmarks will be skipped");
        println!("   To enable cuFFT: build with --features cuda and ensure NVIDIA drivers are installed");
    }

    // Baseline is the single source of truth for validation.
    let reference = GpuFft::new().expect("Failed to init WebGPU reference");

    let mut rivals: Vec<Box<dyn FftExecutor>> = Vec::new();
    rivals.push(Box::new(GpuFft::new().expect("Failed to init WebGPU")));
    rivals.push(Box::new(wgls_rs_fft::rivals::radix4::Radix4Rival::new()));
    rivals.push(Box::new(
        wgls_rs_fft::rivals::radix4_proper::Radix4ProperFft::new(),
    ));
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

    #[cfg(feature = "hipfft")]
    if let Ok(hipfft) = HipFft::new(1024) {
        rivals.push(Box::new(hipfft));
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

        // Add cuFFT benchmark for this size when cuFFT runtime is available.
        if cufft_available {
            #[cfg(feature = "cuda")]
            {
                if let Ok(cufft) = CuFft::new(n) {
                    let result = benchmark_rival(&cufft, &reference, n, batch_size);
                    let status = match &result.validation {
                        ValidationOutcome::Pass => "PASS".to_string(),
                        ValidationOutcome::Fail { max_error } => format!("FAIL({:.2e})", max_error),
                    };
                    println!(
                        "{:>32} | {:>8} | {:>14.2} | {:>10.2} | {:>8}",
                        "cuFFT (NVIDIA Gold Standard)",
                        result.batch_size,
                        result.msamples_per_sec,
                        result.gflops,
                        status,
                    );
                }
            }
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
