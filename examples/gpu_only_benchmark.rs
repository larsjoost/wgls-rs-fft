use wgls_rs_fft::{
    benchmark::{benchmark_gpu_only, benchmark_rival, MAX_TOTAL_SAMPLES},
    GpuFft,
};

fn main() {
    println!("\n=== GPU-ONLY PERFORMANCE BENCHMARK ===");
    println!("This benchmark isolates GPU compute + DMA performance, excluding CPU overhead");

    let gpu_fft = GpuFft::new().expect("Failed to init WebGPU");
    let reference = GpuFft::new().expect("Failed to init WebGPU reference");

    let fft_sizes = [256, 1024, 16384, 65536, 1048576];

    for &n in &fft_sizes {
        let batch_size = choose_batch_size(n);
        println!("\n--- N = {} ---", n);

        // Full benchmark (includes CPU overhead)
        let full_result = benchmark_rival(&gpu_fft, &reference, n, batch_size);

        // GPU-only benchmark (isolated GPU + DMA)
        let gpu_only_result =
            benchmark_gpu_only(&gpu_fft, n, batch_size).expect("GPU-only benchmark failed");

        println!(
            "{:>40} | {:>8} | {:>14} | {:>10} | {:>14} | {:>10}",
            "Implementation", "Batch", "Full MS/s", "Full GFLOPS", "GPU MS/s", "GPU GFLOPS"
        );
        println!("{}", "-".repeat(110));

        println!(
            "{:>40} | {:>8} | {:>14.2} | {:>10.2} | {:>14.2} | {:>10.2}",
            "WebGPU (Full Pipeline)",
            batch_size,
            full_result.msamples_per_sec,
            full_result.gflops,
            "N/A",
            "N/A"
        );

        println!(
            "{:>40} | {:>8} | {:>14} | {:>10} | {:>14.2} | {:>10.2}",
            "WebGPU (GPU-only)",
            batch_size,
            "N/A",
            "N/A",
            gpu_only_result.gpu_msamples_per_sec,
            gpu_only_result.gpu_gflops
        );

        // Calculate overhead percentage
        let full_duration = 1.0 / full_result.msamples_per_sec * 1_000_000.0;
        let gpu_duration = gpu_only_result.gpu_duration_sec * 1_000_000.0; // convert to microseconds
        let overhead_pct = ((full_duration - gpu_duration) / full_duration) * 100.0;

        println!("\nOverhead analysis (N={}):", n);
        println!("  Full pipeline duration: {:.2} μs", full_duration);
        println!("  GPU-only duration: {:.2} μs", gpu_duration);
        println!("  CPU/DMA overhead: {:.2}%", overhead_pct);
    }
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
