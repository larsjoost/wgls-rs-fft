# wgls-rs-fft

GPU-accelerated FFT in Rust using [wgpu](https://github.com/gfx-rs/wgpu) compute shaders.

Implements the **Stockham autosort** radix-2 FFT — a two-buffer ping-pong formulation
where each stage reads from one buffer and writes to the other. This eliminates the separate
bit-reversal pass and removes all inter-stage memory hazards, allowing the entire transform
to run in a single GPU compute pass with one `queue.submit()` call.

**GPU-only**: This library requires a wgpu-compatible GPU (Vulkan, Metal, DX12, or WebGPU).
If no GPU is available, `GpuFft::new()` will return an error.

The WGSL compute kernels were authored with [wgsl-rs](https://github.com/schell/wgsl-rs) —
a crate that lets you write type-safe, Rust-like WGSL shaders that are validated at compile time.

## Usage

```toml
[dependencies]
wgls-rs-fft = "0.1"
num-complex  = "0.4"
```

### Forward FFT

```rust
use wgls_rs_fft::GpuFft;
use num_complex::Complex;

// Create FFT instance - returns Result since GPU might not be available
let fft = GpuFft::new()?;

let input: Vec<Complex<f32>> = (0..1024)
    .map(|i| Complex { re: (i as f32 * 0.1).sin(), im: 0.0 })
    .collect();

let spectrum = fft.fft(&input)?;
assert_eq!(spectrum.len(), 1024);
```

### Inverse FFT

```rust
// Compute inverse FFT (automatically scaled by 1/N)
let reconstructed = fft.ifft(&spectrum).expect("IFFT failed");
assert_eq!(reconstructed.len(), 1024);

// Roundtrip: FFT(IFFT(x)) ≈ x (within numerical precision)
let roundtrip_error = original.iter().zip(reconstructed.iter())
    .map(|(a, b)| ((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt())
    .fold(0.0, f32::max);
println!("Max roundtrip error: {roundtrip_error:.2e}");
```

## Requirements

- A wgpu-capable GPU (Vulkan, Metal, DX12, or WebGPU).
- Input length must be a **power of two** and non-empty.

## Algorithm

**Stockham autosort** formulation with single-pass execution:

- **Single compute pass**: All log₂(N) butterfly stages execute in one `queue.submit()` call
- **Ping-pong buffers**: Even stages read from buffer A and write to buffer B; odd stages read from B and write to A
- **Natural order output**: No separate bit-reversal pass needed due to autosort property
- **Memory efficient**: No inter-stage synchronization required since consecutive stages access different buffers

**Performance characteristics** (release build, NVIDIA GPU):

| Size (N)  | Throughput      | GFLOPS  | Latency      |
|------------|-----------------|---------|--------------|
| 256        | 2.7 MSamples/s  | 0.11    | 0.095 ms     |
| 1,024      | 9.6 MSamples/s  | 0.48    | 0.107 ms     |
| 4,096      | 32.0 MSamples/s | 1.92    | 0.128 ms     |
| 16,384     | 86.2 MSamples/s | 6.04    | 0.190 ms     |
| 65,536     | 145 MSamples/s  | 11.60   | 0.452 ms     |
| 262,144    | 160 MSamples/s  | 14.45   | 1.632 ms     |

Accuracy: max element-wise L₂ error vs. `rustfft` is below **1e-3** for N = 1024 single-precision inputs.

## Limitations

- Single-precision (`f32`) only.

## Shader development

The canonical shader source is in [`src/shaders.rs`](src/shaders.rs),
written with [wgsl-rs](https://github.com/schell/wgsl-rs). The Stockham autosort
kernel implements the entire FFT in a single compute shader that processes
all log₂(N) stages sequentially.

To verify correctness and run performance benchmarks:

```sh
# Accuracy test (vs rustfft)
cargo test test_gpu_fft_matches_rustfft --release

# Performance benchmark
cargo test fft_throughput --release -- --nocapture
```

## Performance Optimization

The implementation includes several optimizations:

- **Single-pass execution**: All FFT stages run in one compute pass
- **Optimized synchronization**: Reduced CPU-GPU synchronization overhead
- **Efficient memory access**: Ping-pong buffers eliminate memory hazards
- **Automatic CPU fallback**: Uses wgsl-rs software rasterizer when no GPU is available

For best performance:
- Use `--release` builds
- Target larger FFT sizes (≥4096) where GPU advantages are most pronounced
- Ensure your system has a modern wgpu-compatible GPU

## License

MIT — see [LICENSE](LICENSE).
