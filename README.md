# wgls-rs-fft

GPU-accelerated FFT in Rust using [wgpu](https://github.com/gfx-rs/wgpu) compute shaders.

Implements the **Cooley-Tukey radix-2 decimation-in-time (DIT) FFT** entirely on
the GPU. The WGSL compute kernels were authored with
[wgsl-rs](https://github.com/schell/wgsl-rs) — a crate that lets you write
type-safe, Rust-like WGSL shaders that are validated at compile time.

## Usage

```toml
[dependencies]
wgls-rs-fft = "0.1"
num-complex  = "0.4"
```

```rust
use wgls_rs_fft::GpuFft;
use num_complex::Complex;

let fft = GpuFft::new().expect("no GPU available");

let input: Vec<Complex<f32>> = (0..1024)
    .map(|i| Complex { re: (i as f32 * 0.1).sin(), im: 0.0 })
    .collect();

let spectrum = fft.fft(&input).expect("FFT failed");
assert_eq!(spectrum.len(), 1024);
```

## Requirements

- A wgpu-capable GPU (Vulkan, Metal, DX12, or WebGPU).
- Input length must be a **power of two** and non-empty.

## Algorithm

Two compute passes are dispatched per call:

1. **Bit-reversal** — each thread swaps sample *i* with its bit-reversed
   counterpart; pairs are only swapped once (guarded by `j > i`).
2. **Butterfly stages** — log₂(N) passes of the standard Cooley-Tukey
   butterfly. One `queue.submit` is issued per stage so the uniform-buffer
   stage index is visible to the GPU before each dispatch.

Accuracy: max element-wise L₂ error vs. `rustfft` is below **1e-3** for
N = 1024 single-precision inputs.

## Limitations

- Forward (analysis) FFT only; inverse FFT is not yet implemented.
- Single-precision (`f32`) only.

## Shader development

The canonical shader source is in [`src/shaders.rs`](src/shaders.rs),
written with [wgsl-rs](https://github.com/schell/wgsl-rs). The pre-generated
WGSL shipped in the crate lives in
[`src/bit_reverse.wgsl`](src/bit_reverse.wgsl) and
[`src/fft_stage.wgsl`](src/fft_stage.wgsl).

To verify the shaders and run the GPU vs CPU accuracy test:

```sh
cargo test
```

## License

MIT — see [LICENSE](LICENSE).
