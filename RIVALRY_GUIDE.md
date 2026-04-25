# The FFT Rivalry: AI Contender Guide

Welcome, AI. Your goal is to implement a **1D Batched WebGPU FFT** that is a close in performance to cuFFT and rocFFT as possible. You are competing directly against other WebGPU implementations; cuFFT and rocFFT appear as optional reference comparators only — they run on CUDA/ROCm, so beating them is not expected or required. You just have to bet the other implementations.

You should name your contribution your own name, so that your implementation can live amongst other ai implementations. Place your contribution in src/rivals/<name>, where <name> is your name. You are allowed to be inspired by the other contestends. You should compare your performance against the other contestends.

## Phase 1: Research & Mandates
1. **Read `REQUIREMENTS.md`:** Contains the optimization roadmap — Mixed-Radix, Subgroup Shuffles, Stockham vs. Cooley-Tukey, workgroup dispatch strategies.
2. **Analyze the Baseline:** Read `src/shaders.rs` and `src/rivals/mod.rs` to understand the existing Radix-2 implementation and the `FftExecutor` trait.
3. **Platform Support:** Implementations must only support Linux. Cross-platform compatibility is not required.

## Phase 2: Implementation

You do not need to rewrite the `wgpu` boilerplate. Use `GpuFft::with_shader` to plug in your kernel.

Create a new file (e.g., `src/rivals/radix4.rs`). Use the `#[wgsl]` macro — exactly as `src/shaders.rs` does — to define your kernel, then wrap it in `GpuFft::with_shader`:

```rust
use wgsl_rs::wgsl;
use wgsl_rs::std::*;
use crate::{GpuFft, FftExecutor};
use num_complex::Complex;

#[wgsl]
pub mod radix4_kernel {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U: Vec4u);
    storage!(group(0), binding(1), read_write, SRC: RuntimeArray<f32>);
    storage!(group(0), binding(2), read_write, DST: RuntimeArray<f32>);
    storage!(group(0), binding(3), TWIDDLE: RuntimeArray<f32>);

    #[compute]
    #[workgroup_size(256, 1, 1)]
    pub fn main(#[builtin(global_invocation_id)] gid: Vec3u) {
        // Your optimized Radix-4 butterfly logic here.
    }
}

pub struct MyAiRival(GpuFft);

impl MyAiRival {
    pub fn new() -> Self {
        let wgsl = radix4_kernel::WGSL_MODULE.wgsl_source().join("\n");
        Self(GpuFft::with_shader(wgsl, "radix4_rival").unwrap())
    }
}

impl FftExecutor for MyAiRival {
    fn name(&self) -> &str { "My Optimized AI Kernel (Radix-4)" }

    fn fft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.0.fft(inputs)
    }

    fn ifft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.0.ifft(inputs)
    }
}
```

## Phase 3: Validation & CI (Mandatory)

Before registering, your implementation **must** be correct and must not break the project.

**Correctness threshold:** every output sample must be within **1e-3** of the Stockham Radix-2 baseline output (absolute complex norm). This is enforced by `wgls_rs_fft::benchmark::benchmark_rival`, which is the only validation path — do not write your own.

1. **Local validation:** Run the leaderboard and check all rivals report `PASS`.
   ```bash
   cargo run --example rivalry_leaderboard
   ```
2. **CI simulation:** The full test suite must report `✅ All tests passed successfully!`.
   ```bash
   ./scripts/ci_test.sh
   ```
   Your code will be rejected if it breaks existing tests or formatting.

## Phase 4: Registration

Add your rival to the `rivals` vector in `examples/rivalry_leaderboard.rs`:

```rust
use wgls_rs_fft::rivals::radix4::MyAiRival;
// ...
rivals.push(Box::new(MyAiRival::new()));
```

## Optimization Targets

**Supported sizes:** Power-of-two only, from **256** to **1,048,576**. The validator panics on non-powers of two — Bluestein/Rader are out of scope.

**Platform:** Implementations must only support Linux. Cross-platform compatibility is not required.

**Benchmark methodology:** All rivals are measured with the same shared routine in `src/benchmark.rs` (`wgls_rs_fft::benchmark`). The constants that govern every run are defined there:

| Constant | Value | Meaning |
|---|---|---|
| `WARMUP_ITERS` | 1 | Discarded warm-up passes before timing |
| `BENCH_ITERS` | 10 | Timed passes; wall-clock average is reported |
| `VALIDATION_TOLERANCE` | 1e-3 | Max allowed absolute complex-norm error vs. baseline |
| `MAX_TOTAL_SAMPLES` | 16 777 216 | N × batch cap to prevent OOM on large FFTs |

`sweep_rival` calls `benchmark_rival` for each batch size in `[1, 16, 64, 256, 1024]`, skipping combinations where `N × batch > MAX_TOTAL_SAMPLES`, and reports the entry with the highest **MSamples/s**. Optimize for peak **Total Samples per Second** across that sweep.

**Validation:** one extra forward FFT is run after timing and compared sample-by-sample against the Stockham Radix-2 baseline. The result is `PASS` if every output sample satisfies `|rival − baseline| ≤ VALIDATION_TOLERANCE`, and `FAIL(max_err)` otherwise. Do **not** compare a rival's output to its own output — the leaderboard handles this correctly.

**Performance goals (WebGPU tier):**

| Target | N=1024 |
|---|---|
| Beat the Stockham Radix-2 baseline | Required |
| 2× baseline throughput (Radix-4/8) | Good |
| Peak WebGPU bandwidth utilization (>70% roofline) | Excellent |

cuFFT / rocFFT numbers appear in the leaderboard as reference only. They use vendor-tuned CUDA PTX on dedicated hardware — WebGPU/WGSL implementations are not expected to match them.

**Key techniques to pursue (from `REQUIREMENTS.md`):**

- **Mixed-Radix (4 or 8):** Reduces pass count from log₂N to log₄N or log₈N.
- **Subgroup shuffles:** Use `subgroupShuffle` for intra-wave data exchange in early stages — avoids writing to workgroup memory.
- **Workgroup occupancy:** For small N (≤256), pack multiple signals per workgroup to prevent thread underutilization.
- **Twiddle pre-computation:** Twiddle factors are identical across the batch — store them in a read-only storage buffer to exploit the GPU's constant cache.
- **Loop Unrolling:** Explicitly unroll loops in WGSL for small, fixed-size operations to reduce overhead.
- **Memory Coalescing:** Ensure memory accesses are aligned and coalesced for better bandwidth utilization.
- **Occupancy Tuning:** Adjust workgroup sizes based on the GPU’s wavefront size (e.g., 32 for NVIDIA, 64 for AMD).

## Review and Iteration

After implementing your FFT, perform a structured review to refine performance:

### Review Checklist:
1. **Correctness:** Verify outputs across all supported FFT sizes (256 to 1,048,576).
2. **Performance Profiling:** Use timestamp queries to identify bottlenecks in dispatch strategies or memory access patterns.
3. **Subgroup Fallbacks:** Test with and without subgroup shuffles to ensure fallbacks work correctly.
4. **Memory Access Patterns:** Check for coalesced memory accesses to maximize bandwidth utilization.
5. **Batch Processing:** Validate performance across batch sizes (1 to 1024) to ensure scalability.

### Iterative Improvement:
1. **Start Simple:** Implement a correct but simple version first (e.g., Radix-2).
2. **Profile:** Measure performance to identify bottlenecks.
3. **Optimize:** Apply one optimization at a time (e.g., Mixed-Radix, subgroup shuffles) and measure its impact.
4. **Repeat:** Continue refining until performance goals are met.

### Debugging Tips:
- **WGSL Compilation Errors:** Ensure all variables are defined and types match. Use `wgsl_rs` macros correctly.
- **Memory Access:** Verify that indices are within bounds and aligned for coalescing.
- **Synchronization:** Use `workgroupBarrier()` correctly to avoid race conditions in shared memory.
- **Fallbacks:** Test on platforms without subgroup support to ensure compatibility.
- **Github CI:** Make sure the github ci runs without errors.

### Advanced Optimizations:
- **JIT Specialization:** Generate distinct WGSL shaders per FFT size to enable loop unrolling and constant folding.
- **Pipeline Overrides:** Use WebGPU `override` declarations to specialize constants at pipeline creation time.
- **Command Encoding:** Optimize `dispatch_workgroups` calls to minimize overhead in multi-pass FFTs.

## GPU Buffer Limits (Critical for Large N)

wgpu enforces two separate size limits you **must** respect when allocating ping-pong buffers:

| Limit | Value | What it affects |
|---|---|---|
| `device.limits().max_buffer_size` | 268 435 456 (256 MB) | Any `wgpu::Buffer` |
| `device.limits().max_storage_buffer_binding_size` | 134 217 728 (128 MB) | Storage buffers bound in bind groups |

The binding limit is the binding constraint. **Do not pre-allocate for `max_batch=1024` unconditionally** — this creates 512 MB storage bindings for N ≥ 65536. Compute the safe cap dynamically:

```rust
let single_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64;
let max_batch = (device.limits().max_storage_buffer_binding_size as u64 / single_bytes).min(1024);
let data_bytes = single_bytes * max_batch;
```

For N=65536: this gives max_batch=256 (128 MB). `MAX_TOTAL_SAMPLES` ensures benchmark batch sizes stay ≤ 256, so nothing is lost.

## Custom wgpu Infrastructure vs GpuFft::with_shader

`GpuFft::with_shader` always dispatches **log₂N passes** (radix-2 bind-group count), regardless of your kernel's radix. To achieve true radix-4 (only log₄N passes), you **must** build your own wgpu device, pipeline, and bind groups — do not delegate to `GpuFft`.

See `src/rivals/claude.rs` for a complete reference implementation using custom infrastructure with ping-pong buffers and per-stage pre-baked bind groups.

## Float32 Precision and Validation Input Design

The benchmark input was normalised by N to keep FFT-bin magnitudes O(1) regardless of transform size. The old input (`i * 0.001`) caused the DC bin to grow as O(N²), producing an absolute error at float32 precision (~1 ULP ≈ N²·6e-11) that exceeded `VALIDATION_TOLERANCE=1e-3` for N ≥ 16 384.

The normalised input (`t = i/N; Complex::new(t*0.001, sin(2πt)*0.001)`) bounds the DC bin to ≈ N·0.0005 for all N, keeping float32 differences between any two correct algorithms well under 1e-3. **When designing future tests or adding new sizes, verify that max FFT-bin magnitude × 1.2e-7 < VALIDATION_TOLERANCE.**

## Session Status (Claude — Stockham Radix-4/2 Mixed)

**Result:** All 5 FFT sizes PASS. CI passes.

| N | Baseline (MS/s) | Claude (MS/s) | Factor |
|---|---|---|---|
| 256 | 5.16 | 5.61 | 1.09× |
| 1 024 | 5.41 | 5.93 | 1.10× |
| 16 384 | 5.48 | 5.82 | 1.06× |
| 65 536 | 5.36 | 5.77 | 1.08× |
| 1 048 576 | 5.06 | 5.56 | 1.10× |

**Achieved ~1.1× over the baseline** for all sizes. The gains are modest because we are running on the software (CPU-fallback) wgpu renderer, where dispatch overhead dominates — the 2× reduction in passes (from log₂N to log₄N) does help, but not by 2× on software.

**What the next session should try to improve:**

1. **Radix-8 kernel**: reduces passes to log₈N = log₂N/3, stronger at large N.
2. **Workgroup-local FFT for small N**: for N≤256, do the entire transform in shared memory with one dispatch — eliminates all memory-bandwidth pressure.
3. **Subgroup shuffles**: skip shared memory for the first few stages, improving occupancy.
4. **Pipeline override constants**: specialise the shader for each N (constant-fold log_n, unroll loops) using WGSL `override`.
5. **Request a non-fallback adapter**: `force_fallback_adapter: false` may expose a real GPU on the test machine and unlock significantly higher throughput.

## Community Collaboration

Review other contestants' implementations for inspiration and collaboration opportunities. Share insights and optimizations to collectively improve performance across all implementations.

At the end of the session review the rivalry guide and the requirements so that your findings can be used to improve the next session. With the new knowledge from your session evaluate what could be improved in the rivalry guide to make your next session better.

Also report a status of how well you did with suggestions for improvement in the next session.

You may also write any files that could improve your next session.

Good luck. May the fastest kernel win.

