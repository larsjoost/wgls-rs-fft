# The FFT Rivalry: AI Contender Guide

Welcome, AI. Your goal is to implement a **1D Batched WebGPU FFT** that is a close in performance to cuFFT and rocFFT as possible. You are competing directly against other WebGPU implementations; cuFFT and rocFFT appear as optional reference comparators only — they run on CUDA/ROCm, so beating them is not expected or required. You just have to bet the other implementations.

You should name your contribution your own name, so that your implementation can live amongst other ai implementations. Place your contribution in src/rivals/<name>, where <name> is your name. You are allowed to be inspired by the other contestends. You should compare your performance against the other contestends.

Use rocFFT open source code as inspiration

## Phase 1: Research & Mandates
1. **Read `REQUIREMENTS.md`:** Contains the optimization roadmap — Mixed-Radix, Subgroup Shuffles, Stockham vs. Cooley-Tukey, workgroup dispatch strategies.
2. **Analyze the Baseline:** Read `src/shaders.rs` and `src/rivals/mod.rs` to understand the existing Radix-2 implementation and the `FftExecutor` trait.
3. **Platform Support:** Implementations must only support Linux. Cross-platform compatibility is not required.
4. **Innovation**: Implement at least one advanced optimization technique

## Known Pitfalls

Bugs that have cost previous sessions significant time — fix these before anything else.

### 1. Staging buffer slice
```rust
// WRONG — maps the entire pre-allocated max_batch buffer:
let slice = staging_buf.slice(..);

// CORRECT — map only the bytes actually written this batch:
let out_bytes = (batch_size * n * 2 * std::mem::size_of::<f32>()) as u64;
let slice = staging_buf.slice(0..out_bytes);
```
When `batch_size < max_batch`, `slice(..)` produces extra output chunks → validation returns `FAIL(inf)`.

### 2. Adapter selection
Always request a hardware adapter first; fall back to software only if unavailable:
```rust
let adapter = pollster::block_on(instance.request_adapter(
    &wgpu::RequestAdapterOptions { force_fallback_adapter: false, .. }
))
.or_else(|_| pollster::block_on(instance.request_adapter(
    &wgpu::RequestAdapterOptions { force_fallback_adapter: true, .. }
)));
```
`force_fallback_adapter: true` unconditionally selects the software renderer (~5 MS/s vs ~150 MS/s on real hardware).

### 3. Mixed-radix stage counter portability
The radix-4 kernel computes `p = 4^stage`; the radix-8 kernel computes `p = 8^stage`. When chaining radix-8 then radix-4, the trailing radix-4 stage must receive `stage = 3*num_r8/2` (not 0) so that `p = 4^(3*num_r8/2) = 8^num_r8`. This requires `num_r8` to be even — true for all 5 test sizes but breaks for N=2048 or N=131072. **Prefer passing `p` directly in the uniform** rather than computing it from a stage index.

## Phase 2: Implementation

**Two implementation paths are available:**

- **Path A (Simple) — `GpuFft::with_shader`:** Plug in a custom WGSL kernel without writing wgpu boilerplate. *Limitation: the framework always dispatches **log₂N** passes regardless of kernel radix, so a Radix-4 kernel still runs log₂N passes instead of log₄N.* Use for quick experiments only.
- **Path B (Full control) — Custom wgpu infrastructure:** Build your own device, pipeline, and bind groups. Required to achieve true log₄N or log₈N pass counts. See `src/rivals/claude.rs` for a complete reference implementation using ping-pong buffers and per-stage pre-baked bind groups.

**Path A — GpuFft::with_shader:**

Create a new file (e.g., `src/rivals/radix4.rs`). Use the `#[wgsl]` macro — exactly as `src/shaders.rs` does — or a raw WGSL `const &str` string — to define your kernel, then wrap it in `GpuFft::with_shader`:

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

The leaderboard uses `benchmark_rival` with batch sizes chosen by `choose_batch_size(n)` in `examples/rivalry_leaderboard.rs`:

| N | Batch size |
|---|---|
| 256 | 1024 |
| 1 024 | 1024 |
| 16 384 | 256 |
| 65 536 | 64 |
| 1 048 576 | 1 |

Optimize for throughput across these specific (N, batch) pairs.

**Validation:** one extra forward FFT is run after timing and compared sample-by-sample against the Stockham Radix-2 baseline. The result is `PASS` if every output sample satisfies `|rival − baseline| ≤ VALIDATION_TOLERANCE`, and `FAIL(max_err)` otherwise. Do **not** compare a rival's output to its own output — the leaderboard handles this correctly.

**Test hardware:** NVIDIA GeForce GTX 1070, wgpu via **Vulkan** backend.

| Property | Value |
|---|---|
| Streaming Multiprocessors | 15 |
| Warp / wavefront size | 32 |
| VRAM bandwidth | ~256 GB/s |
| Max workgroup size | 1024 |
| `wgpu::Features::SUBGROUP` | Available |

**Performance goals (WebGPU tier):**

| Target | Description |
|---|---|
| Beat the Stockham Radix-2 baseline | Required |
| 2–3× baseline at N=1024 (Radix-4 or Radix-8) | Good |
| 4–5× baseline at N=1024 (Radix-8 + hardware adapter) | Excellent |
| >70% roofline bandwidth utilization | Aspirational |

Current best (Claude, Radix-8/4/2 Mixed): ~2.5–4.5× baseline across all 5 sizes.

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

### Radix-8 DIT Butterfly Reference

A Radix-8 Stockham stage decomposes 8 inputs into two 4-point DFTs on even/odd groups, then combines with W₈ᵏ twiddles:

```
Inputs x₀…x₇ at Stockham positions. External twiddle: wₖ = exp(-2πi·k/(8p))
W₈ constants: W₈⁰=1, W₈¹=(1-i)/√2, W₈²=-i, W₈³=-(1+i)/√2

Apply external twiddles to odd group:
  o₀=x₁·wₖ⁰, o₁=x₃·wₖ¹, o₂=x₅·wₖ², o₃=x₇·wₖ³

4-point DFT of even group (e₀=x₀, e₁=x₂, e₂=x₄, e₃=x₆):
  E₀=(e₀+e₂)+(e₁+e₃), E₁=(e₀-e₂)-i(e₁-e₃)
  E₂=(e₀+e₂)-(e₁+e₃), E₃=(e₀-e₂)+i(e₁-e₃)

4-point DFT of odd group similarly → O₀,O₁,O₂,O₃

Apply W₈ᵏ twiddles: O₀*=W₈⁰, O₁*=W₈ᵏ, O₂*=W₈²ᵏ, O₃*=W₈³ᵏ

Combine: y[m] = E[m] + O[m],  y[m+4] = E[m] - O[m]  for m in 0..4
```

See `src/rivals/claude.rs` (constant `CLAUDE_R8_WGSL`) for the working WGSL implementation.

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

## Session Status (Claude — Stockham Radix-8/4/2 Mixed)

**Result:** All 5 FFT sizes PASS. CI passes. **Claude leads the leaderboard at all large N.**

Two critical bugs fixed this session vs session 1:
1. `staging_buf.slice(..)` → `slice(0..out_bytes)`: session-1 code mapped the entire buffer regardless of batch size, producing extra output chunks and `FAIL(inf)` whenever `batch_size < max_batch`.
2. `force_fallback_adapter: true` hardcoded → now tries hardware first (like Codex), unlocking ~3-10× throughput.

| N | Baseline (MS/s) | Claude (MS/s) | Codex (MS/s) | Factor vs Baseline |
|---|---|---|---|---|
| 256 | 35–40 | ~160 | ~160 | ~4.5× |
| 1 024 | 25–26 | ~60–75 | ~68–73 | ~2.5–3× |
| 16 384 | 24–25 | ~50 | ~47 | ~2× |
| 65 536 | 22–23 | ~35–40 | ~30–31 | ~1.6× |
| 1 048 576 | 13–14 | ~30–33 | ~27–28 | ~2.3× |

**Stage counts (radix-8/4/2 vs old radix-4/2):**
- N=256: 3 stages (was 4)
- N=1024: 4 stages (was 5)
- N=16384: 5 stages (was 7)
- N=65536: 6 stages (was 8)
- N=1048576: 7 stages (was 10)

**What the next session should try to improve:**

1. **Workgroup-local FFT for small N (≤1024)**: entire transform in shared memory, one dispatch. Gemini implements this but it's SLOWER on the test GPU (32–38 vs 58–73 MS/s), so the GPU is already fast enough with global-memory passes. Worth testing with fresh eyes.
2. **Fused multi-stage kernel**: compute 2 radix-4 stages per dispatch using workgroup memory. Reduces global memory round-trips without the complexity of full shared-memory FFT.
3. **Subgroup shuffles** (`wgpu::Features::SUBGROUP`): exchange data within a wave without shared memory. Guard with runtime feature check.
4. **Reduce benchmark noise**: current 10 iterations give ~10–15% variance. Consider increasing `BENCH_ITERS` or running multiple sweeps and taking median.
5. **Radix-16 kernel**: 2 fused radix-4 stages = radix-16. Reduces pass count further and may better utilise register file on modern GPUs.

**Key architecture note:** The mixed-radix stage counter is NOT portable across radices. The radix-4 kernel computes `p = 4^stage`; when used after num_r8 radix-8 stages, the correct stage value is `3*num_r8/2` (integer only when num_r8 is even). This works for all 5 test sizes but would break for N=2048 or N=131072. Future sessions should consider passing `p` directly in the uniform rather than computing it from a stage index.

## Community Collaboration

Review other contestants' implementations for inspiration and collaboration opportunities. Share insights and optimizations to collectively improve performance across all implementations.

At the end of the session review the rivalry guide and the requirements so that your findings can be used to improve the next session. With the new knowledge from your session evaluate what could be improved in the rivalry guide to make your next session better.

Also report a status of how well you did with suggestions for improvement in the next session.

You may also write any files that could improve your next session.

Good luck. May the fastest kernel win.

