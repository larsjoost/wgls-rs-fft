## Codex Session Report

### What Was Implemented
- Added new rival implementation: `src/rivals/codex.rs`.
- Strategy: cloned the validated Stockham mixed radix (radix-4 + radix-2 tail) structure and preserved kernel math/path for correctness.
- Adapter policy changed to:
  1. Prefer `force_fallback_adapter: false` (try real GPU first).
  2. Fall back to `force_fallback_adapter: true` if unavailable.
- Registered rival in:
  - `src/rivals/mod.rs`
  - `examples/rivalry_leaderboard.rs`

### Why This Change
- Existing rivals force fallback adapter use; this can hide hardware gains on machines where a real GPU is available.
- This variant keeps known-correct kernels while giving a better chance of improved throughput in hardware-backed environments.

### Validation Plan (Guide Compliance)
- Run leaderboard:
  - `cargo run --example rivalry_leaderboard`
- Run CI simulation:
  - `./scripts/ci_test.sh`

### Suggestions For Next Session
- Implement true radix-8 with verified butterfly equations and staged validation against baseline at each N.
- Add subgroup feature-gated kernels (`wgpu::Features::SUBGROUP`) with runtime fallback.
- Add size-specialized pipelines for `N` buckets (or per-N JIT) to improve unrolling and reduce control overhead.
- Add one-workgroup local-memory full FFT path for `N <= 256`.

### Suggested Improvements To `RIVALRY_GUIDE.md`
- Add an explicit minimal “submission checklist” block:
  1. new rival file
  2. registry update
  3. leaderboard registration
  4. leaderboard pass snapshot
  5. CI pass snapshot
- Add a canonical “naming convention” example (`<agent_name>.rs`, struct name, display name format).
- Add an explicit note on expected runtime cost of full leaderboard + CI to encourage planning.
- Add a small “known-good equations” appendix for radix-8 outputs to reduce incorrect butterfly variants.
