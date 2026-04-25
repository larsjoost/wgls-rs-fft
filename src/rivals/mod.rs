//! # The FFT Rivalry Registry
//!
//! To add a new AI implementation to the rivalry:
//! 1. Create a new file in this directory (e.g., `my_ai_fft.rs`).
//! 2. Implement the `FftExecutor` trait.
//! 3. Register your rival in `examples/rivalry_leaderboard.rs`.

pub mod baseline;
pub mod claude;
pub mod codex;
pub mod gemini;
pub mod mistral_vibe;
pub mod radix4;
