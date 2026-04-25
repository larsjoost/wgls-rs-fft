use crate::{FftExecutor, GpuFft};
use num_complex::Complex;

pub struct StockhamBaseline(GpuFft);

impl StockhamBaseline {
    pub fn new() -> Self {
        Self(GpuFft::new().unwrap())
    }
}

impl FftExecutor for StockhamBaseline {
    fn name(&self) -> &str {
        "Baseline (Stockham Radix-2)"
    }
    fn fft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.0.fft(inputs)
    }
    fn ifft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.0.ifft(inputs)
    }
}
