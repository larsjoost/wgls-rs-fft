use std::any::Any;

use crate::FftExecutor;
use cudarc::cufft::{sys, CudaFft, FftDirection};
use cudarc::driver::{CudaContext, CudaStream};
use num_complex::Complex;
use std::error::Error;
use std::io::{Error as IoError, ErrorKind};
use std::sync::Arc;

/// Wrapper for cuFFT (via cudarc) to compare performance with our WebGPU implementation.
pub struct CuFft {
    plan: CudaFft,
    stream: Arc<CudaStream>,
    fft_size: usize,
}

impl FftExecutor for CuFft {
    fn name(&self) -> &str {
        "cuFFT (NVIDIA Gold Standard)"
    }

    fn fft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn Error>> {
        self.batch_fft(inputs)
    }

    fn ifft(&self, inputs: &[Vec<Complex<f32>>]) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn Error>> {
        self.batch_ifft(inputs)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Note: cuFFT uses a completely different architecture (CUDA vs wgpu)
// so it needs a specialized GPU-only benchmark implementation
// For now, cuFFT will use the full benchmarking path

impl CuFft {
    /// Create a new cuFFT instance.
    pub fn new(fft_size: usize) -> Result<Self, Box<dyn Error>> {
        if fft_size == 0 {
            return Err(IoError::new(ErrorKind::InvalidInput, "fft_size must be > 0").into());
        }

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let plan = CudaFft::plan_1d(
            fft_size as i32,
            sys::cufftType::CUFFT_C2C,
            1,
            stream.clone(),
        )?;

        Ok(Self {
            plan,
            stream,
            fft_size,
        })
    }

    fn ensure_input_size(&self, input: &[Complex<f32>]) -> Result<(), Box<dyn Error>> {
        if input.len() != self.fft_size {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                format!(
                    "input length {} does not match planned fft_size {}",
                    input.len(),
                    self.fft_size
                ),
            )
            .into());
        }
        Ok(())
    }

    fn validate_batch_inputs(inputs: &[Vec<Complex<f32>>]) -> Result<usize, Box<dyn Error>> {
        if inputs.is_empty() {
            return Ok(0);
        }

        let n = inputs[0].len();
        if n == 0 {
            return Err(IoError::new(
                ErrorKind::InvalidInput,
                "batch input vectors must be non-empty",
            )
            .into());
        }

        for (idx, input) in inputs.iter().enumerate() {
            if input.len() != n {
                return Err(IoError::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "all batch inputs must have the same length; input[0]={n}, input[{idx}]={}",
                        input.len()
                    ),
                )
                .into());
            }
        }

        Ok(n)
    }

    fn to_cuda_complex(values: &[Complex<f32>]) -> Vec<sys::float2> {
        values
            .iter()
            .map(|c| sys::float2 { x: c.re, y: c.im })
            .collect()
    }

    fn from_cuda_complex(values: &[sys::float2]) -> Vec<Complex<f32>> {
        values.iter().map(|c| Complex::new(c.x, c.y)).collect()
    }

    fn scale_inverse(output: &mut [Complex<f32>], n: usize) {
        let scale = 1.0 / n as f32;
        for c in output {
            c.re *= scale;
            c.im *= scale;
        }
    }

    /// Perform forward FFT.
    pub fn fft(&self, input: &[Complex<f32>]) -> Result<Vec<Complex<f32>>, Box<dyn Error>> {
        self.ensure_input_size(input)?;

        let host_input = Self::to_cuda_complex(input);
        let mut d_input = self.stream.clone_htod(&host_input)?;
        let mut d_output = self.stream.alloc_zeros::<sys::float2>(self.fft_size)?;

        self.plan
            .exec_c2c(&mut d_input, &mut d_output, FftDirection::Forward)?;

        let host_output: Vec<sys::float2> = self.stream.clone_dtoh(&d_output)?;
        Ok(Self::from_cuda_complex(&host_output))
    }

    /// Perform inverse FFT.
    pub fn ifft(&self, input: &[Complex<f32>]) -> Result<Vec<Complex<f32>>, Box<dyn Error>> {
        self.ensure_input_size(input)?;

        let host_input = Self::to_cuda_complex(input);
        let mut d_input = self.stream.clone_htod(&host_input)?;
        let mut d_output = self.stream.alloc_zeros::<sys::float2>(self.fft_size)?;

        self.plan
            .exec_c2c(&mut d_input, &mut d_output, FftDirection::Inverse)?;

        let host_output: Vec<sys::float2> = self.stream.clone_dtoh(&d_output)?;
        let mut result = Self::from_cuda_complex(&host_output);
        Self::scale_inverse(&mut result, self.fft_size);
        Ok(result)
    }

    /// Perform batch FFT.
    pub fn batch_fft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn Error>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let n = Self::validate_batch_inputs(inputs)?;
        let batch_size = inputs.len();

        let flat_input: Vec<sys::float2> = inputs
            .iter()
            .flat_map(|v| v.iter().map(|c| sys::float2 { x: c.re, y: c.im }))
            .collect();

        let mut d_input = self.stream.clone_htod(&flat_input)?;
        let mut d_output = self.stream.alloc_zeros::<sys::float2>(n * batch_size)?;

        let plan = CudaFft::plan_1d(
            n as i32,
            sys::cufftType::CUFFT_C2C,
            batch_size as i32,
            self.stream.clone(),
        )?;
        plan.exec_c2c(&mut d_input, &mut d_output, FftDirection::Forward)?;

        let flat_output: Vec<sys::float2> = self.stream.clone_dtoh(&d_output)?;
        let result = flat_output.chunks(n).map(Self::from_cuda_complex).collect();

        Ok(result)
    }

    fn batch_ifft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn Error>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let n = Self::validate_batch_inputs(inputs)?;
        let batch_size = inputs.len();

        let flat_input: Vec<sys::float2> = inputs
            .iter()
            .flat_map(|v| v.iter().map(|c| sys::float2 { x: c.re, y: c.im }))
            .collect();

        let mut d_input = self.stream.clone_htod(&flat_input)?;
        let mut d_output = self.stream.alloc_zeros::<sys::float2>(n * batch_size)?;

        let plan = CudaFft::plan_1d(
            n as i32,
            sys::cufftType::CUFFT_C2C,
            batch_size as i32,
            self.stream.clone(),
        )?;
        plan.exec_c2c(&mut d_input, &mut d_output, FftDirection::Inverse)?;

        let flat_output: Vec<sys::float2> = self.stream.clone_dtoh(&d_output)?;

        let mut result: Vec<Vec<Complex<f32>>> =
            flat_output.chunks(n).map(Self::from_cuda_complex).collect();

        for values in &mut result {
            Self::scale_inverse(values, n);
        }

        Ok(result)
    }

    /// Check if CUDA is available.
    pub fn is_available() -> bool {
        Self::new(16).is_ok()
    }
}
