use crate::FftExecutor;
use cufft_rust::*;
use num_complex::Complex;
use std::error::Error;

/// Wrapper for cuFFT to compare performance with our WebGPU implementation
pub struct CuFft {
    plan: CufftPlan,
    stream: CudaStream,
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
        // IFFT needs to be handled via batch if we want to support it properly in the trait
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let n = inputs[0].len();
        let batch_size = inputs.len();

        // Flatten input data
        let flat_input: Vec<Complex<f32>> = inputs.iter().flat_map(|v| v.iter().cloned()).collect();

        // Create device buffers
        let mut d_input = CudaDeviceMemory::new(&flat_input)?;
        let mut d_output = CudaDeviceMemory::new_empty::<Complex<f32>>(n * batch_size)?;

        // Create batch plan
        let mut batch_plan = CufftPlan::new()?;
        batch_plan.set_stream(self.stream.handle())?;
        batch_plan.create_1d(n as i32, CufftType::C2C, batch_size as i32)?;

        // Execute batch FFT
        batch_plan.execute_c2c(&mut d_input, &mut d_output, CufftDirection::Inverse)?;

        // Copy result back to host
        let mut flat_result = vec![Complex::new(0.0, 0.0); n * batch_size];
        d_output.copy_to_host(&mut flat_result)?;

        // Split into individual results and scale
        let scale = 1.0 / n as f32;
        let result: Vec<Vec<Complex<f32>>> = flat_result
            .chunks(n)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|c| Complex::new(c.re * scale, c.im * scale))
                    .collect()
            })
            .collect();

        Ok(result)
    }
}

impl CuFft {
    /// Create a new cuFFT instance
    pub fn new(fft_size: usize) -> Result<Self, Box<dyn Error>> {
        // Initialize CUDA
        cufft_rust::init()?;

        // Create a stream
        let stream = CudaStream::new()?;

        // Create FFT plan
        let mut plan = CufftPlan::new()?;
        plan.set_stream(stream.handle())?;
        plan.create_1d(fft_size as i32, CufftType::C2C, 1)?;

        Ok(Self {
            plan,
            stream,
            fft_size,
        })
    }

    /// Perform forward FFT
    pub fn fft(&self, input: &[Complex<f32>]) -> Result<Vec<Complex<f32>>, Box<dyn Error>> {
        let n = input.len();

        // Create device buffers
        let mut d_input = CudaDeviceMemory::new(input)?;
        let mut d_output = CudaDeviceMemory::new_empty::<Complex<f32>>(n)?;

        // Execute FFT
        self.plan
            .execute_c2c(&mut d_input, &mut d_output, CufftDirection::Forward)?;

        // Copy result back to host
        let mut result = vec![Complex::new(0.0, 0.0); n];
        d_output.copy_to_host(&mut result)?;

        Ok(result)
    }

    /// Perform inverse FFT
    pub fn ifft(&self, input: &[Complex<f32>]) -> Result<Vec<Complex<f32>>, Box<dyn Error>> {
        let n = input.len();

        // Create device buffers
        let mut d_input = CudaDeviceMemory::new(input)?;
        let mut d_output = CudaDeviceMemory::new_empty::<Complex<f32>>(n)?;

        // Execute inverse FFT
        self.plan
            .execute_c2c(&mut d_input, &mut d_output, CufftDirection::Inverse)?;

        // Copy result back to host and scale
        let mut result = vec![Complex::new(0.0, 0.0); n];
        d_output.copy_to_host(&mut result)?;

        // Scale by 1/N for inverse FFT
        let scale = 1.0 / n as f32;
        for c in &mut result {
            *c = Complex::new(c.re * scale, c.im * scale);
        }

        Ok(result)
    }

    /// Perform batch FFT
    pub fn batch_fft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn Error>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let n = inputs[0].len();
        let batch_size = inputs.len();

        // Flatten input data
        let flat_input: Vec<Complex<f32>> = inputs.iter().flat_map(|v| v.iter().cloned()).collect();

        // Create device buffers
        let mut d_input = CudaDeviceMemory::new(&flat_input)?;
        let mut d_output = CudaDeviceMemory::new_empty::<Complex<f32>>(n * batch_size)?;

        // Create batch plan
        let mut batch_plan = CufftPlan::new()?;
        batch_plan.set_stream(self.stream.handle())?;
        batch_plan.create_1d(n as i32, CufftType::C2C, batch_size as i32)?;

        // Execute batch FFT
        batch_plan.execute_c2c(&mut d_input, &mut d_output, CufftDirection::Forward)?;

        // Copy result back to host
        let mut flat_result = vec![Complex::new(0.0, 0.0); n * batch_size];
        d_output.copy_to_host(&mut flat_result)?;

        // Split into individual results
        let result: Vec<Vec<Complex<f32>>> =
            flat_result.chunks(n).map(|chunk| chunk.to_vec()).collect();

        Ok(result)
    }

    /// Check if CUDA is available
    pub fn is_available() -> bool {
        cufft_rust::init().is_ok()
    }
}

impl Drop for CuFft {
    fn drop(&mut self) {
        // Cleanup is handled automatically by cufft_rust
    }
}
