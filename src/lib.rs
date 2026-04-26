//! GPU-accelerated FFT using [wgpu](https://github.com/gfx-rs/wgpu) compute shaders.
//!
//! Implements the **Stockham autosort** radix-2 FFT — a two-buffer ping-pong formulation
//! where each stage reads from one buffer and writes to the other. This eliminates the separate
//! bit-reversal pass and removes all inter-stage memory hazards, allowing the entire transform
//! to run in a single GPU compute pass with one `queue.submit()` call.

use std::num::NonZeroU64;

use num_complex::Complex;

pub mod benchmark;
#[cfg(feature = "cuda")]
mod cufft_wrapper;
#[cfg(feature = "hipfft")]
pub mod hipfft_wrapper;
pub mod rivals;
#[cfg(feature = "rocm")]
mod rocfft_wrapper;
mod shaders;

/// GPU-accelerated FFT executor.
///
/// Uses wgpu compute shaders for GPU acceleration when available.
/// Falls back to CPU-based software rendering when no GPU is available.
use std::cell::RefCell;

use std::any::Any;

/// Trait for FFT implementations that can be benchmarked.
pub trait FftExecutor {
    fn name(&self) -> &str;
    fn fft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>>;
    fn ifft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>>;

    /// Get a reference to the underlying type for downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// Trait for GPU FFT implementations that support GPU-only benchmarking.
pub trait GpuFftTrait {
    /// Benchmark only the GPU compute pass and DMA operations (isolated from CPU overhead).
    /// Returns duration in seconds for the GPU operations only.
    fn benchmark_gpu_only(
        &self,
        sc: &SizeCache,
        batch_size: u32,
        n: usize,
        warmup_iters: usize,
        bench_iters: usize,
    ) -> Result<f64, Box<dyn std::error::Error>>;

    /// Get or build size-specific GPU resources.
    fn get_or_build_size_cache(&self, n: usize, log_n: u32) -> SizeCache;

    /// Prepare input data for GPU processing, applying conjugation for IFFT if needed.
    fn prepare_input_data(&self, input: &[Complex<f32>], inverse: bool) -> Vec<f32>;

    /// Get the queue for GPU operations.
    fn queue(&self) -> &wgpu::Queue;
}

pub struct GpuFft {
    device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    cache: RefCell<std::collections::HashMap<usize, SizeCache>>,
}

impl FftExecutor for GpuFft {
    fn name(&self) -> &str {
        "Baseline (Stockham Radix-2)"
    }

    fn fft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.transform_batch_internal(inputs, false)
    }

    fn ifft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.transform_batch_internal(inputs, true)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl GpuFftTrait for GpuFft {
    fn benchmark_gpu_only(
        &self,
        sc: &SizeCache,
        batch_size: u32,
        n: usize,
        warmup_iters: usize,
        bench_iters: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        use std::time::Instant;

        // Warmup
        for _ in 0..warmup_iters {
            self.execute_compute_pass(sc, batch_size, n);
            // Ensure completion before next iteration
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })?;
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..bench_iters {
            self.execute_compute_pass(sc, batch_size, n);
        }

        // Wait for all submissions to complete
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;

        let duration = start.elapsed();
        Ok(duration.as_secs_f64() / bench_iters as f64)
    }

    fn get_or_build_size_cache(&self, n: usize, log_n: u32) -> SizeCache {
        self.get_or_build_size_cache(n, log_n)
    }

    fn prepare_input_data(&self, input: &[Complex<f32>], inverse: bool) -> Vec<f32> {
        self.prepare_input_data(input, inverse)
    }

    fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_validate_input_size() {
        // This test would require a real GPU instance, so we'll test the logic indirectly
        // through the public API in integration tests
    }

    #[test]
    fn test_prepare_input_data_fft() {
        // Test FFT data preparation (no conjugation)
        let fft = GpuFft::new().expect("Failed to create FFT instance");

        let input = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];

        let result = fft.prepare_input_data(&input, false);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_prepare_input_data_ifft() {
        // Test IFFT data preparation (with conjugation)
        let fft = GpuFft::new().expect("Failed to create FFT instance");

        let input = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];

        let result = fft.prepare_input_data(&input, true);
        assert_eq!(result, vec![1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn test_apply_inverse_transform_postprocessing() {
        let fft = GpuFft::new().expect("Failed to create FFT instance");

        let mut output = vec![Complex::new(2.0, 4.0), Complex::new(6.0, 8.0)];

        fft.apply_inverse_transform_postprocessing(&mut output, 2);

        // Should conjugate and scale by 1/2
        assert_eq!(output[0].re, 1.0); // 2.0 * 0.5
        assert_eq!(output[0].im, -2.0); // -4.0 * 0.5
        assert_eq!(output[1].re, 3.0); // 6.0 * 0.5
        assert_eq!(output[1].im, -4.0); // -8.0 * 0.5
    }

    #[test]
    fn test_roundtrip_consistency() {
        let fft = GpuFft::new().expect("Failed to create FFT instance");

        // Test that FFT(IFFT(x)) ≈ x within numerical precision
        let input: Vec<Complex<f32>> = (0..1024)
            .map(|i| Complex::new(i as f32 * 0.1, 0.0))
            .collect();

        // Convert to batch format (single element batch)
        let batch_input = vec![input];
        let spectrum = fft.fft(&batch_input).expect("FFT failed");
        let reconstructed_batch = fft.ifft(&spectrum).expect("IFFT failed");
        let reconstructed = &reconstructed_batch[0];

        // Check that roundtrip error is small (allow for numerical precision)
        for (original, recon) in batch_input[0].iter().zip(reconstructed.iter()) {
            let error =
                ((original.re - recon.re).powi(2) + (original.im - recon.im).powi(2)).sqrt();
            assert!(error < 1e-4, "Roundtrip error too large: {}", error);
        }
    }
}

/// Pre-allocated GPU resources for a specific FFT size.
#[derive(Clone)]
pub struct SizeCache {
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    #[allow(dead_code)] // Used in shader but not directly accessed in Rust code
    twiddle_buf: wgpu::Buffer,
    #[allow(dead_code)] // Kept for potential future use
    data_bytes: u64,
    stage_bgs: Vec<wgpu::BindGroup>,
    result_in_b: bool,
    wg_n2: u32,
}

/// Uniforms passed to the compute shader (16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FftUniforms {
    n: u32,
    stage: u32,
    log_n: u32,
    _pad: u32,
}

impl GpuFft {
    /// Create a new [`GpuFft`].
    ///
    /// Attempts to acquire a GPU adapter via wgpu, with CPU fallback enabled.
    /// Uses GPU acceleration when available, otherwise falls back to software rendering.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use wgls_rs_fft::GpuFft;
    ///
    /// let fft = GpuFft::new().expect("GPU required");
    /// // Now use fft.fft() and fft.ifft()
    /// ```
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Self::with_shader(
            shaders::stockham::WGSL_MODULE.wgsl_source().join("\n"),
            "stockham_baseline",
        )
    }

    /// Create a new [`GpuFft`] with a custom WGSL shader.
    /// This allows AI rivals to swap kernels easily.
    pub fn with_shader(
        wgsl_source: String,
        label: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: true,
            }))
        })?;

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                ..Default::default()
            }))?;

        let shader_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", label)),
            layout: None,
            module: &shader_mod,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            cache: RefCell::new(std::collections::HashMap::new()),
        })
    }

    /// Check if a GPU is available without creating an instance.
    pub fn is_gpu_available() -> bool {
        let instance = wgpu::Instance::default();
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .is_ok()
    }

    /// Compute the forward FFT for a batch of input vectors.
    ///
    /// Processes multiple FFTs efficiently. For single vector processing,
    /// pass a vector containing one input vector.
    /// All input vectors must have the same length, which must be a power of two.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A vector of input vectors, each containing complex samples.
    ///
    /// # Returns
    ///
    /// A vector of FFT results, one for each input vector.
    ///
    /// # Panics
    ///
    /// Panics if any input vector is empty, has a different length than others,
    /// or if the length is not a power of two.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU operation fails (buffer mapping, device lost, etc.).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use wgls_rs_fft::GpuFft;
    /// use num_complex::Complex;
    ///
    /// let fft = GpuFft::new().expect("GPU or CPU fallback required");
    ///
    /// // Single FFT (pass vector with one element)
    /// let single_input = vec![vec![Complex::new(1.0, 0.0); 1024]];
    /// let single_spectrum = fft.fft(&single_input).expect("FFT failed");
    ///
    /// // Batch FFT
    /// let batch_inputs = vec![
    ///     vec![Complex::new(1.0, 0.0); 1024],
    ///     vec![Complex::new(0.5, 0.0); 1024],
    /// ];
    /// let batch_spectra = fft.fft(&batch_inputs).expect("Batch FFT failed");
    /// ```
    pub fn fft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.transform_batch_internal(inputs, false)
    }

    /// Compute the inverse FFT for a batch of input vectors.
    ///
    /// Processes multiple IFFTs efficiently. For single vector processing,
    /// pass a vector containing one input vector.
    /// All input vectors must have the same length, which must be a power of two.
    /// The output is automatically scaled by `1/N` to maintain the unitary transform property.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A vector of input vectors, each containing complex samples.
    ///
    /// # Returns
    ///
    /// A vector of IFFT results, one for each input vector.
    ///
    /// # Panics
    ///
    /// Panics if any input vector is empty, has a different length than others,
    /// or if the length is not a power of two.
    ///
    /// # Errors
    ///
    /// Returns an error if a GPU operation fails (buffer mapping, device lost, etc.).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use wgls_rs_fft::GpuFft;
    /// use num_complex::Complex;
    ///
    /// let fft = GpuFft::new().expect("GPU or CPU fallback required");
    ///
    /// // Single IFFT (pass vector with one element)
    /// let single_spectrum = vec![vec![Complex::new(1.0, 0.0); 1024]];
    /// let single_reconstructed = fft.ifft(&single_spectrum).expect("IFFT failed");
    ///
    /// // Batch IFFT
    /// let batch_spectra = vec![
    ///     vec![Complex::new(1.0, 0.0); 1024],
    ///     vec![Complex::new(0.5, 0.0); 1024],
    /// ];
    /// let batch_reconstructed = fft.ifft(&batch_spectra).expect("Batch IFFT failed");
    /// ```
    pub fn ifft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.transform_batch_internal(inputs, true)
    }

    /// Internal transform implementation that handles both FFT and IFFT.
    ///
    /// When `inverse` is true, computes IFFT (with conjugation and 1/N scaling).
    /// When `inverse` is false, computes standard FFT.
    fn transform_internal(
        &self,
        input: &[Complex<f32>],
        inverse: bool,
    ) -> Result<Vec<Complex<f32>>, Box<dyn std::error::Error>> {
        // Use batch processing even for single FFT for consistency
        let batch_input = vec![input.to_vec()];
        let batch_result = self.transform_batch_internal(&batch_input, inverse)?;
        Ok(batch_result.into_iter().next().unwrap())
    }

    /// Validate that the input size is a power of two and non-zero.
    fn validate_input_size(&self, n: usize) -> Result<(), Box<dyn std::error::Error>> {
        assert!(
            n.is_power_of_two() && n > 0,
            "Transform length must be a non-zero power of two"
        );
        Ok(())
    }

    /// Internal batch transform implementation that handles both FFT and IFFT for multiple inputs.
    ///
    /// When `inverse` is true, computes IFFT (with conjugation and 1/N scaling).
    /// When `inverse` is false, computes standard FFT.
    fn transform_batch_internal(
        &self,
        inputs: &[Vec<Complex<f32>>],
        inverse: bool,
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Validate all inputs have the same size
        let n = inputs[0].len();
        for input in inputs.iter() {
            assert_eq!(
                input.len(),
                n,
                "All input vectors in a batch must have the same length"
            );
            self.validate_input_size(input.len())?;
        }

        let log_n = n.trailing_zeros();
        let batch_size = inputs.len() as u32;
        let sc = self.get_or_build_size_cache(n, log_n);

        // Prepare all input data for parallel processing
        let mut all_raw_data = Vec::with_capacity((n * 2 * batch_size as usize) as usize);
        for input in inputs {
            let raw = self.prepare_input_data(input, inverse);
            all_raw_data.extend_from_slice(&raw);
        }

        // Upload entire batch to GPU
        self.queue
            .write_buffer(&sc.buf_a, 0, bytemuck::cast_slice(&all_raw_data));

        // Execute compute pass for the entire batch
        self.execute_compute_pass(&sc, batch_size, n);

        // Read back all results
        let mut output = self.readback_results(&sc, batch_size, n)?;

        // Apply post-processing for inverse transforms
        if inverse {
            for chunk in output.chunks_mut(n) {
                self.apply_inverse_transform_postprocessing(chunk, n);
            }
        }

        // Split into individual results
        let results: Vec<Vec<Complex<f32>>> =
            output.chunks(n).map(|chunk| chunk.to_vec()).collect();

        Ok(results)
    }

    /// Prepare input data for GPU processing, applying conjugation for IFFT if needed.
    pub fn prepare_input_data(&self, input: &[Complex<f32>], inverse: bool) -> Vec<f32> {
        if inverse {
            // For IFFT: conjugate input
            input.iter().flat_map(|c| [c.re, -c.im]).collect()
        } else {
            // For FFT: use input as-is
            input.iter().flat_map(|c| [c.re, c.im]).collect()
        }
    }

    /// Execute the compute shader pass with performance optimizations.
    fn execute_compute_pass(&self, sc: &SizeCache, batch_size: u32, n: usize) {
        // Use a single command encoder for the entire operation
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Optimized FFT Pass"),
            });

        // Optimized compute pass with better resource utilization
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFT Compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);

            // Process all stages efficiently
            for bg in &sc.stage_bgs {
                pass.set_bind_group(0, bg, &[]);
                // Optimized workgroup dispatch with batch dimension
                pass.dispatch_workgroups(sc.wg_n2, batch_size, 1);
            }
        }

        // Efficient result transfer
        let result_buf = if sc.result_in_b { &sc.buf_b } else { &sc.buf_a };
        let single_fft_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64;
        enc.copy_buffer_to_buffer(
            result_buf,
            0,
            &sc.staging_buf,
            0,
            single_fft_bytes * batch_size as u64,
        );

        // Submit with minimal overhead
        self.queue.submit(std::iter::once(enc.finish()));
    }

    /// Read back results from GPU and convert to complex numbers.
    fn readback_results(
        &self,
        sc: &SizeCache,
        batch_size: u32,
        n: usize,
    ) -> Result<Vec<Complex<f32>>, Box<dyn std::error::Error>> {
        // Readback
        let single_fft_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64;
        let total_bytes = single_fft_bytes * batch_size as u64;
        let slice = sc.staging_buf.slice(0..total_bytes);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;

        let mapped = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&mapped);
        let output: Vec<Complex<f32>> = floats
            .chunks_exact(2)
            .map(|p| Complex { re: p[0], im: p[1] })
            .collect();

        drop(mapped);
        sc.staging_buf.unmap();

        Ok(output)
    }

    /// Apply postprocessing for inverse transform (conjugation and 1/N scaling).
    fn apply_inverse_transform_postprocessing(&self, output: &mut [Complex<f32>], n: usize) {
        let scale = 1.0 / n as f32;
        for c in output {
            *c = Complex {
                re: c.re * scale,
                im: -c.im * scale,
            };
        }
    }

    /// Benchmark only the GPU compute pass and DMA operations (isolated from CPU overhead).
    /// Returns duration in seconds for the GPU operations only.
    fn benchmark_gpu_only(
        &self,
        sc: &SizeCache,
        batch_size: u32,
        n: usize,
        warmup_iters: usize,
        bench_iters: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        use std::time::Instant;

        // Warmup
        for _ in 0..warmup_iters {
            self.execute_compute_pass(sc, batch_size, n);
            // Ensure completion before next iteration
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })?;
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..bench_iters {
            self.execute_compute_pass(sc, batch_size, n);
        }

        // Wait for all submissions to complete
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;

        let duration = start.elapsed();
        Ok(duration.as_secs_f64() / bench_iters as f64)
    }

    /// Get or build size-specific GPU resources.
    pub fn get_or_build_size_cache(&self, n: usize, log_n: u32) -> SizeCache {
        let mut cache = self.cache.borrow_mut();
        if let Some(sc) = cache.get(&n) {
            return sc.clone();
        }

        let sc = self.build_size_cache(n, log_n);
        cache.insert(n, sc.clone());
        sc
    }

    /// Build GPU buffers and bind groups for a specific FFT size.
    fn build_size_cache(&self, n: usize, log_n: u32) -> SizeCache {
        let single_fft_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64;
        let max_batch_size = (self.device.limits().max_storage_buffer_binding_size as u64
            / single_fft_bytes)
            .min(1024) as u32;
        let data_bytes = single_fft_bytes * max_batch_size as u64;

        let make_buf = |label| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: data_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        let buf_a = make_buf("fft_buf_a");
        let buf_b = make_buf("fft_buf_b");
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_staging"),
            size: data_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform buffer: one slot per stage, stride-aligned.
        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let entry_bytes = std::mem::size_of::<FftUniforms>() as u64;
        let stride = entry_bytes.div_ceil(alignment) * alignment;

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_uniforms"),
            size: stride * log_n as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write uniforms for each stage
        for stage in 0..log_n {
            self.queue.write_buffer(
                &uniform_buf,
                stride * stage as u64,
                bytemuck::bytes_of(&FftUniforms {
                    n: n as u32,
                    stage,
                    log_n,
                    _pad: 0,
                }),
            );
        }

        // Precompute twiddle factors: N/2 complex pairs e^{-2πij/N}.
        let twiddles: Vec<f32> = (0..n / 2)
            .flat_map(|j| {
                let angle = -std::f32::consts::TAU * j as f32 / n as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();

        let twiddle_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_twiddles"),
            size: (twiddles.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&twiddle_buf, 0, bytemuck::cast_slice(&twiddles));

        // Pre-bake bind groups: even stages read A / write B, odd stages read B / write A.
        let uniform_size = NonZeroU64::new(entry_bytes);
        let layout = self.pipeline.get_bind_group_layout(0);

        let make_bg = |src: &wgpu::Buffer, dst: &wgpu::Buffer, uniform_offset: u64| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &uniform_buf,
                            offset: uniform_offset,
                            size: uniform_size,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: twiddle_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let stage_bgs = (0..log_n as usize)
            .map(|s| {
                let (src, dst) = if s % 2 == 0 {
                    (&buf_a, &buf_b)
                } else {
                    (&buf_b, &buf_a)
                };
                make_bg(src, dst, stride * s as u64)
            })
            .collect();

        SizeCache {
            buf_a,
            buf_b,
            staging_buf,
            twiddle_buf,
            data_bytes,
            stage_bgs,
            result_in_b: log_n % 2 == 1,
            wg_n2: (n as u32 / 2).div_ceil(256),
        }
    }
}

impl Default for GpuFft {
    fn default() -> Self {
        Self::new().expect("No GPU available for default GpuFft instance")
    }
}

#[cfg(feature = "cuda")]
pub use cufft_wrapper::CuFft;
#[cfg(feature = "hipfft")]
pub use hipfft_wrapper::HipFft;
#[cfg(feature = "rocm")]
pub use rocfft_wrapper::RocFft;
