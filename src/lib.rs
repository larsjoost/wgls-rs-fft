//! GPU-accelerated FFT using [wgpu](https://github.com/gfx-rs/wgpu) compute shaders.
//!
//! Implements the **Stockham autosort** radix-2 FFT — a two-buffer ping-pong formulation
//! where each stage reads from one buffer and writes to the other. This eliminates the separate
//! bit-reversal pass and removes all inter-stage memory hazards, allowing the entire transform
//! to run in a single GPU compute pass with one `queue.submit()` call.

use std::num::NonZeroU64;

use num_complex::Complex;

mod shaders;

/// GPU-accelerated FFT executor.
///
/// This is a GPU-only library. `GpuFft::new()` will return an error if no compatible
/// GPU is available (Vulkan, Metal, DX12, or WebGPU).
use std::cell::RefCell;

pub struct GpuFft {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    cache: RefCell<std::collections::HashMap<usize, SizeCache>>,
}

/// Pre-allocated GPU resources for a specific FFT size.
#[derive(Clone)]
struct SizeCache {
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    #[allow(dead_code)] // Used in shader but not directly accessed in Rust code
    twiddle_buf: wgpu::Buffer,
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
    /// Attempts to acquire a high-performance GPU adapter via wgpu.
    /// Returns `Err` if no compatible GPU is available.
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
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                ..Default::default()
            }))?;

        let shader_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stockham"),
            source: wgpu::ShaderSource::Wgsl(
                shaders::stockham::WGSL_MODULE
                    .wgsl_source()
                    .join("\n")
                    .into(),
            ),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stockham_pipeline"),
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

    /// Compute the forward FFT of `input`.
    ///
    /// Returns a `Vec` of `N` complex frequency-domain bins where `N = input.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `input` is empty or its length is not a power of two.
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
    /// let fft = GpuFft::new().expect("GPU required");
    /// let input = vec![Complex::new(1.0, 0.0); 1024];
    /// let spectrum = fft.fft(&input).expect("FFT failed");
    /// ```
    pub fn fft(
        &self,
        input: &[Complex<f32>],
    ) -> Result<Vec<Complex<f32>>, Box<dyn std::error::Error>> {
        let n = input.len();
        assert!(
            n.is_power_of_two() && n > 0,
            "FFT length must be a non-zero power of two"
        );

        let log_n = n.trailing_zeros();
        let sc = self.get_or_build_size_cache(n, log_n);

        // Upload input to buf_a (always the starting buffer).
        let raw: Vec<f32> = input.iter().flat_map(|c| [c.re, c.im]).collect();
        self.queue
            .write_buffer(&sc.buf_a, 0, bytemuck::cast_slice(&raw));

        // All stages in one compute pass, one submit.
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            for bg in &sc.stage_bgs {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(sc.wg_n2, 1, 1);
            }
        }
        let result_buf = if sc.result_in_b { &sc.buf_b } else { &sc.buf_a };
        enc.copy_buffer_to_buffer(result_buf, 0, &sc.staging_buf, 0, sc.data_bytes);
        self.queue.submit(std::iter::once(enc.finish()));

        // Readback
        let slice = sc.staging_buf.slice(..);
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

    /// Compute the inverse FFT of `input`.
    ///
    /// Returns a `Vec` of `N` complex time-domain samples where `N = input.len()`.
    /// The output is automatically scaled by `1/N` to maintain the unitary transform property.
    ///
    /// # Panics
    ///
    /// Panics if `input` is empty or its length is not a power of two.
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
    /// let fft = GpuFft::new().expect("GPU required");
    /// let spectrum = vec![Complex::new(1.0, 0.0); 1024];
    /// let reconstructed = fft.ifft(&spectrum).expect("IFFT failed");
    /// // reconstructed ≈ original signal (within numerical precision)
    /// ```
    pub fn ifft(
        &self,
        input: &[Complex<f32>],
    ) -> Result<Vec<Complex<f32>>, Box<dyn std::error::Error>> {
        let n = input.len();
        assert!(
            n.is_power_of_two() && n > 0,
            "IFFT length must be a non-zero power of two"
        );

        // IFFT = FFT with conjugated input and conjugated output, then scale by 1/N
        // Conjugate input
        let conjugated: Vec<Complex<f32>> = input
            .iter()
            .map(|c| Complex {
                re: c.re,
                im: -c.im,
            })
            .collect();

        // Compute FFT of conjugated input
        let mut result = self.fft(&conjugated)?;

        // Conjugate output and scale by 1/N
        let scale = 1.0 / n as f32;
        for c in &mut result {
            *c = Complex {
                re: c.re * scale,
                im: -c.im * scale,
            };
        }

        Ok(result)
    }

    /// Get or build size-specific GPU resources.
    fn get_or_build_size_cache(&self, n: usize, log_n: u32) -> SizeCache {
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
        let data_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64;

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
