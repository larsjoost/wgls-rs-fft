//! GPU-accelerated FFT using [wgpu](https://docs.rs/wgpu) compute shaders,
//! with automatic CPU fallback via the [wgsl-rs](https://github.com/schell/wgsl-rs)
//! software rasterizer when no GPU is available.
//!
//! Implements the **Stockham autosort** radix-2 FFT: a two-buffer ping-pong
//! formulation where each stage reads from one buffer and writes to the other.
//! This eliminates the separate bit-reversal pass and removes all inter-stage
//! memory hazards, allowing the entire transform to run in a single GPU
//! compute pass with one `queue.submit()` call.
//!
//! # Example
//!
//! ```rust,no_run
//! use wgls_rs_fft::GpuFft;
//! use num_complex::Complex;
//!
//! // Always succeeds: uses GPU if available, falls back to CPU otherwise.
//! let fft = GpuFft::new();
//!
//! let input: Vec<Complex<f32>> = (0..1024)
//!     .map(|i| Complex { re: (i as f32 * 0.1).sin(), im: 0.0 })
//!     .collect();
//! let spectrum = fft.fft(&input).unwrap();
//! assert_eq!(spectrum.len(), 1024);
//! ```
//!
//! # Limitations
//!
//! * Input length must be a **power of two** and non-empty.
//! * Only **forward** (analysis) FFT is provided.

mod shaders;

use std::collections::HashMap;
use std::mem::size_of;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use num_complex::Complex;
use wgsl_rs::std::*;

/// Uniforms passed to the compute shader (16-byte aligned).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FftUniforms {
    n: u32,
    stage: u32,
    log_n: u32,
    _pad: u32,
}

/// Pre-allocated GPU resources for a specific FFT size.
///
/// Built once on first use for each size and reused across all subsequent calls.
struct SizeCache {
    /// Ping-pong buffers: stage even reads A, writes B; stage odd reads B, writes A.
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    /// Precomputed twiddle factors: N/2 complex pairs (cos, sin) of e^{-2πij/N}.
    #[allow(dead_code)]
    twiddle_buf: wgpu::Buffer,
    data_bytes: u64,
    /// One bind group per stage, pre-wired to the correct SRC/DST buffer pair.
    stage_bgs: Vec<wgpu::BindGroup>,
    /// True when the final result lands in buf_b (i.e. log_n is odd).
    result_in_b: bool,
    wg_n2: u32,
}

struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    cache: Mutex<HashMap<usize, Arc<SizeCache>>>,
}

enum Backend {
    Gpu(GpuBackend),
    /// CPU fallback: the wgsl-rs module statics are global, so concurrent
    /// CPU FFT calls must be serialized.
    Cpu(Mutex<()>),
}

/// FFT executor that uses a GPU if one is available, or falls back to CPU.
///
/// Create with [`GpuFft::new`] — it never fails.
/// Check which backend is active with [`GpuFft::is_gpu_backed`].
pub struct GpuFft {
    backend: Backend,
}

impl GpuFft {
    /// Create a new [`GpuFft`].
    ///
    /// Attempts to acquire a high-performance GPU adapter via wgpu.
    /// If no compatible GPU is found, transparently falls back to the
    /// wgsl-rs CPU runtime running the same shader code.
    pub fn new() -> Self {
        // Initialize CPU runtime early in case we need to fall back
        ensure_cpu_runtime_initialized();

        match pollster::block_on(GpuBackend::try_new()) {
            Ok(gpu) => Self {
                backend: Backend::Gpu(gpu),
            },
            Err(_) => Self {
                backend: Backend::Cpu(Mutex::new(())),
            },
        }
    }

    /// Returns `true` if this instance is backed by a real GPU.
    pub fn is_gpu_backed(&self) -> bool {
        matches!(self.backend, Backend::Gpu(_))
    }

    /// Compute the forward FFT of `input`.
    ///
    /// Returns a `Vec` of `N` complex frequency-domain bins where `N =
    /// input.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `input` is empty or its length is not a power of two.
    ///
    /// # Errors
    ///
    /// On the GPU path, returns an error if a GPU operation fails (buffer
    /// mapping, device lost, etc.). The CPU path never errors.
    pub fn fft(
        &self,
        input: &[Complex<f32>],
    ) -> Result<Vec<Complex<f32>>, Box<dyn std::error::Error>> {
        let n = input.len();
        assert!(
            n.is_power_of_two() && n > 0,
            "FFT length must be a non-zero power of two"
        );

        match &self.backend {
            Backend::Gpu(gpu) => gpu.fft(input),
            Backend::Cpu(lock) => {
                let _guard = lock.lock().unwrap();
                Ok(cpu_fft(input))
            }
        }
    }

    /// Compute the inverse FFT of `input`.
    ///
    /// Returns a `Vec` of `N` complex time-domain samples where `N =
    /// input.len()`. The output is automatically scaled by `1/N` to
    /// maintain the unitary transform property.
    ///
    /// # Panics
    ///
    /// Panics if `input` is empty or its length is not a power of two.
    ///
    /// # Errors
    ///
    /// On the GPU path, returns an error if a GPU operation fails (buffer
    /// mapping, device lost, etc.). The CPU path never errors.
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
        match &self.backend {
            Backend::Gpu(gpu) => {
                // Conjugate input
                let conjugated: Vec<Complex<f32>> = input
                    .iter()
                    .map(|c| Complex {
                        re: c.re,
                        im: -c.im,
                    })
                    .collect();

                // Compute FFT of conjugated input
                let mut result = gpu.fft(&conjugated)?;

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
            Backend::Cpu(lock) => {
                let _guard = lock.lock().unwrap();
                Ok(cpu_ifft(input))
            }
        }
    }
}

impl Default for GpuFft {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GPU backend
// ---------------------------------------------------------------------------

impl GpuBackend {
    async fn try_new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                ..Default::default()
            })
            .await?;

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
            cache: Mutex::new(HashMap::new()),
        })
    }

    fn get_or_build(&self, n: usize, log_n: u32) -> Arc<SizeCache> {
        {
            let guard = self.cache.lock().unwrap();
            if let Some(sc) = guard.get(&n) {
                return Arc::clone(sc);
            }
        }
        let sc = Arc::new(self.build_size_cache(n, log_n));
        self.cache
            .lock()
            .unwrap()
            .entry(n)
            .or_insert_with(|| Arc::clone(&sc));
        sc
    }

    /// Allocate the ping-pong buffers and pre-bake all per-stage bind groups.
    ///
    /// The uniform buffer holds one `FftUniforms` per stage at aligned offsets,
    /// so no `write_buffer` calls are needed during the hot `fft()` path.
    fn build_size_cache(&self, n: usize, log_n: u32) -> SizeCache {
        let data_bytes = (n * 2 * size_of::<f32>()) as u64;

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
        let entry_bytes = size_of::<FftUniforms>() as u64;
        let stride = entry_bytes.div_ceil(alignment) * alignment;

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_uniforms"),
            size: stride * log_n as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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

        // Pre-bake bind groups: even stages read A / write B, odd stages read B / write A.
        let uniform_size = NonZeroU64::new(entry_bytes);
        let layout = self.pipeline.get_bind_group_layout(0);

        // Precompute twiddle factors: N/2 complex pairs e^{-2πij/N} for j=0..N/2.
        let twiddles: Vec<f32> = (0..n / 2)
            .flat_map(|j| {
                let angle = -std::f32::consts::TAU * j as f32 / n as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let twiddle_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_twiddles"),
            size: (twiddles.len() * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&twiddle_buf, 0, bytemuck::cast_slice(&twiddles));

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

    fn fft(&self, input: &[Complex<f32>]) -> Result<Vec<Complex<f32>>, Box<dyn std::error::Error>> {
        let n = input.len();
        let log_n = n.trailing_zeros();
        let sc = self.get_or_build(n, log_n);

        // Upload input to buf_a (always the starting buffer).
        let raw: Vec<f32> = input.iter().flat_map(|c| [c.re, c.im]).collect();
        self.queue
            .write_buffer(&sc.buf_a, 0, bytemuck::cast_slice(&raw));

        // All stages in one compute pass, one submit.
        // No barriers needed: consecutive stages access different buffers (A vs B).
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

        // Readback with optimized synchronization
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
}

// ---------------------------------------------------------------------------
// CPU fallback — runs the same Stockham kernel via the wgsl-rs CPU runtime.
// ---------------------------------------------------------------------------

/// Initialize the wgsl-rs CPU runtime if not already initialized
fn ensure_cpu_runtime_initialized() {
    // This ensures the wgsl-rs module statics are properly initialized
    // by accessing them at least once before use
    use wgsl_rs::std::*;

    // Trigger initialization by accessing module variables
    let _ = RuntimeArray::<f32>::with_capacity(0);
    let _ = vec4u(0, 0, 0, 0);
}

fn cpu_fft(input: &[Complex<f32>]) -> Vec<Complex<f32>> {
    // Ensure runtime is initialized
    ensure_cpu_runtime_initialized();
    let n = input.len();
    let log_n = n.trailing_zeros();

    // Pack input into RuntimeArray (the wgsl-rs CPU buffer type).
    let mut a = RuntimeArray::with_capacity(n * 2);
    for c in input {
        a.push(c.re);
        a.push(c.im);
    }
    let mut b = RuntimeArray::with_capacity(n * 2);
    for _ in 0..n * 2 {
        b.push(0.0_f32);
    }

    // Each stage: set SRC=a, DST=b, dispatch, then swap:
    // a ← DST output, b ← old SRC (will be overwritten next stage).
    for stage in 0..log_n {
        shaders::stockham::U.set(vec4u(n as u32, stage, log_n, 0));
        shaders::stockham::SRC.set(a);
        shaders::stockham::DST.set(b);
        dispatch_workgroups(((n as u32 / 2).div_ceil(64), 1, 1), (64, 1, 1), |inv| {
            shaders::stockham::main(inv.global_invocation_id)
        });
        a = shaders::stockham::DST.get().clone();
        b = shaders::stockham::SRC.get().clone();
    }

    a.data
        .chunks_exact(2)
        .map(|p| Complex { re: p[0], im: p[1] })
        .collect()
}

fn cpu_ifft(input: &[Complex<f32>]) -> Vec<Complex<f32>> {
    // Ensure runtime is initialized
    ensure_cpu_runtime_initialized();

    let n = input.len();
    let log_n = n.trailing_zeros();

    // Conjugate input for IFFT
    let mut a = RuntimeArray::with_capacity(n * 2);
    for c in input {
        a.push(c.re); // real part
        a.push(-c.im); // negated imaginary part
    }
    let mut b = RuntimeArray::with_capacity(n * 2);
    for _ in 0..n * 2 {
        b.push(0.0_f32);
    }

    // Each stage: set SRC=a, DST=b, dispatch, then swap:
    for stage in 0..log_n {
        shaders::stockham::U.set(vec4u(n as u32, stage, log_n, 0));
        shaders::stockham::SRC.set(a);
        shaders::stockham::DST.set(b);
        dispatch_workgroups(((n as u32 / 2).div_ceil(64), 1, 1), (64, 1, 1), |inv| {
            shaders::stockham::main(inv.global_invocation_id)
        });
        a = shaders::stockham::DST.get().clone();
        b = shaders::stockham::SRC.get().clone();
    }

    // Conjugate output and scale by 1/N
    let scale = 1.0 / n as f32;
    a.data
        .chunks_exact(2)
        .map(|p| Complex {
            re: p[0] * scale,
            im: -p[1] * scale,
        })
        .collect()
}
