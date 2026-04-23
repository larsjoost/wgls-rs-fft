//! GPU-accelerated FFT using [wgpu](https://docs.rs/wgpu) compute shaders,
//! with automatic CPU fallback via the [wgsl-rs](https://github.com/schell/wgsl-rs)
//! software rasterizer when no GPU is available.
//!
//! Implements the Cooley-Tukey radix-2 decimation-in-time (DIT) FFT entirely
//! on the GPU via two WGSL compute kernels, or on the CPU using the same
//! shader code run through the wgsl-rs CPU runtime.
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

/// Uniforms passed to each compute shader (16-byte aligned).
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
/// Built once per size on first use, then reused across calls.
struct SizeCache {
    storage_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    data_bytes: u64,
    bit_rev_bg: wgpu::BindGroup,
    /// One bind group per FFT stage, each pinned to its own uniform slice.
    stage_bgs: Vec<wgpu::BindGroup>,
    wg_n: u32,
    wg_n2: u32,
}

/// State needed when a real GPU is available.
struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bit_reverse_pipeline: wgpu::ComputePipeline,
    fft_stage_pipeline: wgpu::ComputePipeline,
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

        let bit_rev_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bit_reverse"),
            source: wgpu::ShaderSource::Wgsl(
                shaders::bit_reverse::WGSL_MODULE.wgsl_source().join("\n").into(),
            ),
        });
        let fft_stage_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fft_stage"),
            source: wgpu::ShaderSource::Wgsl(
                shaders::fft_stage::WGSL_MODULE.wgsl_source().join("\n").into(),
            ),
        });

        let bit_reverse_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bit_reverse_pipeline"),
                layout: None,
                module: &bit_rev_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let fft_stage_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft_stage_pipeline"),
            layout: None,
            module: &fft_stage_mod,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            bit_reverse_pipeline,
            fft_stage_pipeline,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Return the cached `SizeCache` for `n`, building it if this is the first call.
    fn get_or_build(&self, n: usize, log_n: u32) -> Arc<SizeCache> {
        {
            let guard = self.cache.lock().unwrap();
            if let Some(sc) = guard.get(&n) {
                return Arc::clone(sc);
            }
        }
        // Build outside the lock so device ops don't block other sizes.
        let sc = Arc::new(self.build_size_cache(n, log_n));
        self.cache
            .lock()
            .unwrap()
            .entry(n)
            .or_insert_with(|| Arc::clone(&sc));
        sc
    }

    /// Allocate GPU buffers and pre-bake bind groups for a given FFT size.
    ///
    /// The uniform buffer holds all `1 + log_n` parameter sets (bit-reversal +
    /// one per butterfly stage) at properly aligned offsets, so every dispatch
    /// can be encoded without any interleaved `write_buffer` calls.
    fn build_size_cache(&self, n: usize, log_n: u32) -> SizeCache {
        let data_bytes = (n * 2 * size_of::<f32>()) as u64;

        let storage_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_storage"),
            size: data_bytes,
            // COPY_DST: lets write_buffer upload input; COPY_SRC: lets us copy to staging.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_staging"),
            size: data_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform buffer: one slot per (bit-reversal + log_n stages), stride-aligned.
        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let entry_bytes = size_of::<FftUniforms>() as u64;
        let stride = entry_bytes.div_ceil(alignment) * alignment;
        let uniform_total = stride * (1 + log_n as u64);

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_uniforms"),
            size: uniform_total,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write all uniform data up front — these are flushed before the first submit.
        self.queue.write_buffer(
            &uniform_buf,
            0,
            bytemuck::bytes_of(&FftUniforms {
                n: n as u32,
                stage: 0,
                log_n,
                _pad: 0,
            }),
        );
        for stage in 0..log_n {
            self.queue.write_buffer(
                &uniform_buf,
                stride * (1 + stage as u64),
                bytemuck::bytes_of(&FftUniforms {
                    n: n as u32,
                    stage,
                    log_n,
                    _pad: 0,
                }),
            );
        }

        // Build one bind group per dispatch, each pointing at its own uniform slice.
        let uniform_size = NonZeroU64::new(entry_bytes);
        let make_bg = |pipeline: &wgpu::ComputePipeline, uniform_offset: u64| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
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
                        resource: storage_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let bit_rev_bg = make_bg(&self.bit_reverse_pipeline, 0);
        let stage_bgs = (0..log_n as usize)
            .map(|s| make_bg(&self.fft_stage_pipeline, stride * (1 + s as u64)))
            .collect();

        SizeCache {
            storage_buf,
            staging_buf,
            data_bytes,
            bit_rev_bg,
            stage_bgs,
            wg_n: (n as u32).div_ceil(64),
            wg_n2: (n as u32 / 2).div_ceil(64),
        }
    }

    fn fft(&self, input: &[Complex<f32>]) -> Result<Vec<Complex<f32>>, Box<dyn std::error::Error>> {
        let n = input.len();
        let log_n = n.trailing_zeros();
        let sc = self.get_or_build(n, log_n);

        // Upload input — arrives before the first dispatch in the submit below.
        let raw: Vec<f32> = input.iter().flat_map(|c| [c.re, c.im]).collect();
        self.queue
            .write_buffer(&sc.storage_buf, 0, bytemuck::cast_slice(&raw));

        // Encode everything in one command buffer: bit-reversal, all FFT stages,
        // and the readback copy. A separate compute pass per dispatch ensures the
        // GPU sees each stage's writes before the next one begins.
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.bit_reverse_pipeline);
            pass.set_bind_group(0, &sc.bit_rev_bg, &[]);
            pass.dispatch_workgroups(sc.wg_n, 1, 1);
        }
        for bg in &sc.stage_bgs {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.fft_stage_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(sc.wg_n2, 1, 1);
        }
        enc.copy_buffer_to_buffer(&sc.storage_buf, 0, &sc.staging_buf, 0, sc.data_bytes);
        self.queue.submit(std::iter::once(enc.finish()));

        // Readback
        let slice = sc.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        rx.recv()??;

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
// CPU fallback — runs the exact same wgsl-rs shader code via the CPU runtime
// ---------------------------------------------------------------------------

fn cpu_fft(input: &[Complex<f32>]) -> Vec<Complex<f32>> {
    let n = input.len();
    let log_n = n.trailing_zeros();

    // Pack Complex<f32> → interleaved f32 → RuntimeArray
    let mut data = RuntimeArray::with_capacity(n * 2);
    for c in input {
        data.push(c.re);
        data.push(c.im);
    }

    // --- Bit-reversal pass ---
    shaders::bit_reverse::U.set(vec4u(n as u32, 0, log_n, 0));
    shaders::bit_reverse::DATA.set(data);
    dispatch_workgroups(((n as u32).div_ceil(64), 1, 1), (64, 1, 1), |b| {
        shaders::bit_reverse::main(b.global_invocation_id)
    });
    // Move data from bit_reverse module into fft_stage module
    let data = shaders::bit_reverse::DATA.get().clone();

    // --- Butterfly stages ---
    shaders::fft_stage::DATA.set(data);
    for stage in 0..log_n {
        shaders::fft_stage::U.set(vec4u(n as u32, stage, log_n, 0));
        dispatch_workgroups(((n as u32 / 2).div_ceil(64), 1, 1), (64, 1, 1), |b| {
            shaders::fft_stage::main(b.global_invocation_id)
        });
    }

    // Unpack RuntimeArray → Vec<Complex<f32>>
    let result = shaders::fft_stage::DATA.get();
    result
        .data
        .chunks_exact(2)
        .map(|p| Complex { re: p[0], im: p[1] })
        .collect()
}
