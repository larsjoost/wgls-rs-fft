use std::cell::RefCell;
use std::num::NonZeroU64;

use num_complex::Complex;
use wgsl_rs::wgsl;

use crate::FftExecutor;

// ── WGSL: Stockham Radix-4 DIT ───────────────────────────────────────────────
//
// Uniform U: .x = N, .y = radix-4 stage index s (0 .. log₄N-1)
//
// For stage s:
//   p        = 4^s  = 1u32 << (s*2)
//   quarter_n = N/4
//   stride   = quarter_n >> (s*2)   = quarter_n / p
//
// Source indices (natural-order Stockham read):
//   i0 = j*p+k,  i1 = i0+N/4,  i2 = i0+N/2,  i3 = i0+3N/4
//
// Twiddle indices into the N-entry table (max tw3 < 3N/4 < N — always in bounds):
//   tw1 = k*stride,  tw2 = 2*tw1,  tw3 = 3*tw1
//
// Correct radix-4 DIT butterfly (b = W_tw1·x1, c = W_tw2·x2, d = W_tw3·x3):
//   y0 = x0 + b + c + d
//   y1 = x0 - i·b - c + i·d  →  re: x0r+bi-cr-di,  im: x0i-br-ci+dr
//   y2 = x0 - b + c - d
//   y3 = x0 + i·b - c - i·d  →  re: x0r-bi-cr+di,  im: x0i+br-ci-dr
//
// Output indices (Stockham autosort):
//   o0 = j*4p+k,  o1 = o0+p,  o2 = o0+2p,  o3 = o0+3p
#[wgsl]
pub mod claude_r4_kernel {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U: Vec4u);
    storage!(group(0), binding(1), read_write, SRC: RuntimeArray<f32>);
    storage!(group(0), binding(2), read_write, DST: RuntimeArray<f32>);
    storage!(group(0), binding(3), TWIDDLE: RuntimeArray<f32>);

    #[compute]
    #[workgroup_size(256, 1, 1)]
    pub fn main(#[builtin(global_invocation_id)] gid: Vec3u) {
        let tid = gid.x;
        let batch_id = gid.y;
        let n = get!(U).x;
        let quarter_n = n >> 2u32;
        if tid >= quarter_n {
            return;
        }

        let stage = get!(U).y;
        let p = 1u32 << (stage + stage);
        let four_p = p << 2u32;

        let k = tid % p;
        let j = tid / p;

        let batch_offset = batch_id * n * 2u32;

        let i0 = j * p + k;
        let i1 = i0 + quarter_n;
        let i2 = i0 + quarter_n + quarter_n;
        let i3 = i2 + quarter_n;

        let s0 = batch_offset + 2u32 * i0;
        let s1 = batch_offset + 2u32 * i1;
        let s2 = batch_offset + 2u32 * i2;
        let s3 = batch_offset + 2u32 * i3;

        let x0r = get!(SRC)[s0];
        let x0i = get!(SRC)[s0 + 1u32];
        let x1r = get!(SRC)[s1];
        let x1i = get!(SRC)[s1 + 1u32];
        let x2r = get!(SRC)[s2];
        let x2i = get!(SRC)[s2 + 1u32];
        let x3r = get!(SRC)[s3];
        let x3i = get!(SRC)[s3 + 1u32];

        let stride = quarter_n >> (stage + stage);
        let tw1 = k * stride;
        let tw2 = tw1 * 2u32;
        let tw3 = tw1 * 3u32;

        let wr1 = get!(TWIDDLE)[2u32 * tw1];
        let wi1 = get!(TWIDDLE)[2u32 * tw1 + 1u32];
        let wr2 = get!(TWIDDLE)[2u32 * tw2];
        let wi2 = get!(TWIDDLE)[2u32 * tw2 + 1u32];
        let wr3 = get!(TWIDDLE)[2u32 * tw3];
        let wi3 = get!(TWIDDLE)[2u32 * tw3 + 1u32];

        let br = wr1 * x1r - wi1 * x1i;
        let bi = wr1 * x1i + wi1 * x1r;
        let cr = wr2 * x2r - wi2 * x2i;
        let ci = wr2 * x2i + wi2 * x2r;
        let dr = wr3 * x3r - wi3 * x3i;
        let di = wr3 * x3i + wi3 * x3r;

        let o0 = j * four_p + k;
        let o1 = o0 + p;
        let o2 = o0 + p + p;
        let o3 = o2 + p;

        let d0 = batch_offset + 2u32 * o0;
        let d1 = batch_offset + 2u32 * o1;
        let d2 = batch_offset + 2u32 * o2;
        let d3 = batch_offset + 2u32 * o3;

        get_mut!(DST)[d0] = x0r + br + cr + dr;
        get_mut!(DST)[d0 + 1u32] = x0i + bi + ci + di;
        get_mut!(DST)[d1] = x0r + bi - cr - di;
        get_mut!(DST)[d1 + 1u32] = x0i - br - ci + dr;
        get_mut!(DST)[d2] = x0r - br + cr - dr;
        get_mut!(DST)[d2 + 1u32] = x0i - bi + ci - di;
        get_mut!(DST)[d3] = x0r - bi - cr + di;
        get_mut!(DST)[d3 + 1u32] = x0i + br - ci - dr;
    }
}

// ── WGSL: Stockham Radix-2 (finalisation for odd log₂N) ─────────────────────
//
// Identical to the baseline stockham kernel. Used as a single trailing stage
// when log₂N is odd, with U.y = log₂N − 1, giving p = N/2.
#[wgsl]
pub mod claude_r2_kernel {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U: Vec4u);
    storage!(group(0), binding(1), read_write, SRC: RuntimeArray<f32>);
    storage!(group(0), binding(2), read_write, DST: RuntimeArray<f32>);
    storage!(group(0), binding(3), TWIDDLE: RuntimeArray<f32>);

    #[compute]
    #[workgroup_size(256, 1, 1)]
    pub fn main(#[builtin(global_invocation_id)] gid: Vec3u) {
        let tid = gid.x;
        let batch_id = gid.y;
        let n = get!(U).x;
        let half_n = n >> 1u32;
        if tid >= half_n {
            return;
        }

        let stage = get!(U).y;
        let p = 1u32 << stage;
        let two_p = p + p;

        let k = tid % p;
        let j = tid / p;

        let batch_offset = batch_id * n * 2u32;

        let i1 = j * p + k;
        let i2 = i1 + half_n;

        let src1 = batch_offset + 2u32 * i1;
        let src2 = batch_offset + 2u32 * i2;

        let re1 = get!(SRC)[src1];
        let im1 = get!(SRC)[src1 + 1u32];
        let re2 = get!(SRC)[src2];
        let im2 = get!(SRC)[src2 + 1u32];

        let twiddle_idx = k * (half_n >> stage);
        let wr = get!(TWIDDLE)[2u32 * twiddle_idx];
        let wi = get!(TWIDDLE)[2u32 * twiddle_idx + 1u32];

        let tr = wr * re2 - wi * im2;
        let ti = wr * im2 + wi * re2;

        let out1 = j * two_p + k;
        let out2 = out1 + p;

        let dst1 = batch_offset + 2u32 * out1;
        let dst2 = batch_offset + 2u32 * out2;

        get_mut!(DST)[dst1] = re1 + tr;
        get_mut!(DST)[dst1 + 1u32] = im1 + ti;
        get_mut!(DST)[dst2] = re1 - tr;
        get_mut!(DST)[dst2 + 1u32] = im1 - ti;
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: u32,
    stage: u32,
    log_n: u32,
    _pad: u32,
}

#[derive(Clone)]
struct ClaudeCache {
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    #[allow(dead_code)]
    twiddle_buf: wgpu::Buffer,
    stage_bgs_r4: Vec<wgpu::BindGroup>,
    stage_bg_r2: Option<wgpu::BindGroup>,
    wg_n4: u32,
    wg_n2: u32,
    result_in_b: bool,
}

pub struct ClaudeFft {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_r4: wgpu::ComputePipeline,
    pipeline_r2: wgpu::ComputePipeline,
    cache: RefCell<std::collections::HashMap<usize, ClaudeCache>>,
}

impl ClaudeFft {
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: true,
        }))
        .expect("no wgpu adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            ..Default::default()
        }))
        .expect("no wgpu device");

        let compile = |src: String, label: &str| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{label}_pipeline")),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipeline_r4 = compile(
            claude_r4_kernel::WGSL_MODULE.wgsl_source().join("\n"),
            "claude_r4",
        );
        let pipeline_r2 = compile(
            claude_r2_kernel::WGSL_MODULE.wgsl_source().join("\n"),
            "claude_r2",
        );

        Self {
            device,
            queue,
            pipeline_r4,
            pipeline_r2,
            cache: RefCell::new(std::collections::HashMap::new()),
        }
    }

    fn build_cache(&self, n: usize, log_n: u32) -> ClaudeCache {
        let num_r4 = (log_n / 2) as usize;
        let has_r2 = log_n % 2 == 1;
        let total_stages = num_r4 + has_r2 as usize;

        let single_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64;
        let max_batch =
            (self.device.limits().max_storage_buffer_binding_size as u64 / single_bytes).min(1024);
        let data_bytes = single_bytes * max_batch;

        let make_buf = |label, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: data_bytes,
                usage,
                mapped_at_creation: false,
            })
        };

        let buf_a = make_buf(
            "claude_buf_a",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buf_b = make_buf(
            "claude_buf_b",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let staging_buf = make_buf(
            "claude_staging",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // N-entry twiddle table: e^{-2πij/N} for j = 0..N.
        // Max radix-4 twiddle index ≈ 3N/4 < N, so this table is always sufficient.
        let twiddles: Vec<f32> = (0..n)
            .flat_map(|j| {
                let angle = -std::f32::consts::TAU * j as f32 / n as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let twiddle_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("claude_twiddles"),
            size: (twiddles.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&twiddle_buf, 0, bytemuck::cast_slice(&twiddles));

        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let entry_bytes = std::mem::size_of::<Uniforms>() as u64;
        let stride = entry_bytes.div_ceil(alignment) * alignment;

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("claude_uniforms"),
            size: stride * total_stages as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        for s in 0..num_r4 {
            self.queue.write_buffer(
                &uniform_buf,
                stride * s as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: s as u32,
                    log_n,
                    _pad: 0,
                }),
            );
        }
        if has_r2 {
            self.queue.write_buffer(
                &uniform_buf,
                stride * num_r4 as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: log_n - 1,
                    log_n,
                    _pad: 0,
                }),
            );
        }

        let uniform_size = NonZeroU64::new(entry_bytes);
        let make_bg = |pipeline: &wgpu::ComputePipeline,
                       src: &wgpu::Buffer,
                       dst: &wgpu::Buffer,
                       offset: u64| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &uniform_buf,
                            offset,
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

        // Combined slot s: even → buf_a → buf_b, odd → buf_b → buf_a.
        let stage_bgs_r4: Vec<wgpu::BindGroup> = (0..num_r4)
            .map(|s| {
                let (src, dst) = if s % 2 == 0 {
                    (&buf_a, &buf_b)
                } else {
                    (&buf_b, &buf_a)
                };
                make_bg(&self.pipeline_r4, src, dst, stride * s as u64)
            })
            .collect();

        let stage_bg_r2 = if has_r2 {
            let (src, dst) = if num_r4 % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            Some(make_bg(&self.pipeline_r2, src, dst, stride * num_r4 as u64))
        } else {
            None
        };

        ClaudeCache {
            buf_a,
            buf_b,
            staging_buf,
            twiddle_buf,
            stage_bgs_r4,
            stage_bg_r2,
            wg_n4: (n as u32 / 4).div_ceil(256),
            wg_n2: (n as u32 / 2).div_ceil(256),
            result_in_b: total_stages % 2 == 1,
        }
    }

    fn get_or_build_cache(&self, n: usize, log_n: u32) -> ClaudeCache {
        let mut map = self.cache.borrow_mut();
        if let Some(c) = map.get(&n) {
            return c.clone();
        }
        let c = self.build_cache(n, log_n);
        map.insert(n, c.clone());
        c
    }

    fn transform_batch_internal(
        &self,
        inputs: &[Vec<Complex<f32>>],
        inverse: bool,
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let n = inputs[0].len();
        assert!(
            n.is_power_of_two() && n > 0,
            "FFT size must be a non-zero power of two"
        );
        let log_n = n.trailing_zeros();
        let batch_size = inputs.len() as u32;

        let cache = self.get_or_build_cache(n, log_n);

        let mut raw: Vec<f32> = Vec::with_capacity(n * 2 * inputs.len());
        for input in inputs {
            assert_eq!(input.len(), n, "all inputs must have the same length");
            if inverse {
                raw.extend(input.iter().flat_map(|c| [c.re, -c.im]));
            } else {
                raw.extend(input.iter().flat_map(|c| [c.re, c.im]));
            }
        }
        self.queue
            .write_buffer(&cache.buf_a, 0, bytemuck::cast_slice(&raw));

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("claude_fft"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("claude_fft_compute"),
                timestamp_writes: None,
            });
            for bg in &cache.stage_bgs_r4 {
                pass.set_pipeline(&self.pipeline_r4);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(cache.wg_n4, batch_size, 1);
            }
            if let Some(r2_bg) = &cache.stage_bg_r2 {
                pass.set_pipeline(&self.pipeline_r2);
                pass.set_bind_group(0, r2_bg, &[]);
                pass.dispatch_workgroups(cache.wg_n2, batch_size, 1);
            }
        }

        let result_buf = if cache.result_in_b {
            &cache.buf_b
        } else {
            &cache.buf_a
        };
        let out_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64 * batch_size as u64;
        enc.copy_buffer_to_buffer(result_buf, 0, &cache.staging_buf, 0, out_bytes);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = cache.staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;

        let mapped = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&mapped);
        let mut output: Vec<Complex<f32>> = floats
            .chunks_exact(2)
            .map(|p| Complex { re: p[0], im: p[1] })
            .collect();
        drop(mapped);
        cache.staging_buf.unmap();

        if inverse {
            let scale = 1.0 / n as f32;
            for c in &mut output {
                *c = Complex {
                    re: c.re * scale,
                    im: -c.im * scale,
                };
            }
        }

        Ok(output.chunks(n).map(|ch| ch.to_vec()).collect())
    }
}

impl FftExecutor for ClaudeFft {
    fn name(&self) -> &str {
        "Claude (Stockham Radix-4/2 Mixed)"
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
}
