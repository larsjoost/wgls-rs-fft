use std::cell::RefCell;
use std::num::NonZeroU64;

use num_complex::Complex;
use wgsl_rs::wgsl;

use crate::FftExecutor;

// ── WGSL: Stockham Radix-8 DIT ───────────────────────────────────────────────
//
// Uniform U: .x = N, .y = radix-8 stage index s (0 .. log₈N-1)
//
// For stage s:
//   p        = 8^s  = 1u32 << (s*3)
//   eighth_n  = N/8
//   stride   = eighth_n >> (s*3)   = eighth_n / p
//
// Source indices (natural-order Stockham read):
//   i0 = j*p+k,  i1 = i0+N/8,  i2 = i0+N/4,  i3 = i0+3N/8
//   i4 = i0+N/2,  i5 = i0+5N/8,  i6 = i0+3N/4,  i7 = i0+7N/8
//
// Twiddle indices into the N-entry table:
//   tw1 = k*stride,  tw2 = 2*tw1,  tw3 = 3*tw1,  tw4 = 4*tw1
//   tw5 = 5*tw1,  tw6 = 6*tw1,  tw7 = 7*tw1
//
// Output indices (Stockham autosort):
//   o0 = j*8p+k,  o1 = o0+p,  o2 = o0+2p,  o3 = o0+3p
//   o4 = o0+4p,  o5 = o0+5p,  o6 = o0+6p,  o7 = o0+7p
#[wgsl]
pub mod mistral_r8_kernel {
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
        let eighth_n = n >> 3u32;
        if tid >= eighth_n {
            return;
        }

        let stage = get!(U).y;
        let p = 1u32 << (stage + stage + stage);
        let eight_p = p << 3u32;

        let k = tid % p;
        let j = tid / p;

        let batch_offset = batch_id * n * 2u32;

        let i0 = j * p + k;
        let i1 = i0 + eighth_n;
        let i2 = i0 + eighth_n + eighth_n;
        let i3 = i2 + eighth_n;
        let i4 = i0 + eighth_n + eighth_n + eighth_n + eighth_n;
        let i5 = i4 + eighth_n;
        let i6 = i4 + eighth_n + eighth_n;
        let i7 = i6 + eighth_n;

        let s0 = batch_offset + 2u32 * i0;
        let s1 = batch_offset + 2u32 * i1;
        let s2 = batch_offset + 2u32 * i2;
        let s3 = batch_offset + 2u32 * i3;
        let s4 = batch_offset + 2u32 * i4;
        let s5 = batch_offset + 2u32 * i5;
        let s6 = batch_offset + 2u32 * i6;
        let s7 = batch_offset + 2u32 * i7;

        let x0r = get!(SRC)[s0];
        let x0i = get!(SRC)[s0 + 1u32];
        let x1r = get!(SRC)[s1];
        let x1i = get!(SRC)[s1 + 1u32];
        let x2r = get!(SRC)[s2];
        let x2i = get!(SRC)[s2 + 1u32];
        let x3r = get!(SRC)[s3];
        let x3i = get!(SRC)[s3 + 1u32];
        let x4r = get!(SRC)[s4];
        let x4i = get!(SRC)[s4 + 1u32];
        let x5r = get!(SRC)[s5];
        let x5i = get!(SRC)[s5 + 1u32];
        let x6r = get!(SRC)[s6];
        let x6i = get!(SRC)[s6 + 1u32];
        let x7r = get!(SRC)[s7];
        let x7i = get!(SRC)[s7 + 1u32];

        let stride = eighth_n >> (stage + stage + stage);
        let tw1 = k * stride;
        let tw2 = tw1 * 2u32;
        let tw3 = tw1 * 3u32;
        let tw4 = tw1 * 4u32;
        let tw5 = tw1 * 5u32;
        let tw6 = tw1 * 6u32;
        let tw7 = tw1 * 7u32;

        let wr1 = get!(TWIDDLE)[2u32 * tw1];
        let wi1 = get!(TWIDDLE)[2u32 * tw1 + 1u32];
        let wr2 = get!(TWIDDLE)[2u32 * tw2];
        let wi2 = get!(TWIDDLE)[2u32 * tw2 + 1u32];
        let wr3 = get!(TWIDDLE)[2u32 * tw3];
        let wi3 = get!(TWIDDLE)[2u32 * tw3 + 1u32];
        let wr4 = get!(TWIDDLE)[2u32 * tw4];
        let wi4 = get!(TWIDDLE)[2u32 * tw4 + 1u32];
        let wr5 = get!(TWIDDLE)[2u32 * tw5];
        let wi5 = get!(TWIDDLE)[2u32 * tw5 + 1u32];
        let wr6 = get!(TWIDDLE)[2u32 * tw6];
        let wi6 = get!(TWIDDLE)[2u32 * tw6 + 1u32];
        let wr7 = get!(TWIDDLE)[2u32 * tw7];
        let wi7 = get!(TWIDDLE)[2u32 * tw7 + 1u32];

        // Radix-8 butterfly
        let b1r = wr1 * x1r - wi1 * x1i;
        let b1i = wr1 * x1i + wi1 * x1r;
        let b2r = wr2 * x2r - wi2 * x2i;
        let b2i = wr2 * x2i + wi2 * x2r;
        let b3r = wr3 * x3r - wi3 * x3i;
        let b3i = wr3 * x3i + wi3 * x3r;
        let b4r = wr4 * x4r - wi4 * x4i;
        let b4i = wr4 * x4i + wi4 * x4r;
        let b5r = wr5 * x5r - wi5 * x5i;
        let b5i = wr5 * x5i + wi5 * x5r;
        let b6r = wr6 * x6r - wi6 * x6i;
        let b6i = wr6 * x6i + wi6 * x6r;
        let b7r = wr7 * x7r - wi7 * x7i;
        let b7i = wr7 * x7i + wi7 * x7r;

        let o0 = j * eight_p + k;
        let o1 = o0 + p;
        let o2 = o0 + p + p;
        let o3 = o2 + p;
        let o4 = o0 + p + p + p + p;
        let o5 = o4 + p;
        let o6 = o4 + p + p;
        let o7 = o6 + p;

        let d0 = batch_offset + 2u32 * o0;
        let d1 = batch_offset + 2u32 * o1;
        let d2 = batch_offset + 2u32 * o2;
        let d3 = batch_offset + 2u32 * o3;
        let d4 = batch_offset + 2u32 * o4;
        let d5 = batch_offset + 2u32 * o5;
        let d6 = batch_offset + 2u32 * o6;
        let d7 = batch_offset + 2u32 * o7;

        // Radix-8 output equations - using standard DIT butterfly
        // Y[k] = sum_{m=0}^{7} X[m] * W^{k*m} where W = e^{-2πi/8}
        get_mut!(DST)[d0] = x0r + b1r + b2r + b3r + b4r + b5r + b6r + b7r;
        get_mut!(DST)[d0 + 1u32] = x0i + b1i + b2i + b3i + b4i + b5i + b6i + b7i;

        get_mut!(DST)[d1] = x0r + b1r + b2r + b3r - b4r - b5r - b6r - b7r;
        get_mut!(DST)[d1 + 1u32] = x0i + b1i + b2i + b3i - b4i - b5i - b6i - b7i;

        get_mut!(DST)[d2] = x0r + b1r - b2r + b3r - b4r - b5r + b6r + b7r;
        get_mut!(DST)[d2 + 1u32] = x0i + b1i - b2i + b3i - b4i - b5i + b6i + b7i;

        get_mut!(DST)[d3] = x0r + b1r - b2r - b3r + b4r + b5r - b6r - b7r;
        get_mut!(DST)[d3 + 1u32] = x0i + b1i - b2i - b3i + b4i + b5i - b6i - b7i;

        get_mut!(DST)[d4] = x0r - b1r + b2r - b3r - b4r + b5r - b6r + b7r;
        get_mut!(DST)[d4 + 1u32] = x0i - b1i + b2i - b3i - b4i + b5i - b6i + b7i;

        get_mut!(DST)[d5] = x0r - b1r + b2r + b3r + b4r - b5r - b6r - b7r;
        get_mut!(DST)[d5 + 1u32] = x0i - b1i + b2i + b3i + b4i - b5i - b6i - b7i;

        get_mut!(DST)[d6] = x0r - b1r - b2r + b3r + b4r + b5r + b6r - b7r;
        get_mut!(DST)[d6 + 1u32] = x0i - b1i - b2i + b3i + b4i + b5i + b6i - b7i;

        get_mut!(DST)[d7] = x0r - b1r - b2r - b3r - b4r - b5r - b6r + b7r;
        get_mut!(DST)[d7 + 1u32] = x0i - b1i - b2i - b3i - b4i - b5i - b6i + b7i;
    }
}

// ── WGSL: Stockham Radix-4 DIT (for when log₂N % 3 != 0) ────────────────────
#[wgsl]
pub mod mistral_r4_kernel {
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

// ── WGSL: Stockham Radix-2 (finalisation for odd log₂N) ────────────────────
#[wgsl]
pub mod mistral_r2_kernel {
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
struct MistralCache {
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    #[allow(dead_code)]
    twiddle_buf: wgpu::Buffer,
    stage_bgs_r8: Vec<wgpu::BindGroup>,
    stage_bgs_r4: Vec<wgpu::BindGroup>,
    stage_bg_r2: Option<wgpu::BindGroup>,
    wg_n8: u32,
    wg_n4: u32,
    wg_n2: u32,
    result_in_b: bool,
}

pub struct MistralVibeFft {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_r8: wgpu::ComputePipeline,
    pipeline_r4: wgpu::ComputePipeline,
    pipeline_r2: wgpu::ComputePipeline,
    cache: RefCell<std::collections::HashMap<usize, MistralCache>>,
}

impl MistralVibeFft {
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

        let pipeline_r8 = compile(
            mistral_r8_kernel::WGSL_MODULE.wgsl_source().join("\n"),
            "mistral_r8",
        );
        let pipeline_r4 = compile(
            mistral_r4_kernel::WGSL_MODULE.wgsl_source().join("\n"),
            "mistral_r4",
        );
        let pipeline_r2 = compile(
            mistral_r2_kernel::WGSL_MODULE.wgsl_source().join("\n"),
            "mistral_r2",
        );

        Self {
            device,
            queue,
            pipeline_r8,
            pipeline_r4,
            pipeline_r2,
            cache: RefCell::new(std::collections::HashMap::new()),
        }
    }

    fn build_cache(&self, n: usize, log_n: u32) -> MistralCache {
        // For now, use only Radix-4 and Radix-2 like Claude to ensure correctness
        let num_r8 = 0;
        let num_r4 = (log_n / 2) as usize;
        let has_r2 = log_n % 2 == 1;
        let total_stages = num_r8 + num_r4 + has_r2 as usize;

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
            "mistral_buf_a",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buf_b = make_buf(
            "mistral_buf_b",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let staging_buf = make_buf(
            "mistral_staging",
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // N-entry twiddle table: e^{-2πij/N} for j = 0..N.
        let twiddles: Vec<f32> = (0..n)
            .flat_map(|j| {
                let angle = -std::f32::consts::TAU * j as f32 / n as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let twiddle_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mistral_twiddles"),
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
            label: Some("mistral_uniforms"),
            size: stride * total_stages as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut stage_idx = 0;

        // Radix-8 stages
        for s in 0..num_r8 {
            self.queue.write_buffer(
                &uniform_buf,
                stride * stage_idx as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: s as u32,
                    log_n,
                    _pad: 0,
                }),
            );
            stage_idx += 1;
        }

        // Radix-4 stages
        for s in 0..num_r4 {
            self.queue.write_buffer(
                &uniform_buf,
                stride * stage_idx as u64,
                bytemuck::bytes_of(&Uniforms {
                    n: n as u32,
                    stage: s as u32,
                    log_n,
                    _pad: 0,
                }),
            );
            stage_idx += 1;
        }

        // Radix-2 stage (if needed)
        if has_r2 {
            self.queue.write_buffer(
                &uniform_buf,
                stride * stage_idx as u64,
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
        let mut stage_bgs_r8: Vec<wgpu::BindGroup> = Vec::new();
        let mut stage_bgs_r4: Vec<wgpu::BindGroup> = Vec::new();

        stage_idx = 0;
        for _s in 0..num_r8 {
            let (src, dst) = if stage_idx % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            stage_bgs_r8.push(make_bg(
                &self.pipeline_r8,
                src,
                dst,
                stride * stage_idx as u64,
            ));
            stage_idx += 1;
        }

        // Radix-4 stages
        for _s in 0..num_r4 {
            let (src, dst) = if stage_idx % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            stage_bgs_r4.push(make_bg(
                &self.pipeline_r4,
                src,
                dst,
                stride * stage_idx as u64,
            ));
            stage_idx += 1;
        }

        let stage_bg_r2 = if has_r2 {
            let (src, dst) = if stage_idx % 2 == 0 {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };
            Some(make_bg(
                &self.pipeline_r2,
                src,
                dst,
                stride * stage_idx as u64,
            ))
        } else {
            None
        };

        MistralCache {
            buf_a,
            buf_b,
            staging_buf,
            twiddle_buf,
            stage_bgs_r8,
            stage_bgs_r4,
            stage_bg_r2,
            wg_n8: (n as u32 / 8).div_ceil(256),
            wg_n4: (n as u32 / 4).div_ceil(256),
            wg_n2: (n as u32 / 2).div_ceil(256),
            result_in_b: total_stages % 2 == 1,
        }
    }

    fn get_or_build_cache(&self, n: usize, log_n: u32) -> MistralCache {
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
                label: Some("mistral_fft"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mistral_fft_compute"),
                timestamp_writes: None,
            });

            // Radix-8 stages
            for bg in &cache.stage_bgs_r8 {
                pass.set_pipeline(&self.pipeline_r8);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(cache.wg_n8, batch_size, 1);
            }

            // Radix-4 stage (if needed)
            for bg in &cache.stage_bgs_r4 {
                pass.set_pipeline(&self.pipeline_r4);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(cache.wg_n4, batch_size, 1);
            }

            // Radix-2 stage (if needed)
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

impl FftExecutor for MistralVibeFft {
    fn name(&self) -> &str {
        "MistralVibe (Stockham Radix-4/2 Mixed)"
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
