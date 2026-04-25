use crate::FftExecutor;
use num_complex::Complex;
use std::cell::RefCell;
use std::num::NonZeroU64;

const GEMINI_R4_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let batch_id = gid.y;
    let n = U.x;
    let quarter_n = n >> 2u;
    if (tid >= quarter_n) { return; }

    let stage = U.y;
    let p = 1u << (stage + stage);
    let four_p = p << 2u;

    let k = tid % p;
    let j = tid / p;

    let batch_offset = batch_id * n * 2u;

    let i0 = j * p + k;
    let i1 = i0 + quarter_n;
    let i2 = i1 + quarter_n;
    let i3 = i2 + quarter_n;

    let s0 = batch_offset + 2u * i0;
    let s1 = batch_offset + 2u * i1;
    let s2 = batch_offset + 2u * i2;
    let s3 = batch_offset + 2u * i3;

    let x0r = SRC[s0];
    let x0i = SRC[s0 + 1u];
    let x1r = SRC[s1];
    let x1i = SRC[s1 + 1u];
    let x2r = SRC[s2];
    let x2i = SRC[s2 + 1u];
    let x3r = SRC[s3];
    let x3i = SRC[s3 + 1u];

    let stride = quarter_n >> (stage + stage);
    let tw1 = k * stride;
    let tw2 = tw1 * 2u;
    let tw3 = tw1 * 3u;

    let wr1 = TWIDDLE[2u * tw1];
    let wi1 = TWIDDLE[2u * tw1 + 1u];
    let wr2 = TWIDDLE[2u * tw2];
    let wi2 = TWIDDLE[2u * tw2 + 1u];
    let wr3 = TWIDDLE[2u * tw3];
    let wi3 = TWIDDLE[2u * tw3 + 1u];

    let br = wr1 * x1r - wi1 * x1i;
    let bi = wr1 * x1i + wi1 * x1r;
    let cr = wr2 * x2r - wi2 * x2i;
    let ci = wr2 * x2i + wi2 * x2r;
    let dr = wr3 * x3r - wi3 * x3i;
    let di = wr3 * x3i + wi3 * x3r;

    let o0 = j * four_p + k;
    let o1 = o0 + p;
    let o2 = o1 + p;
    let o3 = o2 + p;

    let d0 = batch_offset + 2u * o0;
    let d1 = batch_offset + 2u * o1;
    let d2 = batch_offset + 2u * o2;
    let d3 = batch_offset + 2u * o3;

    DST[d0] = x0r + br + cr + dr;
    DST[d0 + 1u] = x0i + bi + ci + di;
    DST[d1] = x0r + bi - cr - di;
    DST[d1 + 1u] = x0i - br - ci + dr;
    DST[d2] = x0r - br + cr - dr;
    DST[d2 + 1u] = x0i - bi + ci - di;
    DST[d3] = x0r - bi - cr + di;
    DST[d3 + 1u] = x0i + br - ci - dr;
}
"#;

const GEMINI_R2_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    let batch_id = gid.y;
    let n = U.x;
    let half_n = n >> 1u;
    if (tid >= half_n) { return; }

    let stage = U.y;
    let p = 1u << stage;
    let two_p = p + p;

    let k = tid % p;
    let j = tid / p;

    let batch_offset = batch_id * n * 2u;

    let i1 = j * p + k;
    let i2 = i1 + half_n;

    let src1 = batch_offset + 2u * i1;
    let src2 = batch_offset + 2u * i2;

    let re1 = SRC[src1];
    let im1 = SRC[src1 + 1u];
    let re2 = SRC[src2];
    let im2 = SRC[src2 + 1u];

    let twiddle_idx = k * (half_n >> stage);
    let wr = TWIDDLE[2u * twiddle_idx];
    let wi = TWIDDLE[2u * twiddle_idx + 1u];

    let tr = wr * re2 - wi * im2;
    let ti = wr * im2 + wi * re2;

    let out1 = j * two_p + k;
    let out2 = out1 + p;

    let dst1 = batch_offset + 2u * out1;
    let dst2 = batch_offset + 2u * out2;

    DST[dst1] = re1 + tr;
    DST[dst1 + 1u] = im1 + ti;
    DST[dst2] = re1 - tr;
    DST[dst2 + 1u] = im1 - ti;
}
"#;

const GEMINI_LOCAL_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

var<workgroup> lds_a: array<vec2<f32>, 1024>;
var<workgroup> lds_b: array<vec2<f32>, 1024>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = U.x;
    let log_n = U.z;
    let batch_id = wid.x;
    let tid = lid.x;

    let batch_offset = batch_id * n * 2u;
    let loads_per_thread = n / 256u;

    // Load into LDS
    for (var i = 0u; i < loads_per_thread; i = i + 1u) {
        let idx = tid + i * 256u;
        let s_idx = batch_offset + idx * 2u;
        lds_a[idx] = vec2<f32>(SRC[s_idx], SRC[s_idx + 1u]);
    }
    workgroupBarrier();

    var ping_pong = true;
    let num_r4 = log_n / 2u;

    for (var s = 0u; s < num_r4; s = s + 1u) {
        let p = 1u << (s * 2u);
        let quarter_n = n >> 2u;
        
        if (tid < quarter_n) {
            let k = tid % p;
            let j = tid / p;
            let i0 = j * p + k;
            let i1 = i0 + quarter_n;
            let i2 = i1 + quarter_n;
            let i3 = i2 + quarter_n;

            var x0: vec2<f32>; var x1: vec2<f32>; var x2: vec2<f32>; var x3: vec2<f32>;
            if (ping_pong) {
                x0 = lds_a[i0]; x1 = lds_a[i1]; x2 = lds_a[i2]; x3 = lds_a[i3];
            } else {
                x0 = lds_b[i0]; x1 = lds_b[i1]; x2 = lds_b[i2]; x3 = lds_b[i3];
            }

            let stride = quarter_n >> (s * 2u);
            let tw1 = k * stride;
            let tw2 = tw1 * 2u;
            let tw3 = tw1 * 3u;

            let w1 = vec2<f32>(TWIDDLE[2u*tw1], TWIDDLE[2u*tw1+1u]);
            let w2 = vec2<f32>(TWIDDLE[2u*tw2], TWIDDLE[2u*tw2+1u]);
            let w3 = vec2<f32>(TWIDDLE[2u*tw3], TWIDDLE[2u*tw3+1u]);

            let b = complex_mul(w1, x1);
            let c = complex_mul(w2, x2);
            let d = complex_mul(w3, x3);

            let y0 = x0 + b + c + d;
            let y1 = vec2<f32>(x0.x + b.y - c.x - d.y, x0.y - b.x - c.y + d.x);
            let y2 = x0 - b + c - d;
            let y3 = vec2<f32>(x0.x - b.y - c.x + d.y, x0.y + b.x - c.y - d.x);

            let o0 = j * (p * 4u) + k;
            let o1 = o0 + p;
            let o2 = o1 + p;
            let o3 = o2 + p;

            if (ping_pong) {
                lds_b[o0] = y0; lds_b[o1] = y1; lds_b[o2] = y2; lds_b[o3] = y3;
            } else {
                lds_a[o0] = y0; lds_a[o1] = y1; lds_a[o2] = y2; lds_a[o3] = y3;
            }
        }
        ping_pong = !ping_pong;
        workgroupBarrier();
    }

    if (log_n % 2u == 1u) {
        let half_n = n >> 1u;
        if (tid < half_n) {
            let stage = log_n - 1u;
            let p = 1u << stage;
            let k = tid % p;
            let j = tid / p;
            let i1 = j * p + k;
            let i2 = i1 + half_n;

            var x1: vec2<f32>; var x2: vec2<f32>;
            if (ping_pong) {
                x1 = lds_a[i1]; x2 = lds_a[i2];
            } else {
                x1 = lds_b[i1]; x2 = lds_b[i2];
            }

            let tw_idx = k * (half_n >> stage);
            let w = vec2<f32>(TWIDDLE[2u*tw_idx], TWIDDLE[2u*tw_idx+1u]);
            let t = complex_mul(w, x2);

            let y1 = x1 + t;
            let y2 = x1 - t;

            let o1 = j * (p * 2u) + k;
            let o2 = o1 + p;

            if (ping_pong) {
                lds_b[o1] = y1; lds_b[o2] = y2;
            } else {
                lds_a[o1] = y1; lds_a[o2] = y2;
            }
        }
        ping_pong = !ping_pong;
        workgroupBarrier();
    }

    for (var i = 0u; i < loads_per_thread; i = i + 1u) {
        let idx = tid + i * 256u;
        let d_idx = batch_offset + idx * 2u;
        var val: vec2<f32>;
        if (ping_pong) { val = lds_a[idx]; } else { val = lds_b[idx]; }
        DST[d_idx] = val.x;
        DST[d_idx + 1u] = val.y;
    }
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GeminiUniforms {
    n: u32,
    stage: u32,
    log_n: u32,
    _pad: u32,
}

#[derive(Clone)]
struct GeminiCache {
    buf_a: wgpu::Buffer,
    buf_b: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    #[allow(dead_code)]
    twiddle_buf: wgpu::Buffer,
    stage_bgs_r4: Vec<wgpu::BindGroup>,
    stage_bg_r2: Option<wgpu::BindGroup>,
    local_bg: Option<wgpu::BindGroup>,
    wg_n4: u32,
    wg_n2: u32,
    result_in_b: bool,
    is_local: bool,
}

pub struct GeminiFft {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_r4: wgpu::ComputePipeline,
    pipeline_r2: wgpu::ComputePipeline,
    pipeline_local: wgpu::ComputePipeline,
    cache: RefCell<std::collections::HashMap<usize, GeminiCache>>,
}

impl GeminiFft {
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

        let compile = |src: &str, label: &str| {
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

        let pipeline_r4 = compile(GEMINI_R4_WGSL, "gemini_r4");
        let pipeline_r2 = compile(GEMINI_R2_WGSL, "gemini_r2");
        let pipeline_local = compile(GEMINI_LOCAL_WGSL, "gemini_local");

        Self {
            device,
            queue,
            pipeline_r4,
            pipeline_r2,
            pipeline_local,
            cache: RefCell::new(std::collections::HashMap::new()),
        }
    }

    fn build_cache(&self, n: usize, log_n: u32) -> GeminiCache {
        let is_local = n <= 1024;
        let num_r4 = (log_n / 2) as usize;
        let has_r2 = log_n % 2 == 1;
        let total_stages = if is_local {
            1
        } else {
            num_r4 + has_r2 as usize
        };

        let single_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64;
        let max_batch =
            (self.device.limits().max_storage_buffer_binding_size as u64 / single_bytes).min(1024);
        let data_bytes = single_bytes * max_batch;

        let buf_a = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemini_buf_a"),
            size: data_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_b = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemini_buf_b"),
            size: data_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemini_staging"),
            size: data_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let twiddles: Vec<f32> = (0..n)
            .flat_map(|j| {
                let angle = -std::f32::consts::TAU * j as f32 / n as f32;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let twiddle_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemini_twiddles"),
            size: (twiddles.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&twiddle_buf, 0, bytemuck::cast_slice(&twiddles));

        let alignment = self.device.limits().min_uniform_buffer_offset_alignment as u64;
        let entry_bytes = std::mem::size_of::<GeminiUniforms>() as u64;
        let stride = entry_bytes.div_ceil(alignment) * alignment;

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gemini_uniforms"),
            size: stride * total_stages as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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

        if is_local {
            self.queue.write_buffer(
                &uniform_buf,
                0,
                bytemuck::bytes_of(&GeminiUniforms {
                    n: n as u32,
                    stage: 0,
                    log_n,
                    _pad: 0,
                }),
            );
            let local_bg = make_bg(&self.pipeline_local, &buf_a, &buf_b, 0);
            return GeminiCache {
                buf_a,
                buf_b,
                staging_buf,
                twiddle_buf,
                stage_bgs_r4: vec![],
                stage_bg_r2: None,
                local_bg: Some(local_bg),
                wg_n4: 0,
                wg_n2: 0,
                result_in_b: true,
                is_local: true,
            };
        }

        for s in 0..num_r4 {
            self.queue.write_buffer(
                &uniform_buf,
                stride * s as u64,
                bytemuck::bytes_of(&GeminiUniforms {
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
                bytemuck::bytes_of(&GeminiUniforms {
                    n: n as u32,
                    stage: log_n - 1,
                    log_n,
                    _pad: 0,
                }),
            );
        }

        let stage_bgs_r4 = (0..num_r4)
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

        GeminiCache {
            buf_a,
            buf_b,
            staging_buf,
            twiddle_buf,
            stage_bgs_r4,
            stage_bg_r2,
            local_bg: None,
            wg_n4: (n as u32 / 4).div_ceil(256),
            wg_n2: (n as u32 / 2).div_ceil(256),
            result_in_b: total_stages % 2 == 1,
            is_local: false,
        }
    }

    fn get_or_build_cache(&self, n: usize) -> GeminiCache {
        let mut map = self.cache.borrow_mut();
        if let Some(c) = map.get(&n) {
            return c.clone();
        }
        let log_n = n.trailing_zeros();
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
        let batch_size = inputs.len() as u32;
        let cache = self.get_or_build_cache(n);

        let mut raw = Vec::with_capacity(n * 2 * inputs.len());
        for input in inputs {
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
                label: Some("gemini_fft"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemini_fft_compute"),
                timestamp_writes: None,
            });
            if cache.is_local {
                pass.set_pipeline(&self.pipeline_local);
                pass.set_bind_group(0, cache.local_bg.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(batch_size, 1, 1);
            } else {
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
        }

        let result_buf = if cache.result_in_b {
            &cache.buf_b
        } else {
            &cache.buf_a
        };
        let out_bytes = (n * 2 * std::mem::size_of::<f32>()) as u64 * batch_size as u64;
        enc.copy_buffer_to_buffer(result_buf, 0, &cache.staging_buf, 0, out_bytes);
        self.queue.submit(std::iter::once(enc.finish()));

        let slice = cache.staging_buf.slice(0..out_bytes);
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

impl FftExecutor for GeminiFft {
    fn name(&self) -> &str {
        "Gemini (Mixed-Radix Stockham)"
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
