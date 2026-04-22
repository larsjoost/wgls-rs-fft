mod shaders;

use std::mem::size_of;
use num_complex::Complex;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FftUniforms {
    n:     u32,
    stage: u32,
    log_n: u32,
    _pad:  u32,
}

pub struct GpuFft {
    device: wgpu::Device,
    queue:  wgpu::Queue,
    bit_reverse_pipeline: wgpu::ComputePipeline,
    fft_stage_pipeline:   wgpu::ComputePipeline,
}

impl GpuFft {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Result<Self, Box<dyn std::error::Error>> {
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
            source: wgpu::ShaderSource::Wgsl(shaders::BIT_REVERSE_WGSL.into()),
        });
        let fft_stage_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fft_stage"),
            source: wgpu::ShaderSource::Wgsl(shaders::FFT_STAGE_WGSL.into()),
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
        let fft_stage_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fft_stage_pipeline"),
                layout: None,
                module: &fft_stage_mod,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self { device, queue, bit_reverse_pipeline, fft_stage_pipeline })
    }

    pub fn fft(&self, input: &[Complex<f32>]) -> Result<Vec<Complex<f32>>, Box<dyn std::error::Error>> {
        let n = input.len();
        assert!(n.is_power_of_two() && n > 0, "FFT length must be a power of two");
        let log_n = n.trailing_zeros();

        let raw: Vec<f32> = input.iter().flat_map(|c| [c.re, c.im]).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&raw);

        let storage_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fft_storage"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let uniform_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_uniform"),
            size: size_of::<FftUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft_staging"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let make_bind_group = |pipeline: &wgpu::ComputePipeline| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: storage_buf.as_entire_binding(),
                    },
                ],
            })
        };

        // Bit-reversal pass
        self.queue.write_buffer(
            &uniform_buf,
            0,
            bytemuck::bytes_of(&FftUniforms { n: n as u32, stage: 0, log_n, _pad: 0 }),
        );
        {
            let bg = make_bind_group(&self.bit_reverse_pipeline);
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.bit_reverse_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((n as u32 + 63) / 64, 1, 1);
            }
            self.queue.submit(std::iter::once(enc.finish()));
        }

        // FFT butterfly stages
        let fft_bg = make_bind_group(&self.fft_stage_pipeline);
        for stage in 0..log_n {
            self.queue.write_buffer(
                &uniform_buf,
                0,
                bytemuck::bytes_of(&FftUniforms { n: n as u32, stage, log_n, _pad: 0 }),
            );
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.fft_stage_pipeline);
                pass.set_bind_group(0, &fft_bg, &[]);
                pass.dispatch_workgroups((n as u32 / 2 + 63) / 64, 1, 1);
            }
            self.queue.submit(std::iter::once(enc.finish()));
        }

        // Copy to staging and read back
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&storage_buf, 0, &staging_buf, 0, bytes.len() as u64);
            self.queue.submit(std::iter::once(enc.finish()));
        }

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })?;
        rx.recv()??;

        let mapped = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&*mapped);
        let output: Vec<Complex<f32>> = floats
            .chunks_exact(2)
            .map(|p| Complex { re: p[0], im: p[1] })
            .collect();
        drop(mapped);
        staging_buf.unmap();

        Ok(output)
    }
}
