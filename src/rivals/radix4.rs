use std::any::Any;

use crate::{FftExecutor, GpuFft};
use num_complex::Complex;
use wgsl_rs::wgsl;

// Need wgpu for the GpuFftTrait implementation
use wgpu;

#[wgsl]
pub mod radix4_kernel {
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
        let stage = get!(U).y;
        let batch_offset = batch_id * n * 2u32;

        // For Radix-4, we only process even stages (0, 2, 4, ...)
        // Odd stages (1, 3, 5, ...) are identity operations (copy SRC to DST)
        if stage % 2u32 != 0u32 {
            // Each thread in a radix-2 dispatch processes 2 elements normally.
            // Since we are dispatched with n/2 threads, each thread can copy 2 complex numbers (4 floats).
            let idx1 = tid;
            let idx2 = tid + (n >> 1u32);

            if idx1 < n {
                let s1 = batch_offset + 2u32 * idx1;
                get_mut!(DST)[s1] = get!(SRC)[s1];
                get_mut!(DST)[s1 + 1u32] = get!(SRC)[s1 + 1u32];
            }
            if idx2 < n {
                let s2 = batch_offset + 2u32 * idx2;
                get_mut!(DST)[s2] = get!(SRC)[s2];
                get_mut!(DST)[s2 + 1u32] = get!(SRC)[s2 + 1u32];
            }
            return;
        }

        let quarter_n = n >> 2u32;
        if tid >= quarter_n {
            return;
        }

        let r4_stage = stage / 2u32;

        let p = 1u32 << (r4_stage + r4_stage);
        let four_p = p << 2u32;

        let k = tid % p;
        let j = tid / p;

        // Source: natural-order read (Stockham DIT).
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

        let stride = quarter_n >> (r4_stage + r4_stage);
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

pub struct Radix4Rival(pub GpuFft);

impl Radix4Rival {
    pub fn new() -> Self {
        let wgsl = radix4_kernel::WGSL_MODULE.wgsl_source().join("\n");
        Self(GpuFft::with_shader(wgsl, "radix4_rival").unwrap())
    }
}

impl FftExecutor for Radix4Rival {
    fn name(&self) -> &str {
        "Radix-4 Rival"
    }

    fn fft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.0.fft(inputs)
    }

    fn ifft(
        &self,
        inputs: &[Vec<Complex<f32>>],
    ) -> Result<Vec<Vec<Complex<f32>>>, Box<dyn std::error::Error>> {
        self.0.ifft(inputs)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl crate::GpuFftTrait for Radix4Rival {
    fn benchmark_gpu_only(
        &self,
        sc: &crate::SizeCache,
        batch_size: u32,
        n: usize,
        warmup_iters: usize,
        bench_iters: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        self.0
            .benchmark_gpu_only(sc, batch_size, n, warmup_iters, bench_iters)
    }

    fn get_or_build_size_cache(&self, n: usize, log_n: u32) -> crate::SizeCache {
        self.0.get_or_build_size_cache(n, log_n)
    }

    fn prepare_input_data(&self, input: &[Complex<f32>], inverse: bool) -> Vec<f32> {
        self.0.prepare_input_data(input, inverse)
    }

    fn queue(&self) -> &wgpu::Queue {
        self.0.queue()
    }
}
