use crate::{FftExecutor, GpuFft};
use num_complex::Complex;
use wgsl_rs::wgsl;

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
        let half_n = n >> 1u32;
        if tid >= half_n {
            return;
        }

        let stage = get!(U).y;
        let p = 1u32 << stage;
        let two_p = p + p;
        let three_p = two_p + p;
        let four_p = two_p + two_p;

        let k = tid % p;
        let j = tid / p;

        // Calculate base offset for this batch element
        let batch_offset = batch_id * n * 2u32;

        // Source: natural-order read (Stockham DIT).
        let i1 = j * four_p + k;
        let i2 = i1 + p;
        let i3 = i1 + two_p;
        let i4 = i1 + three_p;

        let src_offset1 = batch_offset + 2u32 * i1;
        let src_offset2 = batch_offset + 2u32 * i2;
        let src_offset3 = batch_offset + 2u32 * i3;
        let src_offset4 = batch_offset + 2u32 * i4;

        let re1 = get!(SRC)[src_offset1];
        let im1 = get!(SRC)[src_offset1 + 1u32];
        let re2 = get!(SRC)[src_offset2];
        let im2 = get!(SRC)[src_offset2 + 1u32];
        let re3 = get!(SRC)[src_offset3];
        let im3 = get!(SRC)[src_offset3 + 1u32];
        let re4 = get!(SRC)[src_offset4];
        let im4 = get!(SRC)[src_offset4 + 1u32];

        // Twiddle lookup: index = k * (half_n / p) = k * (half_n >> stage).
        let twiddle_idx = k * (half_n >> stage);
        let wr1 = get!(TWIDDLE)[2u32 * twiddle_idx];
        let wi1 = get!(TWIDDLE)[2u32 * twiddle_idx + 1u32];
        let wr2 = get!(TWIDDLE)[2u32 * (twiddle_idx * 2u32)];
        let wi2 = get!(TWIDDLE)[2u32 * (twiddle_idx * 2u32) + 1u32];
        let wr3 = get!(TWIDDLE)[2u32 * (twiddle_idx * 3u32)];
        let wi3 = get!(TWIDDLE)[2u32 * (twiddle_idx * 3u32) + 1u32];

        // Radix-4 butterfly
        let tr1 = wr1 * re2 - wi1 * im2;
        let ti1 = wr1 * im2 + wi1 * re2;
        let tr2 = wr2 * re3 - wi2 * im3;
        let ti2 = wr2 * im3 + wi2 * re3;
        let tr3 = wr3 * re4 - wi3 * im4;
        let ti3 = wr3 * im4 + wi3 * re4;

        // Destination: shuffled write for autosort ordering.
        let out1 = j * four_p + k;
        let out2 = out1 + p;
        let out3 = out1 + two_p;
        let out4 = out1 + three_p;

        let dst_offset1 = batch_offset + 2u32 * out1;
        let dst_offset2 = batch_offset + 2u32 * out2;
        let dst_offset3 = batch_offset + 2u32 * out3;
        let dst_offset4 = batch_offset + 2u32 * out4;

        get_mut!(DST)[dst_offset1] = re1 + tr1 + tr2 + tr3;
        get_mut!(DST)[dst_offset1 + 1u32] = im1 + ti1 + ti2 + ti3;
        get_mut!(DST)[dst_offset2] = re1 + tr1 - tr2 - tr3;
        get_mut!(DST)[dst_offset2 + 1u32] = im1 + ti1 - ti2 - ti3;
        get_mut!(DST)[dst_offset3] = re1 - tr1 + tr2 - tr3;
        get_mut!(DST)[dst_offset3 + 1u32] = im1 - ti1 + ti2 - ti3;
        get_mut!(DST)[dst_offset4] = re1 - tr1 - tr2 + tr3;
        get_mut!(DST)[dst_offset4 + 1u32] = im1 - ti1 - ti2 + ti3;
    }
}

pub struct Radix4Rival(GpuFft);

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
}
