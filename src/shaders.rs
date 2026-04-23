// PI must be a literal here — WGSL has no equivalent of f32::consts::PI.
#![allow(clippy::approx_constant, clippy::excessive_precision)]

use wgsl_rs::wgsl;

/// Bit-reversal permutation kernel.
/// Uniform U (.x=n, .z=log_n); Storage DATA: interleaved [re0,im0,re1,im1,...]
#[wgsl]
pub mod bit_reverse {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U: Vec4u);
    storage!(group(0), binding(1), read_write, DATA: RuntimeArray<f32>);

    pub fn rev(x: u32, bits: u32) -> u32 {
        reverse_bits(x) >> (32u32 - bits)
    }

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] gid: Vec3u) {
        let i = gid.x;
        let n = get!(U).x;
        let log_n = get!(U).z;
        if i >= n {
            return;
        }
        let j = rev(i, log_n);
        if j <= i {
            return;
        }
        let re_i = get!(DATA)[2 * i];
        let im_i = get!(DATA)[2 * i + 1];
        let re_j = get!(DATA)[2 * j];
        let im_j = get!(DATA)[2 * j + 1];
        get_mut!(DATA)[2 * i] = re_j;
        get_mut!(DATA)[2 * i + 1] = im_j;
        get_mut!(DATA)[2 * j] = re_i;
        get_mut!(DATA)[2 * j + 1] = im_i;
    }
}

/// Cooley-Tukey butterfly kernel — one pass per FFT stage.
/// Uniform U (.x=n, .y=stage); Storage DATA: interleaved [re0,im0,re1,im1,...]
#[wgsl]
pub mod fft_stage {
    use wgsl_rs::std::*;

    const PI: f32 = 3.14159265358979;

    uniform!(group(0), binding(0), U: Vec4u);
    storage!(group(0), binding(1), read_write, DATA: RuntimeArray<f32>);

    #[compute]
    #[workgroup_size(64)]
    pub fn main(#[builtin(global_invocation_id)] gid: Vec3u) {
        let tid = gid.x;
        let n = get!(U).x;
        let stage = get!(U).y;
        if tid >= (n >> 1u32) {
            return;
        }

        let half_span = 1u32 << stage;
        let full_span = half_span + half_span;
        let grp = tid / half_span;
        let pos = tid % half_span;

        let idx1 = grp * full_span + pos;
        let idx2 = idx1 + half_span;

        let angle = -2.0 * PI * pos as f32 / full_span as f32;
        let wr = cos(angle);
        let wi = sin(angle);

        let re1 = get!(DATA)[2 * idx1];
        let im1 = get!(DATA)[2 * idx1 + 1];
        let re2 = get!(DATA)[2 * idx2];
        let im2 = get!(DATA)[2 * idx2 + 1];

        let tr = wr * re2 - wi * im2;
        let ti = wr * im2 + wi * re2;

        get_mut!(DATA)[2 * idx1] = re1 + tr;
        get_mut!(DATA)[2 * idx1 + 1] = im1 + ti;
        get_mut!(DATA)[2 * idx2] = re1 - tr;
        get_mut!(DATA)[2 * idx2 + 1] = im1 - ti;
    }
}
