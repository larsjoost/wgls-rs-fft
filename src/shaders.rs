use wgsl_rs::wgsl;

/// Stockham autosort FFT kernel.
///
/// Each stage reads from SRC and writes to DST — the two buffers ping-pong
/// between stages, so there is never a read-write hazard between consecutive
/// dispatches. Output is in natural order; no bit-reversal pass is needed.
///
/// Uniform U: .x = N, .y = stage index (0 … log₂N − 1).
/// SRC / DST:   interleaved complex pairs [re₀, im₀, re₁, im₁, …].
/// TWIDDLE:   N/2 precomputed pairs [cos₀, sin₀, cos₁, sin₁, …] where
///            pair j = e^{-2πi·j/N}; accessed at index k*(N/2>>stage).
#[wgsl]
pub mod stockham {
    use wgsl_rs::std::*;

    uniform!(group(0), binding(0), U: Vec4u);
    storage!(group(0), binding(1), read_write, SRC: RuntimeArray<f32>);
    storage!(group(0), binding(2), read_write, DST: RuntimeArray<f32>);
    storage!(group(0), binding(3), TWIDDLE: RuntimeArray<f32>);

    #[compute]
    #[workgroup_size(256)]
    pub fn main(#[builtin(global_invocation_id)] gid: Vec3u) {
        let tid = gid.x;
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

        // Source: natural-order read (Stockham DIT).
        let i1 = j * p + k;
        let i2 = i1 + half_n;

        let re1 = get!(SRC)[2 * i1];
        let im1 = get!(SRC)[2 * i1 + 1];
        let re2 = get!(SRC)[2 * i2];
        let im2 = get!(SRC)[2 * i2 + 1];

        // Twiddle lookup: index = k * (half_n / p) = k * (half_n >> stage).
        let twiddle_idx = k * (half_n >> stage);
        let wr = get!(TWIDDLE)[2 * twiddle_idx];
        let wi = get!(TWIDDLE)[2 * twiddle_idx + 1];

        let tr = wr * re2 - wi * im2;
        let ti = wr * im2 + wi * re2;

        // Destination: shuffled write for autosort ordering.
        let out1 = j * two_p + k;
        let out2 = out1 + p;

        get_mut!(DST)[2 * out1] = re1 + tr;
        get_mut!(DST)[2 * out1 + 1] = im1 + ti;
        get_mut!(DST)[2 * out2] = re1 - tr;
        get_mut!(DST)[2 * out2 + 1] = im1 - ti;
    }
}
