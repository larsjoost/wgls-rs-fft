/// Stockham Radix-4 DIT kernel (derived from Gemini rivalry winner).
///
/// Uniform U: .x = N (transform length), .y = p (stride = 4^stage_index).
/// SRC / DST:  interleaved complex pairs [re₀, im₀, re₁, im₁, …].
/// TWIDDLE:    N complex pairs e^{-2πij/N} for j = 0..N.
///
/// Dispatched log₄N times; each dispatch processes N/4 butterfly groups.
pub const R4_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid       = gid.x;
    let batch_id  = gid.y;
    let n         = U.x;
    let p         = U.y;
    let quarter_n = n >> 2u;
    if tid >= quarter_n { return; }

    let four_p = p << 2u;
    let k      = tid % p;
    let j      = tid / p;
    let bo     = batch_id * n * 2u;

    let i0 = j*p + k;
    let i1 = i0 + quarter_n;
    let i2 = i1 + quarter_n;
    let i3 = i2 + quarter_n;

    var x: array<vec2<f32>, 4>;
    x[0] = vec2<f32>(SRC[bo + 2u*i0], SRC[bo + 2u*i0+1u]);
    x[1] = vec2<f32>(SRC[bo + 2u*i1], SRC[bo + 2u*i1+1u]);
    x[2] = vec2<f32>(SRC[bo + 2u*i2], SRC[bo + 2u*i2+1u]);
    x[3] = vec2<f32>(SRC[bo + 2u*i3], SRC[bo + 2u*i3+1u]);

    let stride = quarter_n / p;
    let tw     = k * stride;
    x[1] = cmul(vec2<f32>(TWIDDLE[2u*tw],   TWIDDLE[2u*tw+1u]),   x[1]);
    x[2] = cmul(vec2<f32>(TWIDDLE[4u*tw],   TWIDDLE[4u*tw+1u]),   x[2]);
    x[3] = cmul(vec2<f32>(TWIDDLE[6u*tw],   TWIDDLE[6u*tw+1u]),   x[3]);

    let s02 = x[0] + x[2]; let d02 = x[0] - x[2];
    let s13 = x[1] + x[3]; let d13 = x[1] - x[3];

    let y0 = s02 + s13;
    let y1 = vec2<f32>(d02.x + d13.y, d02.y - d13.x);
    let y2 = s02 - s13;
    let y3 = vec2<f32>(d02.x - d13.y, d02.y + d13.x);

    let d_base = bo + 2u*(j*four_p + k);
    DST[d_base]          = y0.x; DST[d_base+1u]          = y0.y;
    DST[d_base + 2u*p]   = y1.x; DST[d_base + 2u*p+1u]   = y1.y;
    DST[d_base + 4u*p]   = y2.x; DST[d_base + 4u*p+1u]   = y2.y;
    DST[d_base + 6u*p]   = y3.x; DST[d_base + 6u*p+1u]   = y3.y;
}
"#;

/// Stockham Radix-2 DIT kernel — fallback for the final stage when log₂N is odd.
///
/// Uniform U: .x = N, .y = p (stride = 2^(num_r4_stages * 2)).
/// TWIDDLE:   N complex pairs e^{-2πij/N} for j = 0..N.
pub const R2_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> U: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> SRC: array<f32>;
@group(0) @binding(2) var<storage, read_write> DST: array<f32>;
@group(0) @binding(3) var<storage, read> TWIDDLE: array<f32>;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid      = gid.x;
    let batch_id = gid.y;
    let n        = U.x;
    let p        = U.y;
    let half_n   = n >> 1u;
    if tid >= half_n { return; }

    let two_p = p + p;
    let k     = tid % p;
    let j     = tid / p;
    let bo    = batch_id * n * 2u;

    let i1 = j*p + k;
    let i2 = i1 + half_n;

    let x1 = vec2<f32>(SRC[bo + 2u*i1], SRC[bo + 2u*i1+1u]);
    let x2 = vec2<f32>(SRC[bo + 2u*i2], SRC[bo + 2u*i2+1u]);

    let tw = k * (half_n / p);
    let t  = cmul(vec2<f32>(TWIDDLE[2u*tw], TWIDDLE[2u*tw+1u]), x2);

    let d_base = bo + 2u*(j*two_p + k);
    DST[d_base]          = x1.x + t.x; DST[d_base+1u]          = x1.y + t.y;
    DST[d_base + 2u*p]   = x1.x - t.x; DST[d_base + 2u*p+1u]   = x1.y - t.y;
}
"#;
