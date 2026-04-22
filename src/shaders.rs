pub(crate) const BIT_REVERSE_WGSL: &str = r#"
struct Uniforms {
    n:     u32,
    stage: u32,
    log_n: u32,
    _pad:  u32,
}

@group(0) @binding(0) var<uniform>            u:    Uniforms;
@group(0) @binding(1) var<storage,read_write> data: array<f32>;

fn rev(x: u32, bits: u32) -> u32 {
    return reverseBits(x) >> (32u - bits);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= u.n { return; }
    let j = rev(i, u.log_n);
    if j <= i { return; }
    let ri = data[2u * i];
    let ii = data[2u * i + 1u];
    data[2u * i]      = data[2u * j];
    data[2u * i + 1u] = data[2u * j + 1u];
    data[2u * j]      = ri;
    data[2u * j + 1u] = ii;
}
"#;

pub(crate) const FFT_STAGE_WGSL: &str = r#"
const PI: f32 = 3.14159265358979323846;

struct Uniforms {
    n:     u32,
    stage: u32,
    log_n: u32,
    _pad:  u32,
}

@group(0) @binding(0) var<uniform>            u:    Uniforms;
@group(0) @binding(1) var<storage,read_write> data: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if tid >= u.n >> 1u { return; }

    let half_span = 1u << u.stage;
    let full_span = half_span << 1u;
    let group     = tid / half_span;
    let pos       = tid % half_span;

    let idx1 = group * full_span + pos;
    let idx2 = idx1 + half_span;

    let angle = -2.0 * PI * f32(pos) / f32(full_span);
    let wr = cos(angle);
    let wi = sin(angle);

    let re1 = data[2u * idx1];
    let im1 = data[2u * idx1 + 1u];
    let re2 = data[2u * idx2];
    let im2 = data[2u * idx2 + 1u];

    let tr = wr * re2 - wi * im2;
    let ti = wr * im2 + wi * re2;

    data[2u * idx1]      = re1 + tr;
    data[2u * idx1 + 1u] = im1 + ti;
    data[2u * idx2]      = re1 - tr;
    data[2u * idx2 + 1u] = im1 - ti;
}
"#;
