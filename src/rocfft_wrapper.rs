// ROCm FFT wrapper - Note: This only works on AMD GPUs, not NVIDIA
// The GTX 1070 is an NVIDIA GPU and cannot use ROCm

#[cfg(feature = "rocm")]
use rocm_rs::*;

#[cfg(feature = "rocm")]
pub struct RocFft {
    // ROCm FFT implementation would go here
    // This is a placeholder since ROCm doesn't work on NVIDIA GPUs
}

#[cfg(feature = "rocm")]
impl RocFft {
    pub fn new() -> Result<Self, String> {
        Err("ROCm FFT is only supported on AMD GPUs. Your GTX 1070 is an NVIDIA GPU and requires CUDA/cuFFT.".to_string())
    }

    pub fn is_available() -> bool {
        false // ROCm is not available on NVIDIA hardware
    }
}

#[cfg(not(feature = "rocm"))]
pub struct RocFft {
    // Dummy implementation when ROCm is not enabled
}

#[cfg(not(feature = "rocm"))]
impl RocFft {
    pub fn new() -> Result<Self, String> {
        Err("ROCm support not compiled. Use --features rocm to enable (AMD GPUs only)".to_string())
    }

    pub fn is_available() -> bool {
        false
    }
}
