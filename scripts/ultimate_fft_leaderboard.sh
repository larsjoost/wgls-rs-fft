#!/usr/bin/env bash

# Ultimate FFT Rivalry Leaderboard Script
# Handles detection, retries, and host/direct execution without interactive prompts.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "🚀 Ultimate FFT Rivalry Leaderboard with cuFFT Support"
echo "===================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Globals
NVIDIA_GPU=false
NVIDIA_SMI_WORKS=false
NVIDIA_GPU_NAME=""

RUST_AVAILABLE=false

CONTAINER_RUNTIME="none"
CONTAINER_CLI=""
CONTAINER_GPU_WORKS=false
CONTAINER_GPU_FLAGS=(--gpus all)
GPU_PROBE_IMAGE="docker.io/nvidia/cuda:11.8.0-base-ubuntu22.04"

DOCKER_IMAGE="fft-rivalry-leaderboard"
DOCKERFILE_PATH="scripts/Dockerfile.cuda"

CUDA_BINDGEN_EXTRA_ARGS=""
CUDA_BUILD_LOG=""
CUDA_FAILURE_KIND="none"
CUDA_RUNTIME_LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

RUN_SUCCESS=false

DOCKER_TTY_FLAGS=()
if [ -t 0 ] && [ -t 1 ]; then
    DOCKER_TTY_FLAGS=(-it)
fi

print_status() {
    local color="$1"
    shift
    echo -e "${color}$*${NC}"
}

print_section() {
    echo ""
    print_status "$BLUE" "=== $1 ==="
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

show_troubleshooting() {
    print_section "Troubleshooting & Actions"
    case "$1" in
        "cuda_bindgen_incompatible")
            print_status "$YELLOW" "❌ CUDA Build Failed: bindgen/CUDA header incompatibility"
            print_status "$NC" "Automatic retries were attempted (compat include paths, pkg-config bypass)."
            print_status "$NC" "Current host CUDA toolchain appears incompatible with the configured cuFFT backend."
            print_status "$YELLOW" "Automatic fallback applied: running WebGPU leaderboard without cuFFT."
            ;;
        "cuda_build_fail")
            print_status "$YELLOW" "❌ CUDA Build Failed"
            print_status "$NC" "Automatic retries were attempted (compat include paths, multiple build modes)."
            print_status "$NC" "Likely reasons:"
            print_status "$NC" "1. Missing system C/C++ headers or clang resources"
            print_status "$NC" "2. CUDA headers/libraries mismatch on host"
            print_status "$YELLOW" "Automatic fallback applied: running WebGPU leaderboard without cuFFT."
            ;;
        "docker_fail")
            print_status "$YELLOW" "❌ Container execution failed"
            print_status "$NC" "Likely reasons:"
            print_status "$NC" "1. Container runtime not fully configured for rootless GPU access"
            print_status "$NC" "2. Registry configuration blocks image pull/build"
            print_status "$NC" "3. NVIDIA container runtime integration missing"
            ;;
        "wgpu_fail")
            print_status "$YELLOW" "❌ WebGPU/WGPU initialization failed"
            print_status "$NC" "Likely reasons:"
            print_status "$NC" "1. No compatible Vulkan/Metal/DX12 drivers found"
            print_status "$NC" "2. Running in an environment without GPU/display access"
            ;;
    esac
}

print_log_excerpt() {
    local log_path="$1"
    if [ -f "$log_path" ]; then
        print_status "$YELLOW" "   Last build log lines:"
        tail -n 12 "$log_path" | sed 's/^/   /'
    fi
}

classify_cuda_failure() {
    local log_path="$1"
    CUDA_FAILURE_KIND="unknown"

    if grep -q "is not a valid Ident" "$log_path"; then
        CUDA_FAILURE_KIND="bindgen_ident"
    elif grep -q "fatal error: 'limits.h' file not found" "$log_path"; then
        CUDA_FAILURE_KIND="missing_limits"
    elif grep -q "fatal error: 'utility' file not found" "$log_path"; then
        CUDA_FAILURE_KIND="missing_cpp_headers"
    elif grep -q "Unable to generate bindings" "$log_path"; then
        CUDA_FAILURE_KIND="bindgen_generate"
    fi
}

run_as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    elif command_exists sudo && sudo -n true >/dev/null 2>&1; then
        sudo "$@"
    else
        return 1
    fi
}

can_install_packages_noninteractively() {
    if ! command_exists apt-get; then
        return 1
    fi
    if [ "$(id -u)" -eq 0 ]; then
        return 0
    fi
    command_exists sudo && sudo -n true >/dev/null 2>&1
}

try_install_cuda_build_dependencies() {
    if ! can_install_packages_noninteractively; then
        return 1
    fi

    print_status "$YELLOW" "🔧 Installing missing CUDA build dependencies automatically..."
    if ! run_as_root apt-get update -y >/tmp/fft_apt_update.log 2>&1; then
        print_status "$YELLOW" "⚠️  apt-get update failed; continuing with existing environment."
        return 1
    fi

    if ! run_as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential g++ clang pkg-config libc6-dev >/tmp/fft_apt_install.log 2>&1; then
        print_status "$YELLOW" "⚠️  Automatic dependency install failed; continuing with existing environment."
        return 1
    fi

    print_status "$GREEN" "✅ Dependency install completed"
    return 0
}

compute_bindgen_compat_args() {
    local cxx_ver=""
    local gcc_include=""

    if [ -d /usr/include/c++ ]; then
        cxx_ver="$(ls -1 /usr/include/c++ 2>/dev/null | sort -V | tail -1 || true)"
    fi
    gcc_include="$(ls -d /usr/lib/gcc/x86_64-linux-gnu/*/include 2>/dev/null | sort -V | tail -1 || true)"

    local args=("-x c++" "-std=c++17")
    if [ -n "$cxx_ver" ]; then
        args+=("-isystem /usr/include/c++/${cxx_ver}")
        if [ -d "/usr/include/x86_64-linux-gnu/c++/${cxx_ver}" ]; then
            args+=("-isystem /usr/include/x86_64-linux-gnu/c++/${cxx_ver}")
        fi
    fi
    if [ -n "$gcc_include" ]; then
        args+=("-isystem ${gcc_include}")
    fi
    args+=("-isystem /usr/include" "-isystem /usr/include/x86_64-linux-gnu")

    CUDA_BINDGEN_EXTRA_ARGS="${args[*]}"
}

run_cuda_build_mode() {
    local mode="$1"
    local log_path="$2"

    case "$mode" in
        "plain")
            env LD_LIBRARY_PATH="$CUDA_RUNTIME_LD_LIBRARY_PATH" \
                cargo build --release --features cuda >"$log_path" 2>&1
            ;;
        "compat")
            env LD_LIBRARY_PATH="$CUDA_RUNTIME_LD_LIBRARY_PATH" \
                BINDGEN_EXTRA_CLANG_ARGS="$CUDA_BINDGEN_EXTRA_ARGS" \
                cargo build --release --features cuda >"$log_path" 2>&1
            ;;
        "compat_no_pkg")
            env LD_LIBRARY_PATH="$CUDA_RUNTIME_LD_LIBRARY_PATH" \
                CUDA_NO_PKG_CONFIG=1 BINDGEN_EXTRA_CLANG_ARGS="$CUDA_BINDGEN_EXTRA_ARGS" \
                cargo build --release --features cuda >"$log_path" 2>&1
            ;;
        *)
            return 1
            ;;
    esac
}

run_cuda_leaderboard_mode() {
    local mode="$1"

    case "$mode" in
        "plain")
            env LD_LIBRARY_PATH="$CUDA_RUNTIME_LD_LIBRARY_PATH" \
                cargo run --example rivalry_leaderboard --release --features cuda
            ;;
        "compat")
            env LD_LIBRARY_PATH="$CUDA_RUNTIME_LD_LIBRARY_PATH" \
                BINDGEN_EXTRA_CLANG_ARGS="$CUDA_BINDGEN_EXTRA_ARGS" \
                cargo run --example rivalry_leaderboard --release --features cuda
            ;;
        "compat_no_pkg")
            env LD_LIBRARY_PATH="$CUDA_RUNTIME_LD_LIBRARY_PATH" \
                CUDA_NO_PKG_CONFIG=1 BINDGEN_EXTRA_CLANG_ARGS="$CUDA_BINDGEN_EXTRA_ARGS" \
                cargo run --example rivalry_leaderboard --release --features cuda
            ;;
        *)
            return 1
            ;;
    esac
}

try_direct_cuda() {
    compute_bindgen_compat_args

    CUDA_BUILD_LOG="$(mktemp /tmp/fft_cuda_build.XXXX.log)"
    local modes=("plain" "compat" "compat_no_pkg")
    local mode
    local deps_install_attempted=false

    for mode in "${modes[@]}"; do
        print_status "$GREEN" "🎯 Attempting CUDA build (${mode})..."
        if run_cuda_build_mode "$mode" "$CUDA_BUILD_LOG"; then
            print_status "$GREEN" "✅ CUDA build successful (${mode})"
            if run_cuda_leaderboard_mode "$mode"; then
                return 0
            fi
            print_status "$YELLOW" "⚠️  CUDA runtime execution failed, falling back to standard mode."
            return 1
        fi

        classify_cuda_failure "$CUDA_BUILD_LOG"
        print_status "$YELLOW" "⚠️  CUDA build attempt failed (${mode}) [${CUDA_FAILURE_KIND}]"
        print_log_excerpt "$CUDA_BUILD_LOG"

        if [ "$deps_install_attempted" = false ] && \
            { [ "$CUDA_FAILURE_KIND" = "missing_limits" ] || [ "$CUDA_FAILURE_KIND" = "missing_cpp_headers" ]; }; then
            deps_install_attempted=true
            if try_install_cuda_build_dependencies; then
                print_status "$GREEN" "🔄 Retrying CUDA build after dependency install (${mode})..."
                if run_cuda_build_mode "$mode" "$CUDA_BUILD_LOG"; then
                    print_status "$GREEN" "✅ CUDA build successful after dependency install (${mode})"
                    if run_cuda_leaderboard_mode "$mode"; then
                        return 0
                    fi
                    print_status "$YELLOW" "⚠️  CUDA runtime execution failed, falling back to standard mode."
                    return 1
                fi
                classify_cuda_failure "$CUDA_BUILD_LOG"
                print_status "$YELLOW" "⚠️  CUDA retry still failed [${CUDA_FAILURE_KIND}]"
                print_log_excerpt "$CUDA_BUILD_LOG"
            fi
        fi

        if [ "$CUDA_FAILURE_KIND" = "bindgen_ident" ]; then
            show_troubleshooting "cuda_bindgen_incompatible"
            break
        fi
    done

    if [ "$CUDA_FAILURE_KIND" != "bindgen_ident" ]; then
        show_troubleshooting "cuda_build_fail"
    fi
    return 1
}

run_direct_standard() {
    print_status "$GREEN" "🎯 Building standard version..."
    if ! cargo build --release; then
        print_status "$RED" "❌ Standard build failed"
        return 1
    fi
    cargo run --example rivalry_leaderboard --release
}

run_direct() {
    if [ "$NVIDIA_GPU" = true ]; then
        if try_direct_cuda; then
            return 0
        fi
        print_status "$YELLOW" "🔄 Falling back to standard build..."
    fi

    if run_direct_standard; then
        return 0
    fi

    show_troubleshooting "wgpu_fail"
    return 1
}

probe_container_gpu() {
    local probe_log
    probe_log="$(mktemp /tmp/fft_container_gpu_probe.XXXX.log)"
    local probe_cmd=("bash" "-lc" "test -c /dev/nvidiactl || test -c /dev/nvidia0 || command -v nvidia-smi >/dev/null 2>&1")

    if "$CONTAINER_CLI" run --rm --gpus all "$GPU_PROBE_IMAGE" "${probe_cmd[@]}" >"$probe_log" 2>&1; then
        CONTAINER_GPU_WORKS=true
        CONTAINER_GPU_FLAGS=(--gpus all)
        print_status "$GREEN" "✅ Container GPU support detected (--gpus all)"
        rm -f "$probe_log"
        return 0
    fi

    if "$CONTAINER_CLI" run --rm --device nvidia.com/gpu=all "$GPU_PROBE_IMAGE" "${probe_cmd[@]}" >"$probe_log" 2>&1; then
        CONTAINER_GPU_WORKS=true
        CONTAINER_GPU_FLAGS=(--device nvidia.com/gpu=all)
        print_status "$GREEN" "✅ Container GPU support detected (--device nvidia.com/gpu=all)"
        rm -f "$probe_log"
        return 0
    fi

    CONTAINER_GPU_WORKS=false
    print_status "$YELLOW" "⚠️  Container GPU probe failed"
    print_status "$YELLOW" "   Probe error: $(head -n 2 "$probe_log" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"
    rm -f "$probe_log"
    return 1
}

ensure_container_runtime() {
    if command_exists docker; then
        local docker_info_log
        docker_info_log="$(mktemp /tmp/fft_docker_info.XXXX.log)"
        if docker info >"$docker_info_log" 2>&1; then
            CONTAINER_CLI="docker"
            if grep -qi "Emulate Docker CLI using podman" "$docker_info_log" || grep -qi "buildahVersion" "$docker_info_log"; then
                CONTAINER_RUNTIME="podman-via-docker"
                print_status "$GREEN" "✅ Docker CLI detected (Podman backend)"
            else
                CONTAINER_RUNTIME="docker"
                print_status "$GREEN" "✅ Docker detected and working"
            fi
            rm -f "$docker_info_log"
            probe_container_gpu || true
            return 0
        fi
        rm -f "$docker_info_log"
    fi

    if command_exists podman; then
        if podman info >/dev/null 2>&1; then
            CONTAINER_CLI="podman"
            CONTAINER_RUNTIME="podman"
            print_status "$GREEN" "✅ Podman detected and working"
            probe_container_gpu || true
            return 0
        fi
    fi

    CONTAINER_RUNTIME="none"
    return 1
}

build_container_image_if_needed() {
    if [ ! -f "$DOCKERFILE_PATH" ]; then
        print_status "$RED" "❌ ${DOCKERFILE_PATH} not found"
        return 1
    fi

    if "$CONTAINER_CLI" image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        print_status "$GREEN" "✅ Using existing container image: ${DOCKER_IMAGE}"
        return 0
    fi

    print_status "$YELLOW" "📦 Building container image..."
    "$CONTAINER_CLI" build -t "$DOCKER_IMAGE" -f "$DOCKERFILE_PATH" .
}

run_container_mode() {
    local use_gpu="$1"
    local run_args=(run --rm --name fft-leaderboard)

    if [ "${#DOCKER_TTY_FLAGS[@]}" -gt 0 ]; then
        run_args+=("${DOCKER_TTY_FLAGS[@]}")
    fi

    if [ "$use_gpu" = true ]; then
        run_args+=("${CONTAINER_GPU_FLAGS[@]}")
    fi

    "$CONTAINER_CLI" "${run_args[@]}" "$DOCKER_IMAGE"
}

run_docker_gpu() {
    if [ "$CONTAINER_GPU_WORKS" != true ]; then
        return 1
    fi

    print_status "$GREEN" "🚀 Running with container runtime and NVIDIA GPU..."
    if ! build_container_image_if_needed; then
        show_troubleshooting "docker_fail"
        return 1
    fi

    if run_container_mode true; then
        return 0
    fi

    show_troubleshooting "docker_fail"
    return 1
}

run_docker_cpu() {
    if [ "$CONTAINER_RUNTIME" = "none" ]; then
        return 1
    fi

    print_status "$GREEN" "🚀 Running with container runtime (CPU mode)..."
    if ! build_container_image_if_needed; then
        show_troubleshooting "docker_fail"
        return 1
    fi

    if run_container_mode false; then
        return 0
    fi

    show_troubleshooting "docker_fail"
    return 1
}

ensure_rust() {
    if command_exists cargo; then
        RUST_AVAILABLE=true
        return 0
    fi

    print_status "$YELLOW" "📦 Rust/Cargo not found. Installing Rust..."
    if ! command_exists curl; then
        print_status "$RED" "❌ curl not found, cannot auto-install Rust."
        return 1
    fi

    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env"

    if command_exists cargo; then
        RUST_AVAILABLE=true
        return 0
    fi
    return 1
}

# Check if we're running in a container
if [ -f /.dockerenv ]; then
    print_status "$YELLOW" "ℹ️  Running inside a container"
else
    print_status "$GREEN" "✅ Running on host system"
fi

print_section "Hardware Detection"

if command_exists nvidia-smi; then
    if NVIDIA_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"; then
        NVIDIA_GPU=true
        NVIDIA_SMI_WORKS=true
        print_status "$GREEN" "✅ NVIDIA GPU detected via nvidia-smi: ${NVIDIA_GPU_NAME}"
    else
        print_status "$YELLOW" "⚠️  nvidia-smi exists but failed to query GPU"
    fi
fi

if [ "$NVIDIA_GPU" = false ] && command_exists lspci; then
    if lspci | grep -qi nvidia; then
        NVIDIA_GPU=true
        NVIDIA_GPU_NAME="$(lspci | grep -i nvidia | head -1 | sed 's/.*: //')"
        print_status "$GREEN" "✅ NVIDIA GPU detected via lspci: ${NVIDIA_GPU_NAME}"
        if [ "$NVIDIA_SMI_WORKS" = false ]; then
            print_status "$YELLOW" "⚠️  Drivers may not be fully functional (nvidia-smi query failed)"
        fi
    fi
fi

if [ "$NVIDIA_GPU" = false ]; then
    print_status "$YELLOW" "ℹ️  No NVIDIA GPU detected"
    print_status "$YELLOW" "   cuFFT will be skipped; WebGPU implementations will run"
fi

print_section "Container Runtime"
print_status "$YELLOW" "ℹ️  Container execution is disabled; using host/direct execution only."

print_section "Rust Environment Check"
if ensure_rust; then
    print_status "$GREEN" "✅ Rust/Cargo installed: $(cargo --version)"
else
    print_status "$YELLOW" "⚠️  Rust/Cargo unavailable"
fi

print_section "Execution Strategy"

METHODS=()

if [ "$RUST_AVAILABLE" = true ]; then
    METHODS+=("direct")
else
    METHODS+=("install_rust")
    METHODS+=("direct")
fi

if [ "${#METHODS[@]}" -eq 0 ]; then
    print_status "$RED" "❌ No viable execution method found"
    exit 1
fi

print_status "$GREEN" "🎯 Planned method order: ${METHODS[*]}"

print_section "Executing FFT Rivalry Leaderboard"

for method in "${METHODS[@]}"; do
    case "$method" in
        "install_rust")
            if ensure_rust; then
                print_status "$GREEN" "✅ Rust installation step completed"
            else
                print_status "$YELLOW" "⚠️  Rust installation step failed, trying next method..."
            fi
            ;;
        "direct")
            if [ "$RUST_AVAILABLE" != true ]; then
                print_status "$YELLOW" "⚠️  Skipping direct method: Rust is unavailable"
                continue
            fi
            if run_direct; then
                RUN_SUCCESS=true
                break
            fi
            print_status "$YELLOW" "⚠️  Direct method failed, trying next method..."
            ;;
    esac
done

print_section "Completion"
if [ "$RUN_SUCCESS" = true ]; then
    print_status "$GREEN" "✅ FFT Rivalry Leaderboard script completed successfully"
    echo ""
    exit 0
fi

print_status "$RED" "❌ All execution methods failed"
echo ""
exit 1
