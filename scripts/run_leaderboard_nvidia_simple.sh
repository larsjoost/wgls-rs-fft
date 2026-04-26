#!/bin/bash

# Simple script to run FFT rivalry leaderboard in NVIDIA Docker container
# This script requires:
# - Docker installed
# - NVIDIA Container Toolkit installed
# - NVIDIA GPU with proper drivers

echo "🚀 Starting FFT Rivalry Leaderboard in NVIDIA Docker Container (Simple)"
echo "===================================================================="

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed. Please install Docker first."
    echo "   See: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if NVIDIA GPU is available
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: NVIDIA Container Toolkit may not be properly installed or no GPU detected."
    echo "   To install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo "   Continuing without GPU acceleration..."
    USE_GPU=""
else
    echo "✅ NVIDIA GPU detected - GPU acceleration enabled"
    USE_GPU="--gpus all"
fi

# Get current directory
PROJECT_DIR=$(pwd)
DOCKER_IMAGE="nvidia/cuda:11.8.0-devel-ubuntu22.04"

echo "🐳 Running in NVIDIA CUDA Docker container..."

docker run \
    $USE_GPU \
    --rm \
    -it \
    --name fft-leaderboard \
    -v "$PROJECT_DIR:/app" \
    -w "/app" \
    $DOCKER_IMAGE \
    bash -c "
    apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev cmake git && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    source /root/.cargo/env && \
    cargo build --release --features cuda && \
    cargo run --example rivalry_leaderboard --release --features cuda
    "

echo "🎉 FFT Rivalry Leaderboard completed!"
