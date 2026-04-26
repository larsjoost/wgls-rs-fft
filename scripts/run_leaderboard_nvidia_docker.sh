#!/bin/bash

# Script to run FFT rivalry leaderboard in NVIDIA Docker container
# This script requires:
# - Docker installed
# - NVIDIA Container Toolkit installed
# - NVIDIA GPU with proper drivers

echo "🚀 Starting FFT Rivalry Leaderboard in NVIDIA Docker Container"
echo "============================================================"

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed. Please install Docker first."
    echo "   See: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if nvidia-container-toolkit is available
if ! docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: NVIDIA Container Toolkit may not be properly installed."
    echo "   To install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo "   Continuing without GPU acceleration..."
    USE_GPU="--gpus all"
else
    echo "✅ NVIDIA Container Toolkit detected - GPU acceleration enabled"
    USE_GPU="--gpus all"
fi

# Build the project first
 echo "📦 Building project with CUDA support..."
 if ! cargo build --release --features cuda; then
     echo "❌ Failed to build with CUDA features. Continuing with CPU-only build..."
     cargo build --release
 fi

# Determine the current project directory
PROJECT_DIR=$(pwd)
DOCKER_IMAGE="nvidia/cuda:11.8.0-devel-ubuntu22.04"

 echo "🐳 Building Docker container with project files..."

# Create Dockerfile
temp_dir=$(mktemp -d)
cat > "$temp_dir/Dockerfile" << EOF
FROM $DOCKER_IMAGE

# Install Rust and dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH "/root/.cargo/bin:${PATH}"

# Install CUDA dependencies
RUN apt-get update && apt-get install -y \
    cuda-toolkit-11-8 \
    cuda-libraries-dev-11-8 \
    && rm -rf /var/lib/apt/lists/*

# Set up environment variables
ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH /usr/local/cuda/bin:${PATH}

WORKDIR /app
COPY . .

# Build the project
RUN cargo build --release --features cuda
EOF

# Copy project files to temp directory
cp -r * "$temp_dir/"

 echo "🔧 Building Docker image..."
 docker build -t fft-rivalry-leaderboard -f "$temp_dir/Dockerfile" "$temp_dir"

 if [ $? -ne 0 ]; then
     echo "❌ Failed to build Docker image"
     rm -rf "$temp_dir"
     exit 1
 fi

echo "✅ Docker image built successfully"

 echo "🚀 Running FFT rivalry leaderboard in container..."

docker run \
    $USE_GPU \
    --rm \
    -it \
    --name fft-leaderboard \
    -v "$PROJECT_DIR/target:/app/target" \
    fft-rivalry-leaderboard \
    cargo run --example rivalry_leaderboard --release --features cuda

# Clean up
rm -rf "$temp_dir"

echo "🎉 FFT Rivalry Leaderboard completed!"
