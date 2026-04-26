#!/bin/bash

# Script to build and run FFT rivalry leaderboard in Docker with CUDA support
# This script uses the dedicated Dockerfile.cuda for better maintainability

echo "🚀 Building and Running FFT Rivalry Leaderboard with CUDA Docker"
echo "=================================================================="

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed. Please install Docker first."
    echo "   See: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Dockerfile.cuda exists
if [ ! -f "scripts/Dockerfile.cuda" ]; then
    echo "❌ Error: Dockerfile.cuda not found in scripts/ directory"
    echo "   Expected location: scripts/Dockerfile.cuda"
    exit 1
fi

# Check if NVIDIA GPU is available
USE_GPU=""
if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected - GPU acceleration will be enabled"
    USE_GPU="--gpus all"
else
    echo "⚠️  No NVIDIA GPU detected - running without GPU acceleration"
    echo "   To enable GPU support, install NVIDIA Container Toolkit"
fi

echo "🐳 Building Docker image with CUDA support..."
echo "   This may take a while on first run (downloads ~5GB image)"

docker build -t fft-rivalry-leaderboard -f scripts/Dockerfile.cuda .

if [ $? -ne 0 ]; then
    echo "❌ Failed to build Docker image"
    echo "   Check that all project files are present and Docker is working"
    exit 1
fi

echo "✅ Docker image built successfully"
echo "🚀 Running FFT rivalry leaderboard..."

docker run \
    $USE_GPU \
    --rm \
    -it \
    --name fft-leaderboard \
    fft-rivalry-leaderboard

echo "🎉 FFT Rivalry Leaderboard completed!"

# Show Docker cleanup instructions
echo ""
echo "📋 Docker Cleanup (optional):"
echo "   To remove the built image and free up space (~5GB):"
echo "   docker rmi fft-rivalry-leaderboard"
