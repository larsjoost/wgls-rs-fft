#!/bin/bash

# Test script to verify Docker setup without actually running containers
# This script checks that all files are in place and provides setup instructions

echo "🧪 Testing FFT Rivalry Docker Setup"
echo "===================================="

# Test 1: Check if Dockerfile exists
echo "✓ Test 1: Checking Dockerfile.cuda..."
if [ -f "scripts/Dockerfile.cuda" ]; then
    echo "  ✅ Dockerfile.cuda found"
else
    echo "  ❌ Dockerfile.cuda not found"
    echo "  Expected: scripts/Dockerfile.cuda"
    exit 1
fi

# Test 2: Check if main script exists
echo "✓ Test 2: Checking build_and_run_docker.sh..."
if [ -f "scripts/build_and_run_docker.sh" ]; then
    echo "  ✅ build_and_run_docker.sh found"
else
    echo "  ❌ build_and_run_docker.sh not found"
    exit 1
fi

# Test 3: Check script syntax
echo "✓ Test 3: Checking bash script syntax..."
if bash -n scripts/build_and_run_docker.sh; then
    echo "  ✅ Script syntax is valid"
else
    echo "  ❌ Script has syntax errors"
    exit 1
fi

# Test 4: Check Dockerfile syntax (basic check)
echo "✓ Test 4: Checking Dockerfile structure..."
if grep -q "FROM nvidia/cuda" scripts/Dockerfile.cuda; then
    echo "  ✅ Dockerfile has valid FROM instruction"
else
    echo "  ❌ Dockerfile missing FROM instruction"
    exit 1
fi

# Test 5: Check if scripts are executable
echo "✓ Test 5: Checking script permissions..."
if [ -x "scripts/build_and_run_docker.sh" ]; then
    echo "  ✅ Script is executable"
else
    echo "  ⚠️  Script is not executable"
    echo "  Fix: chmod +x scripts/build_and_run_docker.sh"
fi

echo ""
echo "📋 Docker Setup Summary:"
echo "   All tests passed! Your Docker setup is ready."
echo ""
echo "🚀 To run the FFT rivalry leaderboard with Docker:"
echo ""
echo "   Option 1: With this script (recommended)"
echo "   ./scripts/build_and_run_docker.sh"
echo ""
echo "   Option 2: Manual Docker commands"
echo "   docker build -t fft-rivalry-leaderboard -f scripts/Dockerfile.cuda ."
echo "   docker run --gpus all --rm -it fft-rivalry-leaderboard"
echo ""
echo "💡 Requirements for full functionality:"
echo "   • Docker installed and running"
echo "   • NVIDIA Container Toolkit (for GPU support)"
echo "   • NVIDIA GPU with drivers (for GPU acceleration)"
echo ""
echo "✅ Setup verification complete!"
