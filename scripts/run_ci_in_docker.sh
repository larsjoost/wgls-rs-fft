#!/bin/bash

# Script to run CI tests in Docker container
# This ensures consistent environment between local and GitHub CI

set -e  # Exit immediately if any command fails

echo "🐳 Running CI tests in Docker container"
echo "========================================"

# Build the Docker image
section() {
    echo ""
    echo "📋 $1"
    echo "------------------------------------------"
}

section "Building Docker image"
echo "Running: docker build -t wgls-rs-fft-ci -f Dockerfile.ci ."
docker build -t wgls-rs-fft-ci -f Dockerfile.ci .

section "Running CI tests in container"
echo "Running: docker run --rm -v $(pwd):/workspace -w /workspace wgls-rs-fft-ci ./scripts/ci_test.sh"

# Run the CI test script inside the container
# Mount current directory as /workspace to preserve test results
docker run --rm \
    -v $(pwd):/workspace \
    -w /workspace \
    -e RUST_BACKTRACE=1 \
    -e CARGO_TERM_COLOR=always \
    wgls-rs-fft-ci \
    ./scripts/ci_test.sh

echo ""
echo "🎉 Docker CI test completed successfully"
echo "   Results are in your local directory"

exit 0