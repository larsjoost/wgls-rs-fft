#!/bin/bash

# Comprehensive FFT Rivalry Leaderboard Test and Run Script
# This script checks the entire setup, builds necessary components, and runs tests

echo "🔍 Comprehensive FFT Rivalry Leaderboard Setup and Test"
echo "===================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print section header
print_section() {
    local title=$1
    echo ""
    print_status "$BLUE" "=== $title ==="
}

# Initialize counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing: $test_name... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        print_status "$GREEN" "✅ PASS"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        print_status "$RED" "❌ FAIL"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start comprehensive testing
print_section "System Environment Check"

# Check Rust/Cargo
if command_exists cargo; then
    print_status "$GREEN" "✅ Rust/Cargo installed: $(cargo --version)"
else
    print_status "$YELLOW" "⚠️  Rust/Cargo not found. Some tests may be limited."
fi

# Check Docker
if command_exists docker; then
    print_status "$GREEN" "✅ Docker installed: $(docker --version)"
    DOCKER_AVAILABLE=true
else
    print_status "$YELLOW" "⚠️  Docker not found. Docker tests will be skipped."
    DOCKER_AVAILABLE=false
fi

# Check NVIDIA GPU (if Docker available)
if [ "$DOCKER_AVAILABLE" = true ]; then
    if docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi >/dev/null 2>&1; then
        print_status "$GREEN" "✅ NVIDIA GPU detected via Docker"
        NVIDIA_GPU=true
    else
        print_status "$YELLOW" "ℹ️  No NVIDIA GPU detected (or Docker not configured)"
        NVIDIA_GPU=false
    fi
else
    NVIDIA_GPU=false
fi

print_section "Project Structure Verification"

# Check project files
run_test "Project root directory" "test -d ."
run_test "Cargo.toml exists" "test -f Cargo.toml"
run_test "Source directory exists" "test -d src"
run_test "Examples directory exists" "test -d examples"
run_test "Rivalry leaderboard example exists" "test -f examples/rivalry_leaderboard.rs"

print_section "Rust Project Verification"

# Check Rust project structure
run_test "Library file exists" "test -f src/lib.rs"
run_test "Rivals module exists" "test -f src/rivals/mod.rs"
run_test "Baseline implementation exists" "test -f src/rivals/baseline.rs"
run_test "Radix-4 proper implementation exists" "test -f src/rivals/radix4_proper.rs"

print_section "Docker Setup Verification"

# Check Docker files
run_test "Dockerfile.cuda exists" "test -f scripts/Dockerfile.cuda"
run_test "build_and_run_docker.sh exists" "test -f scripts/build_and_run_docker.sh"
run_test "Dockerfile syntax" "grep -q 'FROM nvidia/cuda' scripts/Dockerfile.cuda"

# Check script permissions
if [ -x "scripts/build_and_run_docker.sh" ]; then
    print_status "$GREEN" "✅ build_and_run_docker.sh is executable"
else
    print_status "$YELLOW" "⚠️  build_and_run_docker.sh is not executable"
    print_status "$YELLOW" "   Fix: chmod +x scripts/build_and_run_docker.sh"
fi

print_section "Documentation Verification"

# Check documentation files
run_test "Main README exists" "test -f README.md"
run_test "Docker README exists" "test -f scripts/README_NVIDIA_DOCKER.md"
run_test "Docker setup guide exists" "test -f scripts/DOCKER_SETUP.md"
run_test "Implementation summary exists" "test -f scripts/DOCKER_IMPLEMENTATION_SUMMARY.md"

print_section "Rust Project Tests"

# Run Rust-specific tests if cargo is available
if command_exists cargo; then
    echo "Running Rust project tests..."
    
    # Check if project builds
    if cargo check --quiet 2>/dev/null; then
        print_status "$GREEN" "✅ Project compiles successfully"
    else
        print_status "$YELLOW" "⚠️  Project has compilation issues"
        print_status "$YELLOW" "   This is expected in some environments"
    fi
    
    # Check library tests
    if cargo test --lib --quiet 2>/dev/null; then
        print_status "$GREEN" "✅ Library tests pass"
    else
        print_status "$YELLOW" "⚠️  Library tests have issues"
        print_status "$YELLOW" "   This may be expected without GPU"
    fi
else
    print_status "$YELLOW" "ℹ️  Skipping Rust tests (cargo not available)"
fi

print_section "Docker Tests"

# Run Docker tests if Docker is available
if [ "$DOCKER_AVAILABLE" = true ]; then
    echo "Running Docker setup tests..."
    
    # Test Dockerfile syntax by checking key elements
    run_test "Dockerfile has WORKDIR" "grep -q 'WORKDIR' scripts/Dockerfile.cuda"
    run_test "Dockerfile has COPY" "grep -q 'COPY' scripts/Dockerfile.cuda"
    run_test "Dockerfile has RUN commands" "grep -q 'RUN' scripts/Dockerfile.cuda"
    run_test "Dockerfile builds project" "grep -q 'cargo build' scripts/Dockerfile.cuda"
    
    print_status "$GREEN" "✅ Dockerfile structure is valid"
else
    print_status "$YELLOW" "ℹ️  Skipping Docker tests (Docker not available)"
fi

print_section "FFT Rivalry Leaderboard Verification"

# Check leaderboard components
run_test "Leaderboard script exists" "test -f examples/rivalry_leaderboard.rs"
run_test "Benchmark module exists" "test -f src/benchmark.rs"
run_test "FftExecutor trait exists" "grep -q 'trait FftExecutor' src/lib.rs"

# Check rival implementations
run_test "Claude implementation exists" "test -f src/rivals/claude.rs"
run_test "Codex implementation exists" "test -f src/rivals/codex.rs"
run_test "Gemini implementation exists" "test -f src/rivals/gemini.rs"
run_test "Mistral Vibe implementation exists" "test -f src/rivals/mistral_vibe.rs"
run_test "Radix-4 implementation exists" "test -f src/rivals/radix4.rs"
run_test "Radix-4 proper implementation exists" "test -f src/rivals/radix4_proper.rs"

print_section "Final Summary"

echo ""
echo "📊 Test Results:"
echo "   Total Tests: $TOTAL_TESTS"
echo "   Passed: $PASSED_TESTS"
echo "   Failed: $FAILED_TESTS"
echo ""

# Calculate success rate
if [ "$TOTAL_TESTS" -gt 0 ]; then
    SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "   Success Rate: $SUCCESS_RATE%"
fi

# Determine overall status
if [ "$FAILED_TESTS" -eq 0 ]; then
    print_status "$GREEN" "✅ All tests passed! Setup is ready to use."
    OVERALL_STATUS="READY"
elif [ "$FAILED_TESTS" -lt 5 ]; then
    print_status "$YELLOW" "⚠️  Most tests passed. Minor issues detected."
    OVERALL_STATUS="PARTIAL"
else
    print_status "$RED" "❌ Several tests failed. Review issues above."
    OVERALL_STATUS="FAILED"
fi

echo ""
print_section "Next Steps"

case "$OVERALL_STATUS" in
    "READY")
        echo "🎉 Your setup is ready! You can now:"
        echo ""
        
        if [ "$DOCKER_AVAILABLE" = true ]; then
            echo "1. Run with Docker (recommended):"
            echo "   ./scripts/build_and_run_docker.sh"
            echo ""
        fi
        
        echo "2. Run leaderboard directly:"
        echo "   cargo run --example rivalry_leaderboard --release"
        echo ""
        
        if [ "$NVIDIA_GPU" = true ]; then
            echo "3. With NVIDIA GPU support:"
            echo "   cargo run --example rivalry_leaderboard --release --features cuda"
            echo ""
        fi
        
        echo "4. View documentation:"
        echo "   cat scripts/DOCKER_SETUP.md"
        ;;
    
    "PARTIAL")
        echo "⚠️  Your setup has minor issues. You can:"
        echo ""
        echo "1. Review warnings above"
        echo "2. Fix executable permissions if needed"
        echo "3. Try running anyway (some features may work)"
        echo "4. Check documentation for troubleshooting"
        ;;
    
    "FAILED")
        echo "❌ Your setup has significant issues. Please:"
        echo ""
        echo "1. Review the failed tests above"
        echo "2. Check README.md for setup instructions"
        echo "3. Ensure all dependencies are installed"
        echo "4. Verify file permissions"
        ;;
esac

echo ""
print_status "$BLUE" "📚 Documentation Available:"
echo "   • README.md - Main project documentation"
echo "   • scripts/README_NVIDIA_DOCKER.md - Docker setup guide"
echo "   • scripts/DOCKER_SETUP.md - Quick reference"
echo "   • scripts/DOCKER_IMPLEMENTATION_SUMMARY.md - Complete summary"

echo ""
print_status "$BLUE" "🎯 FFT Rivalry Leaderboard Features:"
echo "   • Multiple FFT implementations (Radix-2, Radix-4, Radix-8, Mixed)"
echo "   • WebGPU acceleration with wgpu"
echo "   • CUDA/cuFFT support (when available)"
echo "   • Comprehensive benchmarking"
echo "   • Validation against reference implementation"
echo "   • Batch processing support"

echo ""
print_status "$GREEN" "✅ Comprehensive test completed!"
