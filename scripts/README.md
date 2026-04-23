# CI Test Scripts

This directory contains scripts for running the complete CI test suite locally.

## `ci_test.sh`

The main CI test script that runs all the same tests that GitHub CI would run.

### Usage

```bash
./scripts/ci_test.sh
```

### What it does

1. **Checks required tools** (cargo, rustc)
2. **Checks system dependencies** (pkg-config, GPU libraries)
3. **Runs cargo tests** with single threading to avoid GPU resource contention
4. **Checks code formatting** with `cargo fmt --check`
5. **Runs clippy analysis** with expected shader warnings
6. **Runs documentation tests** with `cargo test --doc`
7. **Reports timing and results**

### Features

- **Identical to GitHub CI**: Runs the exact same tests as the GitHub Actions workflow
- **Local testing**: Catch all CI issues before pushing to GitHub
- **Detailed output**: Clear section headers and progress reporting
- **Error handling**: Proper exit codes and error messages
- **Dependency checking**: Warns about missing system dependencies
- **Timing**: Shows total duration of the test suite

### Exit Codes

- `0`: All tests passed successfully
- `1`: Required tools missing or formatting issues
- Other: Test failures with specific exit codes

### System Requirements

For full functionality, ensure you have these system dependencies installed:

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y \
  pkg-config libx11-dev libasound2-dev libudev-dev \
  libwayland-dev libxkbcommon-dev
```

**Fedora:**
```bash
sudo dnf install -y \
  pkg-config libX11-devel alsa-lib-devel systemd-devel \
  wayland-devel libxkbcommon-devel
```

### Running in CI

The GitHub Actions workflow has been updated to use this script. See `.github/workflows/ci.yml` for details.

### Development Workflow

1. Make your changes
2. Run `./scripts/ci_test.sh` locally
3. Fix any issues
4. Commit and push
5. GitHub CI will run the same tests

This ensures you catch all issues locally before they reach CI!

## GitHub CI Integration

The CI workflow now uses this script to ensure consistency between local and CI testing. See the workflow file for details on how it's integrated.

## Troubleshooting

If tests fail:
1. Check the specific error messages
2. Run individual test commands to isolate issues
3. Check system dependencies
4. Try running with `RUST_BACKTRACE=1` for more details

Example:
```bash
RUST_BACKTRACE=1 cargo test --verbose -- --test-threads=1
```