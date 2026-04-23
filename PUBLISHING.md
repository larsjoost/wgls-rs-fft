# Publishing Status and Guidelines

## Current Publishing Status

❌ **This crate cannot currently be published to crates.io** due to a dependency on a specific git revision of `wgsl-rs`.

## Reason for Publishing Block

The project depends on a specific git commit of [`wgsl-rs`](https://github.com/schell/wgsl-rs) that uses a different API than the published version on crates.io:

```toml
# Current dependency (cannot be published)
wgsl-rs = { git = "https://github.com/schell/wgsl-rs", rev = "07175c49dfc231f309c5fc6a4b86750d00dfd7cc" }

# Published version (different API, incompatible)
wgsl-rs = "0.0.0-reserved"  # Placeholder with different API
```

## Options to Make Publishable

### Option 1: Fork and Publish Compatible Version (Recommended)

1. **Fork the wgsl-rs repository** at the specific commit
2. **Publish it to crates.io** with a compatible API
3. **Update this crate's dependency** to use the published version
4. **Publish this crate**

**Pros:**
- Maintains current functionality
- Proper dependency management
- Can receive updates

**Cons:**
- Requires maintaining a fork
- Need to coordinate with upstream

### Option 2: Adapt Code to Published API

1. **Update shader code** to work with published `wgsl-rs` API
2. **Test compatibility**
3. **Publish this crate**

**Pros:**
- Uses official published dependencies
- No fork maintenance

**Cons:**
- Significant code changes required
- May lose some functionality
- API differences may be substantial

### Option 3: Vendor the Dependency

1. **Copy wgsl-rs source code** into this project
2. **Update Cargo.toml** to use local path dependency
3. **Publish this crate** with vendored dependency

**Pros:**
- Self-contained project
- No external dependencies

**Cons:**
- Larger crate size
- Harder to update wgsl-rs
- Duplicate code maintenance

### Option 4: Keep as Git-Only Project (Current Status)

1. **Document the git dependency requirement**
2. **Use directly from GitHub**
3. **Don't publish to crates.io**

**Pros:**
- No changes required
- Works with current setup

**Cons:**
- Users must clone from GitHub
- Harder dependency management
- No crates.io benefits

## Current Workaround

If you need to use this crate now:

```toml
[dependencies]
wgls-rs-fft = { git = "https://github.com/larsjoost/wgls-rs-fft" }
```

## Development Workflow

Despite the publishing limitation, you can still:

1. **Develop locally**: All tests pass
2. **Run CI locally**: `./scripts/ci_test.sh`
3. **Use in projects**: Via git dependency
4. **Contribute**: Submit PRs to the GitHub repository

## Future Publishing Plan

When resources allow, the recommended approach is:

1. **Fork wgsl-rs** at commit `07175c49dfc231f309c5fc6a4b86750d00dfd7cc`
2. **Publish as `wgsl-rs-compat`** or similar name
3. **Update this crate** to use the published compatible version
4. **Publish this crate** to crates.io

## Checking Publishing Readiness

To check if this crate can be published:

```bash
cargo publish --dry-run
```

This will show the current publishing blockers.

## Alternative Distribution

If publishing to crates.io is not possible, consider:

1. **GitHub Releases**: Package and release binaries
2. **Container Images**: Docker images with pre-built binaries
3. **Pre-built Binaries**: Release compiled binaries for major platforms
4. **Private Crate Registry**: Host on a private cargo registry

## Contact

For questions about publishing or to discuss alternatives:
- Open an issue on GitHub
- Contact the maintainer

## License

This project is licensed under MIT, so forking and republishing is permitted under the license terms.