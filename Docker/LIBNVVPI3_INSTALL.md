# libnvvpi3 Installation Guide

## Overview

`libnvvpi3` (NVIDIA Vision Programming Interface library) is a **required dependency** for DeepStream 6.4. It's now automatically installed in the Dockerfile, but this guide explains the options if you need to customize.

## Current Setup

The Dockerfile (`Dockerfile.realsense`) automatically installs `libnvvpi3` from NVIDIA's Jetson apt repository during build. This means:

- ✅ Every container built from this image will have `libnvvpi3`
- ✅ No manual installation needed at runtime
- ✅ DeepStream installation will work without dependency errors

## Installation Methods

### Option 1: Automatic (Current - Recommended)

The Dockerfile uses NVIDIA's Jetson apt repository:

```dockerfile
# Installs from: https://repo.download.nvidia.com/jetson/
# For JetPack 6.0 / L4T R36
```

**If this fails**, check:

- JetPack version matches (currently set to `r36` for JetPack 6.0)
- SOC type matches (currently `t234` for Orin, adjust if using Xavier/Nano)
- Network access to NVIDIA repos

### Option 2: Local .deb File

If automatic installation fails or you want to use a specific version:

1. **Get the .deb file:**

   - From Jetson SDK Manager installation
   - Or download from NVIDIA Developer site
   - Or copy from a Jetson device: `/var/cache/apt/archives/libnvvpi3_*.deb`

2. **Place it in the project:**

   ```bash
   # Copy your .deb file here
   cp /path/to/libnvvpi3_*.deb Docker/debs/libnvvpi3.deb
   ```

3. **Modify Dockerfile:**

   - Comment out Option 1 (lines 132-149)
   - Uncomment Option 2 (lines 151-156)

4. **Rebuild:**
   ```bash
   docker build -f Docker/Dockerfile.realsense -t cv-realsense:latest .
   ```

## Adjusting for Different Jetson Models

If you're using a different Jetson model, adjust the repository paths in the Dockerfile:

| Jetson Model | SOC Code | Example Path           |
| ------------ | -------- | ---------------------- |
| Orin         | t234     | `jetson/t234 r36 main` |
| Xavier       | t194     | `jetson/t194 r36 main` |
| Nano         | t186     | `jetson/t186 r36 main` |

Also adjust `r36` if using different JetPack version:

- JetPack 6.0 → `r36`
- JetPack 5.1 → `r35`
- JetPack 5.0 → `r35`

## Verifying Installation

After building the image, verify `libnvvpi3` is installed:

```bash
# Inside container
dpkg -l | grep libnvvpi3
# Should show: ii  libnvvpi3  X.X.X  arm64  NVIDIA Vision Programming Interface library

# Check version requirement
dpkg -l libnvvpi3 | grep -q ">=3.0.0" && echo "Version OK" || echo "Version check needed"
```

## Troubleshooting

**Error: "Failed to install libnvvpi3 from NVIDIA repo"**

1. Check your JetPack version: `cat /etc/nv_tegra_release`
2. Verify SOC type matches (Orin/Xavier/Nano)
3. Try Option 2 with local .deb file
4. Or install manually in container after build:
   ```bash
   # Add repo manually
   curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | \
       gpg --dearmor -o /usr/share/keyrings/jetson-ota-public.gpg
   echo "deb [signed-by=/usr/share/keyrings/jetson-ota-public.gpg] https://repo.download.nvidia.com/jetson/common r36 main" > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
   apt-get update && apt-get install -y libnvvpi3
   ```

## Notes

- `libnvvpi3` must be version >= 3.0.0 for DeepStream 6.4
- The package is Jetson-specific and won't work on x86_64 systems
- Once installed in the Docker image, it persists across all containers built from that image
