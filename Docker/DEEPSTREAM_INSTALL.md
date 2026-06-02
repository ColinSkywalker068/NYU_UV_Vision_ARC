# DeepStream Installation Guide

## Problem

DeepStream installation often fails during Docker build because:

- NGC (NVIDIA GPU Cloud) requires authentication
- Network issues during build
- Missing `libnvvpi3` dependency (NVIDIA VPI library)

## Solution

Use the dedicated C++ image:

1. `Docker/Dockerfile.cplusplus` bakes DeepStream into a separate image.
2. The original Python image stays unchanged for the old hybrid flow.
3. `libnvvpi3` is installed during the C++ image build so DeepStream can load cleanly.

## Install DeepStream After Build

### Option 1: Use the Installation Script (Recommended)

1. Build the C++ Docker image:

   ```bash
   docker build -f Docker/Dockerfile.cplusplus -t cv-cplusplus:latest .
   ```

2. Run the container:

   ```bash
   ./services/run-cplusplus.sh
   ```

3. DeepStream is already installed in the image. If you are testing a custom base image, you can still install it manually after the fact.

### Option 2: Manual Installation

If the script fails, download DeepStream manually:

1. **Download from NVIDIA NGC:**

   - Go to: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream
   - Or search for: `deepstream-6.4_6.4.0-1_arm64.deb`
   - Download the `.deb` file

2. **Copy into running container:**

   ```bash
   # Find container name/ID
   docker ps

   # Copy file
   docker cp deepstream-6.4_6.4.0-1_arm64.deb <container>:/tmp/
   ```

3. **Install inside container:**
   ```bash
   # Inside container
   dpkg -i /tmp/deepstream-6.4_6.4.0-1_arm64.deb
   apt-get update
   apt-get -y -f install
   ```

### About `libnvvpi3`

**Good news:** `libnvvpi3` is automatically installed in the C++ Dockerfile from NVIDIA's Jetson apt repository. This means:

- ✅ No manual installation needed
- ✅ Available in every container built from this image
- ✅ DeepStream installation should work without dependency errors

**If the automatic installation fails** (wrong JetPack version, network issues):

1. Rebuild from a Jetson image that can reach NVIDIA's apt repository.
2. Or install manually in a running container: `apt-get install -y libnvvpi3` (if NVIDIA repo is accessible)

3. **Verify installation:**
   ```bash
   gst-inspect-1.0 nvstreammux
   ```

## Verify DeepStream is Working

After installation, verify:

```bash
# Check if directory exists
ls /opt/nvidia/deepstream/deepstream

# Check if plugins exist
ls /opt/nvidia/deepstream/deepstream/lib/gst-plugins/

# Test GStreamer can find plugins
gst-inspect-1.0 nvstreammux
```

## Notes

- DeepStream is baked into the C++ image by default
- DeepStream still must be installed inside the container if you are testing a custom base image
