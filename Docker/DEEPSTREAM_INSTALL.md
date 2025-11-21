# DeepStream Installation Guide

## Problem

DeepStream installation often fails during Docker build because:
- NGC (NVIDIA GPU Cloud) requires authentication
- Network issues during build
- The download URL may change

## Solution

The Dockerfile now makes DeepStream installation **optional** during build. The image will build successfully even if DeepStream isn't installed.

## Install DeepStream After Build

### Option 1: Use the Installation Script (Recommended)

1. Build the Docker image (it will succeed even without DeepStream):
   ```bash
   docker build -f Docker/Dockerfile.realsense -t cv-realsense:latest .
   ```

2. Run the container:
   ```bash
   ./services/run-deepstream.sh
   ```

3. Inside the container, install DeepStream:
   ```bash
   ./yolov8_ros/scripts/install_deepstream.sh
   ```

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

### Handling `libnvvpi3`

`libnvvpi3` ships with the Jetson Multimedia/VPI SDK and is not always
available in standard apt repositories. The installer now handles it
automatically, but you can control the behavior:

```bash
# Skip libnvvpi3 entirely (if you only need basic DeepStream features)
./yolov8_ros/scripts/install_deepstream.sh --skip-libnvvpi

# Provide a local .deb downloaded from NVIDIA SDK Manager / Jetson BSP
./yolov8_ros/scripts/install_deepstream.sh --libnvvpi-deb /tmp/libnvvpi3.deb
```

If the package is still missing:
- Download the Jetson Multimedia API / VPI package matching your Jetson release
- Install the `libnvvpi3` .deb manually (`dpkg -i libnvvpi3_*.deb && apt-get -y -f install`)
- Re-run the DeepStream installer

4. **Verify installation:**
   ```bash
   ./yolov8_ros/scripts/check_deepstream.sh
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

- DeepStream must be installed **inside the container**, not on the host
- The installation script handles environment variables automatically
- If you rebuild the image, you'll need to reinstall DeepStream (or save it in a volume)

