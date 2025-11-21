#!/usr/bin/env bash
# Manual DeepStream installation script for containers where build failed
# This can be run inside a running container

set -e

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --skip-libnvvpi           Skip libnvvpi3 installation (continue even if missing)
  --libnvvpi-deb <path>     Provide local libnvvpi3 .deb file for installation
  -h, --help                Show this message
EOF
}

SKIP_LIBNVVPI=${SKIP_LIBNVVPI:-0}
LIBNVVPI_DEB_PATH=${LIBNVVPI_DEB_PATH:-""}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-libnvvpi)
            SKIP_LIBNVVPI=1
            shift
            ;;
        --libnvvpi-deb)
            shift
            if [[ -z "${1:-}" ]]; then
                echo "ERROR: --libnvvpi-deb requires a file path"
                exit 1
            fi
            LIBNVVPI_DEB_PATH="$1"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

install_libnvvpi3() {
    if dpkg -s libnvvpi3 >/dev/null 2>&1; then
        echo "libnvvpi3 already installed."
        return 0
    fi

    echo "Attempting to install libnvvpi3 via apt..."
    if apt-get install -y libnvvpi3 >/dev/null 2>&1; then
        echo "✓ libnvvpi3 installed via apt."
        return 0
    fi

    if [[ -n "$LIBNVVPI_DEB_PATH" && -f "$LIBNVVPI_DEB_PATH" ]]; then
        echo "Installing libnvvpi3 from provided .deb: $LIBNVVPI_DEB_PATH"
        if dpkg -i "$LIBNVVPI_DEB_PATH"; then
            apt-get -y -f install || true
            if dpkg -s libnvvpi3 >/dev/null 2>&1; then
                echo "✓ libnvvpi3 installed from provided .deb."
                return 0
            fi
        fi
    fi

    echo "⚠ libnvvpi3 not found in apt repositories."
    echo "  This package ships with the Jetson Multimedia/VPI SDK."
    echo "  Download from https://developer.nvidia.com/embedded/jetson-linux or use NVIDIA SDK Manager."

    if [[ "$SKIP_LIBNVVPI" -eq 1 ]]; then
        echo "Skipping libnvvpi3 installation (per --skip-libnvvpi)."
        return 1
    fi

    read -p "Continue without libnvvpi3? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting until libnvvpi3 is provided."
        exit 1
    fi

    echo "Continuing without libnvvpi3. Some DeepStream features may be unavailable."
    return 1
}

echo "=== DeepStream 6.4 Manual Installation ==="
echo ""

# Check if already installed
if [ -d "/opt/nvidia/deepstream/deepstream" ]; then
    echo "DeepStream appears to already be installed at /opt/nvidia/deepstream/deepstream"
    echo "Checking plugins..."
    if [ -f "/opt/nvidia/deepstream/deepstream/lib/gst-plugins/libgstnvstreammux.so" ]; then
        echo "✓ DeepStream plugins found. Installation looks good!"
        exit 0
    else
        echo "⚠ DeepStream directory exists but plugins not found. Reinstalling..."
    fi
fi

# Check architecture
ARCH=$(dpkg --print-architecture)
if [ "$ARCH" != "arm64" ]; then
    echo "WARNING: This script is for arm64 (Jetson). Your architecture is: $ARCH"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Download DeepStream
echo "Downloading DeepStream 6.4..."
DEEPSTREAM_DEB="/tmp/deepstream-6.4_6.4.0-1_arm64.deb"

# Note: This may require NGC authentication
# If download fails, you may need to:
# 1. Get the .deb file manually from NGC
# 2. Copy it into the container
# 3. Run: dpkg -i /path/to/deepstream-6.4_6.4.0-1_arm64.deb

# Try multiple download methods
echo "Attempting to download DeepStream 6.4..."
DEEPSTREAM_DEB="/tmp/deepstream-6.4_6.4.0-1_arm64.deb"

# Method 1: Direct API download (may require authentication)
if curl -L --fail --silent --show-error \
    'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/6.4/files?redirect=true&path=deepstream-6.4_6.4.0-1_arm64.deb' \
    -o "$DEEPSTREAM_DEB" 2>&1; then
    echo "✓ Downloaded via API"
elif [ -f "$DEEPSTREAM_DEB" ] && [ -s "$DEEPSTREAM_DEB" ]; then
    echo "✓ Found existing download file"
else
    echo "✗ Failed to download DeepStream. Possible reasons:"
    echo "  1. Network connectivity issue"
    echo "  2. NGC authentication required"
    echo "  3. URL changed"
    echo ""
    echo "SOLUTION: Download manually from NVIDIA NGC:"
    echo "  1. Go to: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream"
    echo "  2. Or search for 'deepstream-6.4_6.4.0-1_arm64.deb'"
    echo "  3. Download the .deb file"
    echo "  4. Copy into container:"
    echo "     docker cp deepstream-6.4_6.4.0-1_arm64.deb <container>:/tmp/"
    echo "  5. Then run: dpkg -i /tmp/deepstream-6.4_6.4.0-1_arm64.deb && apt-get -y -f install"
    echo ""
    read -p "Do you have the .deb file ready to copy? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Please copy the file to /tmp/deepstream-6.4_6.4.0-1_arm64.deb and run this script again."
    fi
    exit 1
fi

# Install dependencies first (required by DeepStream)
echo "Installing DeepStream dependencies..."
apt-get update

# Install GStreamer development packages (required by DeepStream)
echo "Installing GStreamer development packages..."
apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    || {
        echo "ERROR: Failed to install GStreamer dev packages. These are required."
        exit 1
    }

# Install DeepStream
echo "Installing DeepStream..."
if ! dpkg -i "$DEEPSTREAM_DEB"; then
    echo "Fixing remaining dependencies..."
    # Try to fix dependencies
    if ! apt-get -y -f install; then
        echo "Attempting to install with dependency resolution..."
        # Try installing missing packages that might be available
        apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev || true
        
        # Try to install libnvvpi3 if available (Jetson SDK package)
        install_libnvvpi3 || true
        
        # Try installing DeepStream with --force-depends if regular install fails
        echo "Attempting DeepStream installation with --force-depends..."
        if ! dpkg -i --force-depends "$DEEPSTREAM_DEB"; then
            echo "ERROR: DeepStream installation failed even with --force-depends"
            echo "You may need to install libnvvpi3 manually from Jetson SDK (VPI package)."
            exit 1
        fi
        # Fix broken dependencies after force install
        apt-get -y -f install || true
    fi
fi

# Verify installation
echo ""
echo "Verifying installation..."
if [ -d "/opt/nvidia/deepstream/deepstream" ]; then
    echo "✓ DeepStream directory created"
    
    PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins"
    if [ -d "$PLUGIN_PATH" ]; then
        echo "✓ Plugin directory found"
        
        # Check key plugins
        MISSING=0
        for plugin in libgstnvstreammux.so libgstnvinfer.so libgstnvvideoconvert.so libgstnvdsosd.so; do
            if [ -f "$PLUGIN_PATH/$plugin" ]; then
                echo "  ✓ $plugin"
            else
                echo "  ✗ $plugin (MISSING)"
                MISSING=1
            fi
        done
        
        if [ $MISSING -eq 0 ]; then
            echo ""
            echo "✓ DeepStream installation successful!"
            echo ""
            echo "Setting environment variables..."
            export GST_PLUGIN_PATH="$PLUGIN_PATH"
            export GST_PLUGIN_PATH_1_0="$PLUGIN_PATH"
            export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH}"
            echo ""
            echo "To make these permanent, add to ~/.bashrc:"
            echo "export GST_PLUGIN_PATH=$PLUGIN_PATH"
            echo "export GST_PLUGIN_PATH_1_0=$PLUGIN_PATH"
            echo "export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:\$LD_LIBRARY_PATH"
        else
            echo "⚠ Some plugins are missing. Installation may be incomplete."
            exit 1
        fi
    else
        echo "✗ Plugin directory not found: $PLUGIN_PATH"
        exit 1
    fi
else
    echo "✗ DeepStream directory not found after installation"
    exit 1
fi

# Cleanup
rm -f "$DEEPSTREAM_DEB"

echo ""
echo "Installation complete! You can now run the GStreamer pipeline."

