#!/usr/bin/env bash
# Manual DeepStream installation script for containers where build failed
# This can be run inside a running container

set -e

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

if ! curl -L 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/6.4/files?redirect=true&path=deepstream-6.4_6.4.0-1_arm64.deb' \
    -o "$DEEPSTREAM_DEB"; then
    echo "ERROR: Failed to download DeepStream. Possible reasons:"
    echo "  1. Network connectivity issue"
    echo "  2. NGC authentication required (may need API key)"
    echo "  3. URL changed"
    echo ""
    echo "Alternative: Download manually from NVIDIA NGC and copy into container:"
    echo "  docker cp deepstream-6.4_6.4.0-1_arm64.deb <container>:/tmp/"
    echo "  Then run this script again or: dpkg -i /tmp/deepstream-6.4_6.4.0-1_arm64.deb"
    exit 1
fi

# Install
echo "Installing DeepStream..."
if ! dpkg -i "$DEEPSTREAM_DEB"; then
    echo "Fixing dependencies..."
    apt-get update
    apt-get -y -f install
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

