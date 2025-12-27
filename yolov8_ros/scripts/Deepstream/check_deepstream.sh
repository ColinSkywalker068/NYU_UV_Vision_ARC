#!/usr/bin/env bash
# Diagnostic script to check DeepStream installation

echo "=== DeepStream Plugin Check ==="
echo ""

# Check if DeepStream directory exists
DEEPSTREAM_PATH="/opt/nvidia/deepstream/deepstream"
if [ -d "$DEEPSTREAM_PATH" ]; then
    echo "✓ DeepStream directory found: $DEEPSTREAM_PATH"
else
    echo "✗ DeepStream directory NOT found: $DEEPSTREAM_PATH"
    echo "  DeepStream may not be installed in this container."
    exit 1
fi

# Check plugin directory
PLUGIN_PATH="$DEEPSTREAM_PATH/lib/gst-plugins"
if [ -d "$PLUGIN_PATH" ]; then
    echo "✓ Plugin directory found: $PLUGIN_PATH"
    
    # Check for key plugins
    echo ""
    echo "Checking for DeepStream plugins:"
    for plugin in libgstnvstreammux.so libgstnvinfer.so libgstnvvideoconvert.so libgstnvdsosd.so; do
        if [ -f "$PLUGIN_PATH/$plugin" ]; then
            echo "  ✓ $plugin"
        else
            echo "  ✗ $plugin (MISSING)"
        fi
    done
else
    echo "✗ Plugin directory NOT found: $PLUGIN_PATH"
    exit 1
fi

# Check GST_PLUGIN_PATH
echo ""
echo "=== GStreamer Environment ==="
echo "GST_PLUGIN_PATH: ${GST_PLUGIN_PATH:-not set}"
echo "GST_PLUGIN_PATH_1_0: ${GST_PLUGIN_PATH_1_0:-not set}"

# Try to inspect plugin
echo ""
echo "=== Testing GStreamer Plugin Detection ==="
if command -v gst-inspect-1.0 &> /dev/null; then
    if gst-inspect-1.0 nvstreammux &> /dev/null; then
        echo "✓ nvstreammux plugin is detectable by GStreamer"
    else
        echo "✗ nvstreammux plugin NOT detectable by GStreamer"
        echo "  Try: export GST_PLUGIN_PATH=$PLUGIN_PATH"
        echo "  Then: gst-inspect-1.0 nvstreammux"
    fi
else
    echo "✗ gst-inspect-1.0 not found"
fi

echo ""
echo "=== Recommendations ==="
if [ ! -d "$PLUGIN_PATH" ] || [ ! -f "$PLUGIN_PATH/libgstnvstreammux.so" ]; then
    echo "1. Verify you're using the correct Docker image with DeepStream installed"
    echo "2. Check Dockerfile.realsense installs DeepStream 6.4"
    echo "3. Rebuild the Docker image if needed"
else
    echo "✓ DeepStream appears to be installed correctly"
    echo "  If plugins still not found, ensure GST_PLUGIN_PATH is set:"
    echo "  export GST_PLUGIN_PATH=$PLUGIN_PATH"
fi

