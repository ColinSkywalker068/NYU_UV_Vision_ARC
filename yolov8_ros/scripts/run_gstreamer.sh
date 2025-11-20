#!/usr/bin/env bash
# Simple wrapper script to run the GStreamer pipeline
# Usage: ./run_gstreamer.sh [--display] [--model-engine PATH] [other args...]

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
MODEL_ENGINE="${MODEL_ENGINE:-${PROJECT_ROOT}/yolov8_ros/models/best.engine}"
NUM_CLASSES="${NUM_CLASSES:-1}"
THRESHOLD="${THRESHOLD:-0.5}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.4}"
DEVICE="${DEVICE:-/dev/video0}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
DISPLAY_FLAG=""

# Parse simple arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --display)
      DISPLAY_FLAG="--display"
      shift
      ;;
    --model-engine)
      MODEL_ENGINE="$2"
      shift 2
      ;;
    --num-classes)
      NUM_CLASSES="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--display] [--model-engine PATH] [--num-classes N] [--threshold T] [--device DEV]"
      exit 1
      ;;
  esac
done

# Build command
CMD="python3 ${SCRIPT_DIR}/yolov8_op.py \
  --model-engine ${MODEL_ENGINE} \
  --num-classes ${NUM_CLASSES} \
  --threshold ${THRESHOLD} \
  --nms-threshold ${NMS_THRESHOLD} \
  --device ${DEVICE} \
  --width ${WIDTH} \
  --height ${HEIGHT} \
  --fps ${FPS}"

# Add display flag if requested
if [ -n "${DISPLAY_FLAG}" ]; then
  CMD="${CMD} ${DISPLAY_FLAG}"
fi

echo "Running GStreamer pipeline..."
echo "Command: ${CMD}"
echo ""

eval ${CMD}

