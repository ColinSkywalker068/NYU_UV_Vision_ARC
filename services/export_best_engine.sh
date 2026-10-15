#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
MODEL_PATH="${REPO_ROOT}/yolo_cpp/models/best.pt"
ENGINE_PATH="${REPO_ROOT}/yolo_cpp/models/best.engine"
R39_IMAGE="${CV_IMAGE:-cv-detection-cplusplus-r39:local}"

usage() {
  cat <<'EOF'
Usage: services/export_best_engine.sh

Exports yolo_cpp/models/best.pt to yolo_cpp/models/best.engine on this Jetson.
The TensorRT engine is device-local: regenerate it after changing best.pt,
TensorRT, CUDA, or the Jetson platform image.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

[[ -f "$MODEL_PATH" ]] || {
  echo "Missing PyTorch checkpoint: $MODEL_PATH" >&2
  exit 1
}

INVOCATION_ID="${INVOCATION_ID:-export-best-engine}" \
  "${SCRIPT_DIR}/start-container.sh" "$R39_IMAGE" bash -lc "
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libdrm.so.2\${LD_PRELOAD:+:\$LD_PRELOAD}
  cd $(printf '%q' "$REPO_ROOT")
  python3 - $(printf '%q' "$MODEL_PATH") <<'PY'
from ultralytics import YOLO
import sys

YOLO(sys.argv[1]).export(format='engine', imgsz=640, device=0, half=True)
PY
"

[[ -s "$ENGINE_PATH" ]] || {
  echo "TensorRT export did not produce $ENGINE_PATH" >&2
  exit 1
}

printf 'Exported TensorRT engine: %s\n' "$ENGINE_PATH"
