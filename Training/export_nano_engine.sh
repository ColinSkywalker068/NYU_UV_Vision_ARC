#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODE="convert"
INPUT_MODEL=""
ENGINE_OUTPUT=""
ONNX_OUTPUT=""
IMGSZ="640"
TRTEXEC_BIN="${TRTEXEC_BIN:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") --input MODEL [options]

Modes:
  convert   Export .pt to .onnx if needed, then build a TensorRT FP16 engine.
  node      Export .pt to .onnx if needed, then launch yolo_cpp on the ONNX file.

Options:
  --input PATH          Input .pt or .onnx model.
  --engine-output PATH  Output .engine path. Defaults to INPUT stem + .engine.
  --onnx-output PATH    Output .onnx path when converting from .pt.
  --imgsz N             Export image size for ONNX conversion. Default: 640.
  --mode MODE           convert or node. Default: convert.
  --trtexec PATH        Explicit TensorRT trtexec binary.
  -h, --help            Show this help.
EOF
}

log() {
  printf '[export-nano-engine] %s\n' "$*"
}

die() {
  printf '[export-nano-engine] ERROR: %s\n' "$*" >&2
  exit 1
}

abspath() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s\n' "$path"
    return
  fi

  local dir base
  dir="$(cd -P "$(dirname "$path")" && pwd)"
  base="$(basename "$path")"
  printf '%s/%s\n' "$dir" "$base"
}

resolve_trtexec() {
  if [[ -n "$TRTEXEC_BIN" ]]; then
    [[ -x "$TRTEXEC_BIN" ]] || die "TRTEXEC_BIN is set but not executable: $TRTEXEC_BIN"
    printf '%s\n' "$TRTEXEC_BIN"
    return
  fi

  if command -v trtexec >/dev/null 2>&1; then
    command -v trtexec
    return
  fi

  for candidate in \
    /usr/src/tensorrt/bin/trtexec \
    /usr/bin/trtexec \
    /usr/local/bin/trtexec \
    /opt/tensorrt/bin/trtexec
  do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  die "trtexec was not found. Set TRTEXEC_BIN or source the TensorRT environment first."
}

export_onnx_from_pt() {
  local input_pt="$1"
  local onnx_path="$2"
  local imgsz="$3"

  log "Exporting ONNX from ${input_pt}"
  python3 - "$input_pt" "$onnx_path" "$imgsz" <<'PY'
from pathlib import Path
import sys

from ultralytics import YOLO

input_pt = Path(sys.argv[1]).resolve()
onnx_path = Path(sys.argv[2]).resolve()
imgsz = int(sys.argv[3])

model = YOLO(str(input_pt))
exported = model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, opset=17)
exported_path = Path(exported).resolve() if exported else input_pt.with_suffix(".onnx")
if exported_path != onnx_path:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.exists():
        exported_path.replace(onnx_path)
    else:
        raise FileNotFoundError(f"ONNX export did not create {exported_path}")
print(onnx_path)
PY
}

build_engine_from_onnx() {
  local onnx_path="$1"
  local engine_path="$2"
  local trtexec
  trtexec="$(resolve_trtexec)"

  mkdir -p "$(dirname "$engine_path")"

  log "Building FP16 engine with ${trtexec}"
  log "ONNX: ${onnx_path}"
  log "Engine: ${engine_path}"
  "$trtexec" --onnx="$onnx_path" --saveEngine="$engine_path" --fp16
}

launch_detector_on_onnx() {
  local onnx_path="$1"
  log "Launching detector on ${onnx_path}"
  cd "$REPO_ROOT"
  exec uv run yolo_cpp basler.launch.py \
    show:=false \
    detector_backend:=deepstream \
    model_path:="$onnx_path"
}

while (($#)); do
  case "$1" in
    --input)
      INPUT_MODEL="${2:-}"
      shift 2
      ;;
    --engine-output)
      ENGINE_OUTPUT="${2:-}"
      shift 2
      ;;
    --onnx-output)
      ONNX_OUTPUT="${2:-}"
      shift 2
      ;;
    --imgsz)
      IMGSZ="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --trtexec)
      TRTEXEC_BIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$INPUT_MODEL" ]] || { usage; exit 2; }

INPUT_MODEL="$(abspath "$INPUT_MODEL")"
[[ -f "$INPUT_MODEL" ]] || die "Input model not found: $INPUT_MODEL"

INPUT_STEM="${INPUT_MODEL%.*}"
if [[ -z "$ENGINE_OUTPUT" ]]; then
  ENGINE_OUTPUT="${INPUT_STEM}.engine"
fi
ENGINE_OUTPUT="$(abspath "$ENGINE_OUTPUT")"

INPUT_EXT="${INPUT_MODEL##*.}"
case "$INPUT_EXT" in
  pt)
    if [[ -z "$ONNX_OUTPUT" ]]; then
      ONNX_OUTPUT="${INPUT_STEM}.onnx"
    fi
    ONNX_OUTPUT="$(abspath "$ONNX_OUTPUT")"
    export_onnx_from_pt "$INPUT_MODEL" "$ONNX_OUTPUT" "$IMGSZ"
    ;;
  onnx)
    if [[ -z "$ONNX_OUTPUT" ]]; then
      ONNX_OUTPUT="$INPUT_MODEL"
    fi
    ONNX_OUTPUT="$(abspath "$ONNX_OUTPUT")"
    if [[ "$INPUT_MODEL" != "$ONNX_OUTPUT" ]]; then
      cp -f "$INPUT_MODEL" "$ONNX_OUTPUT"
    fi
    ;;
  *)
    die "Expected a .pt or .onnx input, got: $INPUT_MODEL"
    ;;
esac

case "$MODE" in
  convert)
    build_engine_from_onnx "$ONNX_OUTPUT" "$ENGINE_OUTPUT"
    ;;
  node)
    launch_detector_on_onnx "$ONNX_OUTPUT"
    ;;
  *)
    die "Unsupported mode: $MODE"
    ;;
esac
