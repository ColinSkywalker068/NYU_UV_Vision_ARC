#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
exporter="${repo_root}/services/export_best_engine.sh"

if [[ ! -x "${exporter}" ]]; then
  echo "Expected the best.pt TensorRT export helper to be executable." >&2
  exit 1
fi

help_text="$("${exporter}" --help)"
if [[ "${help_text}" != *"best.pt"* || "${help_text}" != *"best.engine"* ]]; then
  echo "Expected export helper help to document best.pt input and best.engine output." >&2
  exit 1
fi

if ! grep -q 'INVOCATION_ID=.*export-best-engine' "${exporter}"; then
  echo "Expected export helper to disable interactive Docker TTY attachment." >&2
  exit 1
fi

if ! grep -q 'from ultralytics import YOLO' "${exporter}"; then
  echo "Expected export helper to use the installed Ultralytics Python API." >&2
  exit 1
fi

if ! grep -q "format='engine'" "${exporter}"; then
  echo "Expected export helper to request TensorRT engine output." >&2
  exit 1
fi
