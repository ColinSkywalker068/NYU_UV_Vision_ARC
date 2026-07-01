#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/standalone_tracker"
BIN="${BUILD_DIR}/tracker_replay"

mkdir -p "${BUILD_DIR}"

clang++ -std=c++17 -Wall -Wextra -Wpedantic \
  -I"${SCRIPT_DIR}" \
  -I"${REPO_ROOT}/yolo_cpp/include" \
  -I"${REPO_ROOT}/yolo_cpp/src" \
  "${SCRIPT_DIR}/tracker_replay.cpp" \
  "${REPO_ROOT}/yolo_cpp/src/tracker.cpp" \
  "${REPO_ROOT}/yolo_cpp/src/tracking_math.cpp" \
  -o "${BIN}"

exec "${BIN}" "$@"
