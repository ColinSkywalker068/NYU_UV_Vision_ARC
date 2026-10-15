#!/usr/bin/env bash
# Observational CPU benchmark: actual tracker/prediction sources, no ROS or hardware.
# Requires the existing local test image. Does not build/pull images or change gains.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${CV_TEST_IMAGE:-cv-detection-test:humble}
output=${CV_TEST_OUTPUT:-"$repo/.local-test-output"}
mkdir -p "$output"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "Local test image unavailable: $image. Prepare it with CV_TEST_IMAGE set to that name using tools/run-local-tracking-tests.sh." >&2
  exit 1
fi
# Source mount is read-only. No host networking, devices, or persistent container output.
docker run --rm --pull never --network none -v "$repo:/src:ro" "$image" bash -lc '
  set -e
  g++ -std=c++17 -I/src/yolo_cpp/include -I/usr/include/opencv4 \
    /src/yolo_cpp/src/tracker.cpp /src/yolo_cpp/src/tracking_math.cpp \
    /src/yolo_cpp/src/aim_prediction.cpp /src/yolo_cpp/src/aim_geometry.cpp \
    /src/tools/benchmark-moving-following.cpp -lopencv_core -o /tmp/moving-following
  /tmp/moving-following
' 2>&1 | tee "$output/moving-following-benchmark.txt"
