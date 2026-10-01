#!/usr/bin/env bash
# Build and exercise production ROS nodes using synthetic images and disposable PTYs.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=${CV_TEST_IMAGE:-cv-detection-local-tests:humble}
output=${CV_TEST_OUTPUT:-"$repo/.local-test-output"}
mkdir -p "$output"
if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build -f "$repo/Docker/Dockerfile.local-test" -t "$image" "$repo/Docker"
fi
# No device mounts or host networking. The bridge tests open only temporary PTYs.
docker run --rm -v "$repo:/ws/src/CV_Detection:ro" -v "$output:/ws/out" "$image" bash -lc '
  set -e
  source /opt/ros/humble/setup.bash
  cd /ws/out
  colcon build --base-paths /ws/src/CV_Detection --packages-select cv_auto_aim_interfaces yolo_cpp --cmake-args -DBUILD_TESTING=ON
  source install/setup.bash
  python3 /ws/src/CV_Detection/services/test_typec_link_ros.py
  colcon test --base-paths /ws/src/CV_Detection --packages-select yolo_cpp --return-code-on-test-failure
  colcon test-result --verbose
'
