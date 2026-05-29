#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
HOST_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
DEFAULT_REPO_ROOT="/home/ultraviolet/CV_Detection"
DEFAULT_IMAGE="nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus"

REPO_ROOT="${CV_STM_REPO_ROOT:-${HOST_REPO_ROOT}}"

log() {
  printf '[fps-benchmark] %s\n' "$*"
}

die() {
  printf '[fps-benchmark] ERROR: %s\n' "$*" >&2
  exit 1
}

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES|on|On|ON) return 0 ;;
    *) return 1 ;;
  esac
}

source_setup_file() {
  local setup="$1"
  local had_nounset=0
  case "$-" in
    *u*)
      had_nounset=1
      set +u
      ;;
  esac

  # ROS/colcon setup files may read optional variables before defining them.
  # shellcheck source=/dev/null
  source "$setup"
  local status=$?

  if [ "$had_nounset" -eq 1 ]; then
    set -u
  fi
  return "$status"
}

source_ros() {
  local setup
  for setup in /opt/ros/humble/setup.bash /opt/ros/humble/install/setup.bash; do
    if [ -f "$setup" ] && source_setup_file "$setup"; then
      return
    fi
  done
  die "ROS 2 Humble setup file was not found under /opt/ros/humble"
}

source_workspace() {
  local setup="${REPO_ROOT}/install/setup.bash"
  if [ ! -f "$setup" ]; then
    log "Missing ${setup}"
    die "Build first with 'uv build yolo_cpp' or the equivalent colcon build."
  fi
  source_setup_file "$setup" || die "Failed to source workspace setup: ${setup}"
}

delegate_to_container_if_needed() {
  if is_true "${CV_FPS_BENCH_INSIDE_CONTAINER:-false}" ||
    is_true "${CV_FPS_BENCH_NO_CONTAINER:-false}"
  then
    return
  fi

  if command -v ros2 >/dev/null 2>&1; then
    return
  fi

  [ -x "${SCRIPT_DIR}/start-container.sh" ] ||
    die "ros2 is not available and ${SCRIPT_DIR}/start-container.sh is not executable"

  local image="${CV_STM_IMAGE:-${DEFAULT_IMAGE}}"
  local container_repo="${CV_STM_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
  local quoted_args=""
  if [ "$#" -gt 0 ]; then
    printf -v quoted_args ' %q' "$@"
  fi

  log "ros2 is not available in this shell; starting ${image} and running inside the container"
  exec env \
    CV_FPS_BENCH_INSIDE_CONTAINER=true \
    UV_CONTAINER_NAME="${UV_CONTAINER_NAME:-cv-fps-benchmark}" \
    "${SCRIPT_DIR}/start-container.sh" "${image}" \
    bash -lc "cd '${container_repo}' && exec ./services/fps-benchmark.sh${quoted_args}"
}

main() {
  delegate_to_container_if_needed "$@"
  cd "${REPO_ROOT}"
  source_ros
  source_workspace
  exec ros2 run yolo_cpp fps_benchmark "$@"
}

main "$@"
