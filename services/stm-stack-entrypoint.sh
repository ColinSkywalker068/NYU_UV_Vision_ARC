#!/usr/bin/env bash
set -euo pipefail

DEFAULT_REPO_ROOT="/home/ultraviolet/CV_Detection"
DEFAULT_MODEL_PATH="/home/ultraviolet/CV_Detection/model_test/best-roboflow-nitish-obb.onnx_b1_gpu0_fp16.engine"

REPO_ROOT="${CV_STM_REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
MODE="${1:-run}"

log() {
  printf '[stm-stack] %s\n' "$*"
}

die() {
  printf '[stm-stack] ERROR: %s\n' "$*" >&2
  exit 1
}

is_true() {
  case "${1:-}" in
    true|True|TRUE|1|yes|Yes|YES|on|On|ON) return 0 ;;
    *) return 1 ;;
  esac
}

source_ros() {
  local setup
  for setup in /opt/ros/humble/setup.bash /opt/ros/humble/install/setup.bash; do
    if [ -f "$setup" ]; then
      # shellcheck source=/dev/null
      source "$setup"
      return
    fi
  done
  die "ROS 2 Humble setup file was not found under /opt/ros/humble"
}

source_workspace() {
  local setup="${REPO_ROOT}/install/setup.bash"
  if [ ! -f "$setup" ]; then
    die "Missing ${setup}. Competition startup does not build automatically; run 'uv build yolo_cpp' before starting the service."
  fi
  # shellcheck source=/dev/null
  source "$setup"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command is missing: $1"
}

require_file() {
  [ -e "$1" ] || die "Required path is missing: $1"
}

require_dir() {
  [ -d "$1" ] || die "Required directory is missing: $1"
}

require_gst_element() {
  gst-inspect-1.0 "$1" >/dev/null 2>&1 || die "Required GStreamer element is unavailable: $1"
}

check_ros_package() {
  require_command ros2
  ros2 pkg prefix yolo_cpp >/dev/null 2>&1 || die "ROS package 'yolo_cpp' is not available from the sourced workspace"
}

model_path() {
  printf '%s' "${CV_STM_MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
}

camera_type() {
  printf '%s' "${CV_STM_CAMERA_TYPE:-basler}"
}

detector_backend() {
  printf '%s' "${CV_STM_DETECTOR_BACKEND:-deepstream}"
}

serial_port() {
  printf '%s' "${CV_STM_SERIAL_PORT:-/dev/ttyUSB0}"
}

run_preflight() {
  log "Running headless competition preflight"
  check_ros_package

  require_file "$(model_path)"

  if [ "$(detector_backend)" = "deepstream" ]; then
    require_command gst-inspect-1.0
    require_gst_element nvstreammux
    require_gst_element nvinfer
    require_gst_element nvvideoconvert
  fi

  if [ "$(camera_type)" = "basler" ]; then
    require_command gst-inspect-1.0
    require_dir /opt/pylon
    require_gst_element pylonsrc
  fi

  if is_true "${CV_STM_REQUIRE_SERIAL:-true}"; then
    require_file "$(serial_port)"
  elif [ ! -e "$(serial_port)" ]; then
    log "Serial path $(serial_port) is missing, but CV_STM_REQUIRE_SERIAL=false"
  fi

  log "Preflight OK: package, model, camera/inference plugins, and serial prerequisites are ready"
}

append_optional_launch_arg() {
  local env_name="$1"
  local launch_name="$2"
  local value="${!env_name:-}"
  if [ -n "$value" ]; then
    LAUNCH_ARGS+=("${launch_name}:=${value}")
  fi
}

run_launch() {
  run_preflight

  LAUNCH_ARGS=(
    "show:=${CV_STM_SHOW:-false}"
    "camera_type:=$(camera_type)"
    "detector_backend:=$(detector_backend)"
    "allow_native_tensorrt_fallback:=${CV_STM_ALLOW_NATIVE_TENSORRT_FALLBACK:-false}"
    "debug_publish_image_raw:=${CV_STM_DEBUG_PUBLISH_IMAGE_RAW:-false}"
    "serial_verbose:=${CV_STM_SERIAL_VERBOSE:-false}"
    "serial_port:=$(serial_port)"
    "model_path:=$(model_path)"
  )

  append_optional_launch_arg CV_STM_BASLER_EXPOSURE_US basler_exposure_us
  append_optional_launch_arg CV_STM_BASLER_GAIN_DB basler_gain_db
  append_optional_launch_arg CV_STM_BASLER_FPS basler_fps
  append_optional_launch_arg CV_STM_BASLER_UNCAPPED_FPS basler_uncapped_fps
  append_optional_launch_arg CV_STM_BASLER_USE_FULL_SENSOR basler_use_full_sensor
  append_optional_launch_arg CV_STM_BASLER_FLIP basler_flip

  if [ -n "${CV_STM_EXTRA_LAUNCH_ARGS:-}" ]; then
    local -a extra_args=()
    read -r -a extra_args <<< "${CV_STM_EXTRA_LAUNCH_ARGS}"
    LAUNCH_ARGS+=("${extra_args[@]}")
  fi

  log "Launching yolo_cpp stm.launch.py in competition mode"
  exec ros2 launch yolo_cpp stm.launch.py "${LAUNCH_ARGS[@]}"
}

graph_has_line() {
  local haystack="$1"
  local needle="$2"
  printf '%s\n' "$haystack" | grep -Fxq "$needle"
}

check_graph_once() {
  local report="${1:-false}"
  local nodes topics missing=0
  nodes="$(ros2 node list 2>/dev/null || true)"
  topics="$(ros2 topic list 2>/dev/null || true)"

  for node in /armor_detector_node /armor_tracker_node /armor_processor_node /aim_solver_node /stm32_serial_bridge_node; do
    if ! graph_has_line "$nodes" "$node"; then
      [ "$report" = "true" ] && printf '[stm-stack] missing node: %s\n' "$node" >&2
      missing=1
    fi
  done

  for topic in /detector/armors /tracker/tracked_armors /processor/target /aim_solution; do
    if ! graph_has_line "$topics" "$topic"; then
      [ "$report" = "true" ] && printf '[stm-stack] missing topic: %s\n' "$topic" >&2
      missing=1
    fi
  done

  return "$missing"
}

run_health() {
  check_ros_package
  local timeout="${CV_STM_HEALTH_TIMEOUT_S:-10}"
  local deadline=$(( $(date +%s) + timeout ))

  while true; do
    if check_graph_once false; then
      log "Health OK: expected STM stack nodes and topics are present"
      return
    fi

    if [ "$(date +%s)" -ge "$deadline" ]; then
      check_graph_once true || true
      die "Health check failed before timeout (${timeout}s)"
    fi
    sleep 1
  done
}

usage() {
  cat <<'EOF'
Usage: stm-stack-entrypoint.sh [run|preflight|health]

run       Validate prerequisites, then exec ros2 launch yolo_cpp stm.launch.py.
preflight Validate package, model, plugin, camera, and serial prerequisites only.
health    Validate the live ROS graph for the expected STM stack nodes and topics.
EOF
}

main() {
  case "$MODE" in
    -h|--help|help) usage; return ;;
  esac

  cd "$REPO_ROOT" || die "Repository root is unavailable: ${REPO_ROOT}"
  source_ros
  source_workspace

  case "$MODE" in
    run) run_launch ;;
    preflight) run_preflight ;;
    health) run_health ;;
    *) usage >&2; exit 2 ;;
  esac
}

main "$@"
