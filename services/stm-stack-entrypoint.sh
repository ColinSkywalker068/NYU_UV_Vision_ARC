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

source_setup_file() {
  local setup="$1"
  local had_nounset=0
  case "$-" in
    *u*)
      had_nounset=1
      set +u
      ;;
  esac

  # ROS/colcon setup files read optional variables such as COLCON_TRACE before
  # defining them, so source them with nounset relaxed.
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
    die "Missing ${setup}. Competition startup does not build automatically; run 'uv build yolo_cpp' before starting the service."
  fi
  source_setup_file "$setup" || die "Failed to source workspace setup: ${setup}"
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
  local element="$1"
  if gst-inspect-1.0 "$element" >/dev/null 2>&1; then
    return
  fi

  log "GStreamer element ${element} is unavailable in CV_STM_IMAGE=${CV_STM_IMAGE:-unknown}"
  log "GST_PLUGIN_PATH=${GST_PLUGIN_PATH:-}"
  log "GST_PLUGIN_PATH_1_0=${GST_PLUGIN_PATH_1_0:-}"
  log "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  if [ "$element" = "pylonsrc" ]; then
    log "pylonsrc is provided by the Basler Pylon GStreamer package; direct runs may work if they use a different or manually modified container"
  fi
  die "Required GStreamer element is unavailable: $element"
}

check_ros_package() {
  require_command ros2
  ros2 pkg prefix yolo_cpp >/dev/null 2>&1 || die "ROS package 'yolo_cpp' is not available from the sourced workspace"
}

model_path() {
  printf '%s' "${CV_STM_MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
}

launch_profile() {
  printf '%s' "${CV_STM_LAUNCH_PROFILE:-basler}"
}

launch_file() {
  case "$(launch_profile)" in
    basler) printf '%s' "basler.launch.py" ;;
    realsense) printf '%s' "realsense.launch.py" ;;
    *) die "Unsupported CV_STM_LAUNCH_PROFILE=$(launch_profile). Expected basler or realsense." ;;
  esac
}

camera_type() {
  if [ -n "${CV_STM_CAMERA_TYPE:-}" ]; then
    printf '%s' "$CV_STM_CAMERA_TYPE"
    return
  fi
  case "$(launch_profile)" in
    basler) printf '%s' "basler" ;;
    realsense) printf '%s' "realsense" ;;
    *) die "Unsupported CV_STM_LAUNCH_PROFILE=$(launch_profile). Expected basler or realsense." ;;
  esac
}

basler_pixel_format() {
  printf '%s' "${CV_STM_BASLER_PIXEL_FORMAT:-bayer_rggb8}"
}

is_basler_bayer_format() {
  case "$(basler_pixel_format)" in
    bayer*|Bayer*|rggb|Rggb|RGGB|bggr|Bggr|BGGR|grbg|Grbg|GRBG|gbrg|Gbrg|GBRG) return 0 ;;
    *) return 1 ;;
  esac
}

detector_backend() {
  if [ -n "${CV_STM_DETECTOR_BACKEND:-}" ]; then
    printf '%s' "$CV_STM_DETECTOR_BACKEND"
    return
  fi
  case "$(launch_profile)" in
    basler) printf '%s' "deepstream" ;;
    realsense) printf '%s' "native_tensorrt" ;;
    *) die "Unsupported CV_STM_LAUNCH_PROFILE=$(launch_profile). Expected basler or realsense." ;;
  esac
}

serial_port() {
  printf '%s' "${CV_STM_SERIAL_PORT:-/dev/ttyUSB0}"
}

serial_required() {
  if [ -n "${CV_STM_REQUIRE_SERIAL:-}" ]; then
    printf '%s' "$CV_STM_REQUIRE_SERIAL"
    return
  fi
  case "$(launch_profile)" in
    basler) printf '%s' "true" ;;
    realsense) printf '%s' "false" ;;
    *) die "Unsupported CV_STM_LAUNCH_PROFILE=$(launch_profile). Expected basler or realsense." ;;
  esac
}

run_preflight() {
  log "Running $(launch_profile) preflight"
  check_ros_package

  require_file "$(model_path)"

  if [ "$(launch_profile)" = "basler" ] && [ "$(detector_backend)" = "deepstream" ]; then
    require_command gst-inspect-1.0
    require_gst_element nvstreammux
    require_gst_element nvinfer
    require_gst_element nvvideoconvert
  fi

  if [ "$(launch_profile)" = "basler" ]; then
    require_command gst-inspect-1.0
    if [[ "${CV_STM_IMAGE:-}" != *cplusplus* ]]; then
      log "Basler profile is using CV_STM_IMAGE=${CV_STM_IMAGE:-unknown}; the default C++ image is expected unless this custom image includes pylonsrc"
    fi
    require_gst_element pylonsrc
    if [ ! -d /opt/pylon ]; then
      log "Pylon SDK directory /opt/pylon is not mounted; continuing because pylonsrc is available"
    fi
    if is_basler_bayer_format; then
      require_gst_element bayer2rgb
    fi
  fi

  if is_true "$(serial_required)"; then
    require_file "$(serial_port)"
  elif [ ! -e "$(serial_port)" ]; then
    log "Serial path $(serial_port) is missing, but serial is not required for this profile"
  fi

  log "Preflight OK for $(launch_profile): package, model, camera/inference prerequisites, and profile-specific serial policy are ready"
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
    "debug_publish_image_raw:=${CV_STM_DEBUG_PUBLISH_IMAGE_RAW:-false}"
    "model_path:=$(model_path)"
  )
  append_optional_launch_arg CV_STM_TARGET_COLOR target_color
  append_optional_launch_arg CV_STM_USE_MODEL_COLOR_CLASSES use_model_color_classes
  append_optional_launch_arg CV_STM_ENABLE_COLOR_VERIFICATION enable_color_verification
  append_optional_launch_arg CV_STM_SAVE_TARGET_FRAMES save_target_frames
  append_optional_launch_arg CV_STM_TARGET_FRAME_SAVE_DIR target_frame_save_dir
  append_optional_launch_arg CV_STM_TARGET_FRAME_SAVE_MAX target_frame_save_max
  append_optional_launch_arg CV_STM_TARGET_FRAME_JPEG_QUALITY target_frame_jpeg_quality

  case "$(launch_profile)" in
    basler)
      LAUNCH_ARGS+=(
        "camera_type:=$(camera_type)"
        "detector_backend:=$(detector_backend)"
        "allow_native_tensorrt_fallback:=${CV_STM_ALLOW_NATIVE_TENSORRT_FALLBACK:-false}"
        "serial_verbose:=${CV_STM_SERIAL_VERBOSE:-false}"
        "serial_port:=$(serial_port)"
      )
      append_optional_launch_arg CV_STM_BASLER_EXPOSURE_US basler_exposure_us
      append_optional_launch_arg CV_STM_BASLER_GAIN_DB basler_gain_db
      append_optional_launch_arg CV_STM_BASLER_FPS basler_fps
      append_optional_launch_arg CV_STM_BASLER_UNCAPPED_FPS basler_uncapped_fps
      append_optional_launch_arg CV_STM_BASLER_PIXEL_FORMAT basler_pixel_format
      append_optional_launch_arg CV_STM_BASLER_SOURCE_WIDTH basler_source_width
      append_optional_launch_arg CV_STM_BASLER_SOURCE_HEIGHT basler_source_height
      append_optional_launch_arg CV_STM_BASLER_USE_FULL_SENSOR basler_use_full_sensor
      append_optional_launch_arg CV_STM_BASLER_FLIP basler_flip
      ;;
    realsense)
      append_optional_launch_arg CV_STM_REALSENSE_WIDTH rs_width
      append_optional_launch_arg CV_STM_REALSENSE_HEIGHT rs_height
      append_optional_launch_arg CV_STM_REALSENSE_FPS rs_fps
      append_optional_launch_arg CV_STM_REALSENSE_EXPOSURE rs_exposure
      append_optional_launch_arg CV_STM_REALSENSE_AUTO_EXPOSURE rs_auto_exposure
      append_optional_launch_arg CV_STM_REALSENSE_USE_POLL rs_use_poll
      append_optional_launch_arg CV_STM_REALSENSE_USE_MAX_NATIVE_FPS rs_use_max_native_fps
      ;;
    *) die "Unsupported CV_STM_LAUNCH_PROFILE=$(launch_profile). Expected basler or realsense." ;;
  esac

  if [ -n "${CV_STM_EXTRA_LAUNCH_ARGS:-}" ]; then
    local -a extra_args=()
    read -r -a extra_args <<< "${CV_STM_EXTRA_LAUNCH_ARGS}"
    LAUNCH_ARGS+=("${extra_args[@]}")
  fi

  log "Launching yolo_cpp $(launch_file) with profile $(launch_profile)"
  exec ros2 launch yolo_cpp "$(launch_file)" "${LAUNCH_ARGS[@]}"
}

graph_has_line() {
  local haystack="$1"
  local needle="$2"
  printf '%s\n' "$haystack" | grep -Fxq "$needle"
}

check_graph_once() {
  local report="${1:-false}"
  local nodes topics missing=0
  local -a expected_nodes=(
    /armor_detector_node
    /armor_tracker_node
    /armor_processor_node
    /aim_solver_node
  )
  if [ "$(launch_profile)" != "realsense" ]; then
    expected_nodes+=(/stm32_serial_bridge_node)
  fi
  nodes="$(ros2 node list 2>/dev/null || true)"
  topics="$(ros2 topic list 2>/dev/null || true)"

  for node in "${expected_nodes[@]}"; do
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
      log "Health OK: expected $(launch_profile) nodes and topics are present"
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

run       Validate prerequisites, then exec the selected launch profile.
preflight Validate package, model, plugin, camera, and serial prerequisites only.
health    Validate the live ROS graph for the selected profile.

CV_STM_LAUNCH_PROFILE defaults to basler. Supported profiles: basler, realsense.
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
