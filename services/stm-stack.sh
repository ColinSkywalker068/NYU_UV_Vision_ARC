#!/usr/bin/env bash
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SERVICE_NAME="${CV_STM_SERVICE_NAME:-stm-stack.service}"
SERVICE_SRC="${SCRIPT_DIR}/${SERVICE_NAME}"
SERVICE_DEST="/etc/systemd/system/${SERVICE_NAME}"
ENV_DIR="${CV_STM_ENV_DIR:-/etc/cv-detection}"
ENV_FILE="${CV_STM_ENV_FILE:-${ENV_DIR}/stm-stack.env}"
DEFAULT_IMAGE="nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon"
LEGACY_CPLUSPLUS_IMAGE="nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus"
ORIN_MODEL_RELATIVE="yolo_cpp/models/nitish-orin.engine"
AGX_MODEL_RELATIVE="yolo_cpp/models/nitish-agx.engine"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

CV_STM_IMAGE="${CV_STM_IMAGE:-${DEFAULT_IMAGE}}"
if [ "$CV_STM_IMAGE" = "$LEGACY_CPLUSPLUS_IMAGE" ]; then
  CV_STM_IMAGE="$DEFAULT_IMAGE"
fi
CV_STM_REPO_ROOT="${CV_STM_REPO_ROOT:-/home/ultraviolet/CV_Detection}"
UV_CONTAINER_NAME="${UV_CONTAINER_NAME:-cv-stm-stack}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-37}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"

DOCKER_CMD=(docker)

log() {
  printf '[stm-stack] %s\n' "$*"
}

die() {
  printf '[stm-stack] ERROR: %s\n' "$*" >&2
  exit 1
}

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

ensure_docker() {
  if docker info >/dev/null 2>&1; then
    DOCKER_CMD=(docker)
    return
  fi
  if command -v sudo >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
    DOCKER_CMD=(sudo docker)
    return
  fi
  die "Docker is not available or the daemon is not running"
}

docker_cmd() {
  "${DOCKER_CMD[@]}" "$@"
}

container_running() {
  docker_cmd inspect -f '{{.State.Running}}' "$UV_CONTAINER_NAME" 2>/dev/null | grep -Fxq true
}

write_default_env_file() {
  local tmp
  tmp="$(mktemp)"
  cat > "$tmp" <<EOF
# Optional overrides for stm-stack.service and services/stm-stack.sh.
# Keep startup prebuilt and deterministic; build explicitly with 'uv build yolo_cpp'.
CV_STM_IMAGE=${DEFAULT_IMAGE}
# To test a locally baked Pylon image before pushing:
# CV_STM_IMAGE=cv-detection-cplusplus-pylon:local
UV_CONTAINER_NAME=cv-stm-stack
CV_STM_LAUNCH_PROFILE=basler
# Keep the competition graph local to this robot by default. Override these
# when intentionally using remote ROS tools across machines.
ROS_DOMAIN_ID=37
ROS_LOCALHOST_ONLY=1
# Optional. When unset, yolo_cpp/config/default_model_path.txt supplies the robot-local fallback.
# CV_STM_MODEL_PATH=/path/to/model.engine
# CV_STM_TARGET_COLOR=blue
# Default is to trust model color classes and skip HSV verification.
# Accept both model colors during detector bring-up by setting both false:
# CV_STM_USE_MODEL_COLOR_CLASSES=false
# CV_STM_ENABLE_COLOR_VERIFICATION=false
# CV_STM_SERIAL_PORT=/dev/ttyUSB0
# CV_STM_REQUIRE_SERIAL=true
CV_STM_PREVIEW_MAX_FPS=60.0
# Toggle the OpenCV preview window for detector frames. The preview state is
# applied at startup, so use './services/stm-stack.sh preview-on/off'.
CV_STM_SHOW=false
# Save up to 100 compressed target frames in pictures/ next to vision_logs/.
# CV_STM_SAVE_TARGET_FRAMES=false
# CV_STM_TARGET_FRAME_SAVE_DIR=
# CV_STM_TARGET_FRAME_SAVE_MAX=100
# CV_STM_TARGET_FRAME_JPEG_QUALITY=85
# RealSense profile skips Basler/DeepStream/Pylon checks and serial by default:
# CV_STM_LAUNCH_PROFILE=realsense
# CV_STM_REQUIRE_SERIAL=false
# Field service defaults intentionally request a deterministic 160 FPS Basler cap
# while keeping preview off. Other Basler launch knobs stay on launch defaults.
CV_STM_BASLER_FPS=160
CV_STM_BASLER_UNCAPPED_FPS=false
# CV_STM_DEEPSTREAM_PROFILE_STAGES=false
# Diagnostic A/B for Nano CUDA Bayer crashes; keeps CUDA Bayer but uses synchronous copies.
# YOLO_CUDA_BAYER_SYNC_MEMCPY=1
# CV_STM_EXTRA_LAUNCH_ARGS="basler_exposure_us:=5000.0 basler_gain_db:=5.0"
# CV_STM_REALSENSE_WIDTH=640
# CV_STM_REALSENSE_HEIGHT=480
# CV_STM_REALSENSE_FPS=60
EOF
  run_sudo install -m 0644 "$tmp" "$ENV_FILE"
  rm -f "$tmp"
}

install_service() {
  [ -f "$SERVICE_SRC" ] || die "Service file not found: ${SERVICE_SRC}"
  run_sudo install -d -m 0755 "$ENV_DIR"
  run_sudo install -m 0644 "$SERVICE_SRC" "$SERVICE_DEST"

  if [ ! -f "$ENV_FILE" ]; then
    write_default_env_file
    log "Created ${ENV_FILE}"
  else
    if grep -q "^CV_STM_IMAGE=${LEGACY_CPLUSPLUS_IMAGE}$" "$ENV_FILE"; then
      set_env_value CV_STM_IMAGE "$DEFAULT_IMAGE"
      log "Updated stale CV_STM_IMAGE in ${ENV_FILE} to ${DEFAULT_IMAGE}"
    fi
    log "Keeping existing ${ENV_FILE}"
  fi

  run_sudo systemctl daemon-reload
  run_sudo systemctl enable "$SERVICE_NAME"
  log "Installed and enabled ${SERVICE_NAME}"
  log "Run './services/stm-stack.sh preflight' before starting the competition stack"
}

systemctl_action() {
  run_sudo systemctl "$1" "$SERVICE_NAME"
}

set_env_value() {
  local key="$1"
  local value="$2"
  local tmp
  tmp="$(mktemp)"

  run_sudo install -d -m 0755 "$ENV_DIR"
  if [ -f "$ENV_FILE" ]; then
    awk -v key="$key" -v value="$value" '
      BEGIN { found = 0 }
      $0 ~ "^[[:space:]]*" key "=" {
        print key "=" value
        found = 1
        next
      }
      { print }
      END {
        if (!found) {
          print key "=" value
        }
      }
    ' "$ENV_FILE" > "$tmp"
  else
    {
      printf '# Optional overrides for stm-stack.service and services/stm-stack.sh.\n'
      printf '%s=%s\n' "$key" "$value"
    } > "$tmp"
  fi

  run_sudo install -m 0644 "$tmp" "$ENV_FILE"
  rm -f "$tmp"
}

unset_env_values() {
  local tmp
  tmp="$(mktemp)"
  local keys="$*"

  run_sudo install -d -m 0755 "$ENV_DIR"
  if [ -f "$ENV_FILE" ]; then
    awk -v keys="$keys" '
      BEGIN {
        split(keys, key_list, " ")
        for (i in key_list) {
          remove[key_list[i]] = 1
        }
      }
      {
        line = $0
        sub(/^[[:space:]]*/, "", line)
        split(line, parts, "=")
        if (!(parts[1] in remove)) {
          print
        }
      }
    ' "$ENV_FILE" > "$tmp"
  else
    printf '# Optional overrides for stm-stack.service and services/stm-stack.sh.\n' > "$tmp"
  fi

  run_sudo install -m 0644 "$tmp" "$ENV_FILE"
  rm -f "$tmp"
}

set_preview() {
  local show_value="$1"
  local state_label="$2"
  set_env_value CV_STM_SHOW "$show_value"
  log "Set CV_STM_SHOW=${show_value} in ${ENV_FILE}"
  if [ "$show_value" = "true" ]; then
    local display_value="${DISPLAY:-:0}"
    set_env_value DISPLAY "$display_value"
    log "Set DISPLAY=${display_value} in ${ENV_FILE}"
    if command -v xhost >/dev/null 2>&1; then
      if DISPLAY="$display_value" xhost +si:localuser:root >/dev/null; then
        log "Allowed local root X11 access for the service container"
      else
        log "Warning: xhost could not allow local root X11 access; the browser preview at http://<robot-host>:8080/ may still work"
      fi
    else
      log "Warning: xhost is not installed; the browser preview at http://<robot-host>:8080/ may still work"
    fi
  fi
  log "Restarting ${SERVICE_NAME} so preview ${state_label} takes effect"
  systemctl_action restart
}

restore_basler_launch_defaults() {
  unset_env_values \
    CV_STM_BASLER_PIXEL_FORMAT \
    CV_STM_BASLER_SOURCE_WIDTH \
    CV_STM_BASLER_SOURCE_HEIGHT \
    CV_STM_BASLER_USE_FULL_SENSOR \
    CV_STM_BASLER_FLIP \
    CV_STM_DEEPSTREAM_PRE_MUX_RESIZE \
    CV_STM_DEEPSTREAM_PRE_MUX_WIDTH \
    CV_STM_DEEPSTREAM_PRE_MUX_HEIGHT \
    CV_STM_DEEPSTREAM_PROFILE_STAGES \
    CV_STM_DEEPSTREAM_PROFILE_STAGE_INTERVAL \
    CV_STM_DEEPSTREAM_BAYER_CONVERSION_BACKEND
  set_env_value CV_STM_BASLER_FPS 160
  set_env_value CV_STM_BASLER_UNCAPPED_FPS false
  set_env_value CV_STM_SHOW false
  set_env_value CV_STM_LAUNCH_PROFILE basler
  log "Restored field Basler defaults in ${ENV_FILE}"
  log "Restarting ${SERVICE_NAME} with basler_fps=160, basler_uncapped_fps=false, and show=false"
  systemctl_action restart
}

set_orin_profile() {
  local model_path="${CV_STM_REPO_ROOT%/}/${ORIN_MODEL_RELATIVE}"
  set_env_value CV_STM_MODEL_PATH "$model_path"
  set_env_value CV_STM_DETECTOR_BACKEND native_tensorrt
  set_env_value CV_STM_BASLER_PIXEL_FORMAT bayer_rggb8
  set_env_value CV_STM_BASLER_FPS 160
  set_env_value CV_STM_BASLER_EXPOSURE_US 3000.0
  set_env_value CV_STM_BASLER_USE_FULL_SENSOR true
  set_env_value CV_STM_BASLER_SOURCE_WIDTH 1920
  set_env_value CV_STM_BASLER_SOURCE_HEIGHT 1200
  set_env_value CV_STM_SHOW false
  set_env_value CV_STM_LAUNCH_PROFILE basler
  log "Set Orin native TensorRT + full-sensor Bayer profile in ${ENV_FILE}"
  log "Restarting ${SERVICE_NAME} with Orin model and stable native Basler defaults"
  systemctl_action restart
}

set_agx_profile() {
  local model_path="${CV_STM_REPO_ROOT%/}/${AGX_MODEL_RELATIVE}"
  set_env_value CV_STM_MODEL_PATH "$model_path"
  unset_env_values \
    CV_STM_DETECTOR_BACKEND \
    CV_STM_BASLER_PIXEL_FORMAT \
    CV_STM_BASLER_EXPOSURE_US \
    CV_STM_BASLER_USE_FULL_SENSOR \
    CV_STM_BASLER_SOURCE_WIDTH \
    CV_STM_BASLER_SOURCE_HEIGHT
  set_env_value CV_STM_BASLER_FPS 160
  set_env_value CV_STM_BASLER_UNCAPPED_FPS false
  set_env_value CV_STM_SHOW false
  set_env_value CV_STM_LAUNCH_PROFILE basler
  log "Set AGX DeepStream profile in ${ENV_FILE}"
  log "Restarting ${SERVICE_NAME} with AGX model"
  systemctl_action restart
}

show_status() {
  systemctl status "$SERVICE_NAME" --no-pager
}

show_logs() {
  run_sudo journalctl -u "$SERVICE_NAME" -f --no-pager
}

run_preflight() {
  ensure_docker
  docker_cmd image inspect "$CV_STM_IMAGE" >/dev/null 2>&1 || die "Docker image is missing: ${CV_STM_IMAGE}"

  local preflight_container="${UV_CONTAINER_NAME}-preflight"
  docker_cmd rm -f "$preflight_container" >/dev/null 2>&1 || true

  log "Running preflight in ${CV_STM_IMAGE}"
  local status=0
  UV_CONTAINER_NAME="$preflight_container" \
  CV_STM_IMAGE="$CV_STM_IMAGE" \
  CV_STM_REPO_ROOT="$CV_STM_REPO_ROOT" \
    "${SCRIPT_DIR}/start-container.sh" "$CV_STM_IMAGE" \
      bash -lc "cd '${CV_STM_REPO_ROOT}' && exec ./services/stm-stack-entrypoint.sh preflight" || status=$?

  docker_cmd rm -f "$preflight_container" >/dev/null 2>&1 || true
  return "$status"
}

run_foreground() {
  ensure_docker
  docker_cmd image inspect "$CV_STM_IMAGE" >/dev/null 2>&1 || die "Docker image is missing: ${CV_STM_IMAGE}"
  docker_cmd rm -f "$UV_CONTAINER_NAME" >/dev/null 2>&1 || true

  log "Running foreground stack in ${CV_STM_IMAGE}"
  UV_CONTAINER_NAME="$UV_CONTAINER_NAME" \
  CV_STM_IMAGE="$CV_STM_IMAGE" \
  CV_STM_REPO_ROOT="$CV_STM_REPO_ROOT" \
    "${SCRIPT_DIR}/start-container.sh" "$CV_STM_IMAGE" \
      bash -lc "cd '${CV_STM_REPO_ROOT}' && exec ./services/stm-stack-entrypoint.sh run"
}

run_health() {
  ensure_docker
  if ! container_running; then
    die "Container ${UV_CONTAINER_NAME} is not running"
  fi
  docker_cmd exec "$UV_CONTAINER_NAME" \
    bash -lc "cd '${CV_STM_REPO_ROOT}' && exec ./services/stm-stack-entrypoint.sh health"
}

run_yaw_smoke() {
  ensure_docker
  docker_cmd image inspect "$CV_STM_IMAGE" >/dev/null 2>&1 || die "Docker image is missing: ${CV_STM_IMAGE}"

  local serial_port="${CV_STM_SERIAL_PORT:-/dev/ttyUSB0}"
  local yaw_rad="${CV_STM_YAW_SMOKE_RAD:-0.05}"
  local count="${CV_STM_YAW_SMOKE_COUNT:-1}"
  local interval_ms="${CV_STM_YAW_SMOKE_INTERVAL_MS:-50}"
  local baud="${CV_STM_YAW_SMOKE_BAUD:-115200}"
  local -a docker_args=(
    run --rm --privileged --network host
    --volume "${REPO_ROOT}:${CV_STM_REPO_ROOT}"
    --volume /dev:/dev
    --workdir "$CV_STM_REPO_ROOT"
  )

  log "Running yaw smoke test in ${CV_STM_IMAGE}"
  log "Default packet: port=${serial_port}, baud=${baud}, yaw_rad=${yaw_rad}, count=${count}"
  log "Use extra args to override, for example: cv yaw-smoke --yaw-rad -0.05 --count 3"
  log "If the STM stack is running and owns the serial port, stop it first with: cv stop"

  docker_cmd "${docker_args[@]}" "$CV_STM_IMAGE" bash -lc '
    set -euo pipefail
    smoke_bin=/tmp/stm32_yaw_packet_smoke
    g++ -std=c++17 -O2 \
      -I yolo_cpp/include \
      yolo_cpp/tools/stm32_yaw_packet_smoke.cpp \
      yolo_cpp/src/seasky_protocol.cpp \
      -o "$smoke_bin"
    exec "$smoke_bin" "$@"
  ' bash \
    --port "$serial_port" \
    --baud "$baud" \
    --yaw-rad "$yaw_rad" \
    --count "$count" \
    --interval-ms "$interval_ms" \
    "$@"
}

open_shell() {
  ensure_docker
  if ! container_running; then
    die "Container ${UV_CONTAINER_NAME} is not running"
  fi
  docker_cmd exec -it "$UV_CONTAINER_NAME" \
    bash -lc "cd '${CV_STM_REPO_ROOT}' && { source /opt/ros/humble/setup.bash 2>/dev/null || source /opt/ros/humble/install/setup.bash 2>/dev/null || true; }; source install/setup.bash 2>/dev/null || true; exec bash"
}

usage() {
  cat <<EOF
Usage: ./services/stm-stack.sh <command>

Commands:
  install    Install and enable ${SERVICE_NAME}; does not start it.
  run        Run the selected launch profile in the foreground without systemd.
  start      Start ${SERVICE_NAME}.
  stop       Stop ${SERVICE_NAME}.
  restart    Restart ${SERVICE_NAME}.
  preview-on Persist CV_STM_SHOW=true and restart ${SERVICE_NAME}.
  preview-off
             Persist CV_STM_SHOW=false and restart ${SERVICE_NAME}.
  basler-defaults
             Restore field Basler defaults and restart with basler_fps=160, basler_uncapped_fps=false, show=false.
  model-orin  Persist the Orin model and native TensorRT + full-sensor Bayer defaults, then restart ${SERVICE_NAME}.
  model-nano  Legacy alias for model-orin.
  model-agx   Persist the AGX model and DeepStream defaults, then restart ${SERVICE_NAME}.
  status     Show systemd status.
  logs       Follow systemd logs.
  preflight  Validate image, ROS package, model, and selected profile prerequisites.
  health     Check selected-profile ROS nodes and topics inside ${UV_CONTAINER_NAME}.
  yaw-smoke  Compile and send one yaw-only STM packet without a full ROS build.
  shell      Open a shell inside ${UV_CONTAINER_NAME}.
EOF
}

main() {
  local command="${1:-}"
  case "$command" in
    install) install_service ;;
    run) run_foreground ;;
    start) systemctl_action start ;;
    stop) systemctl_action stop ;;
    restart) systemctl_action restart ;;
    preview-on) set_preview true on ;;
    preview-off) set_preview false off ;;
    basler-defaults) restore_basler_launch_defaults ;;
    model-orin|model-nano) set_orin_profile ;;
    model-agx) set_agx_profile ;;
    status) show_status ;;
    logs) show_logs ;;
    preflight) run_preflight ;;
    health) run_health ;;
    yaw-smoke|smoke-yaw|stm-yaw-smoke) shift; run_yaw_smoke "$@" ;;
    shell) open_shell ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
}

main "$@"
