#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SERVICE_NAME="${CV_STM_SERVICE_NAME:-stm-stack.service}"
SERVICE_SRC="${SCRIPT_DIR}/${SERVICE_NAME}"
SERVICE_DEST="/etc/systemd/system/${SERVICE_NAME}"
ENV_DIR="${CV_STM_ENV_DIR:-/etc/cv-detection}"
ENV_FILE="${CV_STM_ENV_FILE:-${ENV_DIR}/stm-stack.env}"
DEFAULT_IMAGE="nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
fi

CV_STM_IMAGE="${CV_STM_IMAGE:-${DEFAULT_IMAGE}}"
CV_STM_REPO_ROOT="${CV_STM_REPO_ROOT:-/home/ultraviolet/CV_Detection}"
UV_CONTAINER_NAME="${UV_CONTAINER_NAME:-cv-stm-stack}"

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
# For Basler service startup, prefer a Pylon-enabled local image created with:
#   UPDATE_STM_ENV=1 ./Docker/bake_cplusplus_pylon_image.sh
# CV_STM_IMAGE=cv-detection-cplusplus-pylon:local
UV_CONTAINER_NAME=cv-stm-stack
CV_STM_LAUNCH_PROFILE=basler
# CV_STM_MODEL_PATH=/home/ultraviolet/CV_Detection/yolo_cpp/models/best-roboflow-nitish-obb.onnx_b1_gpu0_fp16.engine
# CV_STM_TARGET_COLOR=blue
# Default is to trust model color classes and skip HSV verification.
# Accept both model colors during detector bring-up by setting both false:
# CV_STM_USE_MODEL_COLOR_CLASSES=false
# CV_STM_ENABLE_COLOR_VERIFICATION=false
# CV_STM_SERIAL_PORT=/dev/ttyUSB0
# CV_STM_REQUIRE_SERIAL=true
CV_STM_PREVIEW_MAX_FPS=60.0
# Save up to 100 compressed target frames in pictures/ next to vision_logs/.
# CV_STM_SAVE_TARGET_FRAMES=false
# CV_STM_TARGET_FRAME_SAVE_DIR=
# CV_STM_TARGET_FRAME_SAVE_MAX=100
# CV_STM_TARGET_FRAME_JPEG_QUALITY=85
# RealSense profile skips Basler/DeepStream/Pylon checks and serial by default:
# CV_STM_LAUNCH_PROFILE=realsense
# CV_STM_REQUIRE_SERIAL=false
CV_STM_BASLER_PIXEL_FORMAT=bayer_rggb8
CV_STM_BASLER_FPS=160
CV_STM_BASLER_SOURCE_WIDTH=1920
CV_STM_BASLER_SOURCE_HEIGHT=1200
CV_STM_DEEPSTREAM_PRE_MUX_RESIZE=true
CV_STM_DEEPSTREAM_PRE_MUX_WIDTH=0
CV_STM_DEEPSTREAM_PRE_MUX_HEIGHT=0
CV_STM_DEEPSTREAM_PROFILE_STAGES=true
# CV_STM_DEEPSTREAM_PROFILE_STAGE_INTERVAL=300
CV_STM_DEEPSTREAM_BAYER_CONVERSION_BACKEND=cuda
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
  status     Show systemd status.
  logs       Follow systemd logs.
  preflight  Validate image, ROS package, model, and selected profile prerequisites.
  health     Check selected-profile ROS nodes and topics inside ${UV_CONTAINER_NAME}.
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
    status) show_status ;;
    logs) show_logs ;;
    preflight) run_preflight ;;
    health) run_health ;;
    shell) open_shell ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
}

main "$@"
