#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE_CONTAINER="uv-nyuultraviolet-cv-dockerfile-jp60-pytorch21-humble-tensorrt-cplusplus"
DEFAULT_BASE_IMAGE="nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus"
DEFAULT_TARGET_IMAGE="cv-detection-cplusplus-pylon:local"

SOURCE_CONTAINER="${1:-${PYLON_SOURCE_CONTAINER:-${DEFAULT_SOURCE_CONTAINER}}}"
TARGET_IMAGE="${2:-${TARGET_IMAGE:-${DEFAULT_TARGET_IMAGE}}}"
BASE_IMAGE="${BASE_IMAGE:-${DEFAULT_BASE_IMAGE}}"
UPDATE_STM_ENV="${UPDATE_STM_ENV:-0}"
ENV_FILE="${CV_STM_ENV_FILE:-/etc/cv-detection/stm-stack.env}"

log() {
  printf '[bake-pylon] %s\n' "$*"
}

die() {
  printf '[bake-pylon] ERROR: %s\n' "$*" >&2
  exit 1
}

run_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command is missing: $1"
}

require_command docker
docker inspect "$SOURCE_CONTAINER" >/dev/null 2>&1 ||
  die "Source container not found: ${SOURCE_CONTAINER}"

copy_from_container() {
  local source_path="$1"
  local dest_path="${TMP_DIR}/rootfs${source_path}"
  mkdir -p "$(dirname "$dest_path")"
  docker cp "${SOURCE_CONTAINER}:${source_path}" "$dest_path"
}

log "Validating source container ${SOURCE_CONTAINER}"
docker exec "$SOURCE_CONTAINER" bash -lc 'test -d /opt/pylon' ||
  die "Source container does not have /opt/pylon"
docker exec "$SOURCE_CONTAINER" bash -lc 'gst-inspect-1.0 pylonsrc >/dev/null' ||
  die "Source container cannot load pylonsrc"

PLUGIN_PATH="$(
  docker exec "$SOURCE_CONTAINER" bash -lc \
    "gst-inspect-1.0 pylonsrc | sed -n 's/^[[:space:]]*Filename[[:space:]]*//p' | head -n 1"
)"
log "pylonsrc plugin path: ${PLUGIN_PATH:-unknown}"

PYLON_DEPENDENCIES="$(
  docker exec "$SOURCE_CONTAINER" bash -lc '
    plugin="$1"
    ldd "$plugin" |
      awk "/=> \// {print \$3} /^\// {print \$1}" |
      grep -Ei "/(libgstpylon|.*pylon.*)\.so" |
      while read -r path; do
        printf "%s\n" "$path"
        realpath -e "$path" 2>/dev/null || true
      done |
      sort -u
  ' _ "$PLUGIN_PATH"
)"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "${TMP_DIR}/rootfs/opt"
docker cp "${SOURCE_CONTAINER}:/opt/pylon" "${TMP_DIR}/rootfs/opt/"

if [ -n "${PLUGIN_PATH}" ] && [[ "${PLUGIN_PATH}" != /opt/pylon/* ]]; then
  copy_from_container "$PLUGIN_PATH"
fi

if [ -n "${PYLON_DEPENDENCIES}" ]; then
  log "Copying Pylon plugin dependencies"
  while IFS= read -r dependency; do
    [ -n "$dependency" ] || continue
    [[ "$dependency" == /opt/pylon/* ]] && continue
    log "dependency: ${dependency}"
    copy_from_container "$dependency"
  done <<< "${PYLON_DEPENDENCIES}"
fi

cat > "${TMP_DIR}/Dockerfile" <<'EOF'
ARG BASE_IMAGE=nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus
FROM ${BASE_IMAGE}

COPY rootfs/ /

ENV PYLON_ROOT=/opt/pylon
ENV PATH=${PATH}:/opt/pylon/bin
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream-6.4/lib/gst-plugins:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/opt/pylon/lib/gstreamer-1.0:/opt/pylon/lib/gst-plugins
ENV GST_PLUGIN_PATH_1_0=/opt/nvidia/deepstream/deepstream-6.4/lib/gst-plugins:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:/usr/lib/aarch64-linux-gnu/gstreamer-1.0:/opt/pylon/lib/gstreamer-1.0:/opt/pylon/lib/gst-plugins
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-6.4/lib:/opt/nvidia/deepstream/deepstream/lib:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:/opt/pylon/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/aarch64-linux-gnu

RUN rm -rf /root/.cache/gstreamer-1.0
CMD ["/bin/bash"]
EOF

log "Building ${TARGET_IMAGE} from ${BASE_IMAGE}"
DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --network=host \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -t "${TARGET_IMAGE}" \
  "${TMP_DIR}"

log "Verifying pylonsrc in ${TARGET_IMAGE}"
docker run --rm --network host --privileged "${TARGET_IMAGE}" \
  bash -lc 'gst-inspect-1.0 pylonsrc >/dev/null'

if [ "${UPDATE_STM_ENV}" = "1" ]; then
  log "Writing CV_STM_IMAGE=${TARGET_IMAGE} to ${ENV_FILE}"
  run_sudo mkdir -p "$(dirname "${ENV_FILE}")"
  if [ ! -f "${ENV_FILE}" ]; then
    printf 'CV_STM_IMAGE=%s\n' "${TARGET_IMAGE}" | run_sudo tee "${ENV_FILE}" >/dev/null
  elif grep -q '^CV_STM_IMAGE=' "${ENV_FILE}"; then
    TMP_ENV="$(mktemp)"
    sed "s|^CV_STM_IMAGE=.*|CV_STM_IMAGE=${TARGET_IMAGE}|" "${ENV_FILE}" > "${TMP_ENV}"
    run_sudo install -m 0644 "${TMP_ENV}" "${ENV_FILE}"
    rm -f "${TMP_ENV}"
  else
    printf 'CV_STM_IMAGE=%s\n' "${TARGET_IMAGE}" | run_sudo tee -a "${ENV_FILE}" >/dev/null
  fi
fi

log "Done: ${TARGET_IMAGE}"
log "Test with: CV_STM_IMAGE=${TARGET_IMAGE} CV_STM_REQUIRE_SERIAL=false ./services/stm-stack.sh preflight"
