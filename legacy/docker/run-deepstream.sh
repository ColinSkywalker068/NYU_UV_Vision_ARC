#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   IMAGE=cv-realsense:latest VIDEO_DEV=/dev/video0 ./services/run-deepstream.sh
# Defaults:
IMAGE="${IMAGE:-cv-realsense:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-cv-ds}"
VIDEO_DEV="${VIDEO_DEV:-/dev/video0}"
WORKDIR_IN="/home/ultraviolet/CV_Detection"

# Optional display
DISPLAY_FLAGS=()
if [[ -n "${DISPLAY:-}" ]]; then
	# Allow X11
	xhost +local:root >/dev/null 2>&1 || true
	DISPLAY_FLAGS=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
fi

docker run --rm -it --runtime nvidia \
	--name "${CONTAINER_NAME}" \
	--device "${VIDEO_DEV}:${VIDEO_DEV}" \
	--device /dev/nvhost-ctrl \
	--device /dev/nvhost-ctrl-gpu \
	--device /dev/nvhost-prof-gpu \
	--device /dev/nvhost-vic \
	--device /dev/nvmap \
	-v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
	-v /usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/gstreamer-1.0:ro \
	-v "$PWD":"${WORKDIR_IN}" \
	--workdir "${WORKDIR_IN}" \
	--network host --ipc host \
	"${DISPLAY_FLAGS[@]}" \
	"${IMAGE}" /bin/bash


