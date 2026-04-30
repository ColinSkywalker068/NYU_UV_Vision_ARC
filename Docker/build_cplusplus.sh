#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

IMAGE_REPO="${1:-nyuultraviolet/cv-dockerfile}"
BASE_TAG="${2:-jp60-pytorch21-humble-tensorrt-realsense-wrapper-ultra}"
TARGET_TAG="${3:-jp60-pytorch21-humble-tensorrt-cplusplus}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.cplusplus}"

BASE_IMAGE="${IMAGE_REPO}:${BASE_TAG}"
TARGET_IMAGE="${IMAGE_REPO}:${TARGET_TAG}"

echo "Building ${TARGET_IMAGE} from ${BASE_IMAGE}..."
if ! DOCKER_BUILDKIT=1 docker build \
  --progress=plain \
  --network=host \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -f "${DOCKERFILE}" \
  -t "${TARGET_IMAGE}" \
  .; then
  echo "Primary build failed. Retrying once with --no-cache..."
  DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    --network=host \
    --no-cache \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -f "${DOCKERFILE}" \
    -t "${TARGET_IMAGE}" \
    .
fi

echo "Done: ${TARGET_IMAGE}"
