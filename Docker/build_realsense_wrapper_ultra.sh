#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

IMAGE_REPO="${1:-nyuultraviolet/cv-dockerfile}"
BASE_TAG="${2:-jp60-pytorch21-humble-tensorrt-realsense-wrapper}"
TARGET_TAG="${3:-jp60-pytorch21-humble-tensorrt-realsense-wrapper-ultra}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.realsense-wrapper-ultra}"

BASE_IMAGE="${IMAGE_REPO}:${BASE_TAG}"
TARGET_IMAGE="${IMAGE_REPO}:${TARGET_TAG}"

echo "Building ${TARGET_IMAGE} from ${BASE_IMAGE}..."
if ! docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -f "${DOCKERFILE}" \
  -t "${TARGET_IMAGE}" \
  .; then
  echo "Primary build failed. Retrying with host networking and BuildKit disabled..."
  DOCKER_BUILDKIT=0 docker build \
    --network=host \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -f "${DOCKERFILE}" \
    -t "${TARGET_IMAGE}" \
    .
fi

echo "Logging into Docker registry..."
docker login

echo "Pushing ${TARGET_IMAGE}..."
docker push "${TARGET_IMAGE}"

echo "Done: ${TARGET_IMAGE}"
