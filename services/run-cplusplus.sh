#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT}/start-container.sh" "${IMAGE}" "$@"
