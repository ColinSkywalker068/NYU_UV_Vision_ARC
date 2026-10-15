#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
package_dir="${repo_root}/yolo_cpp"

output="$(
  cd "${package_dir}"
  CV_DOCKER_DRY_RUN=1 script -qec "${repo_root}/services/uv.sh build yolo_cpp" /dev/null
)"

expected="cd\\ ${repo_root}\\;\\ uv\\ build\\ yolo_cpp"
if [[ "${output}" != *"${expected}"* ]]; then
  printf 'Expected uv to restart from the workspace root. Output was:\n%s\n' "${output}" >&2
  exit 1
fi

r39_output="$(
  cd "${repo_root}"
  CV_DOCKER_DRY_RUN=1 script -qec "${repo_root}/services/uv.sh build yolo_cpp" /dev/null
)"
if [[ "${r39_output}" != *"cv-detection-cplusplus-r39:local"* ]]; then
  printf 'Expected an r39 host to select the local r39 image. Output was:\n%s\n' \
    "${r39_output}" >&2
  exit 1
fi
