#!/usr/bin/env bash
# Regression check: preview helpers must keep operator edit access and safety gates.
# All service actions are mocked; only a temporary configuration is changed.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT
mkdir "$TEST_DIR/bin"
printf '#!/bin/sh\nexec "$@"\n' > "$TEST_DIR/bin/sudo"
printf '#!/bin/sh\nexit 0\n' > "$TEST_DIR/bin/systemctl"
chmod +x "$TEST_DIR/bin/"*
cp "$REPO_ROOT/services/stm-stack.env.example" "$TEST_DIR/config.env"
chmod 0660 "$TEST_DIR/config.env"
BEFORE_OWNER="$(stat -c %u:%g "$TEST_DIR/config.env")"
for mode in preview-off preview-web; do
  PATH="$TEST_DIR/bin:$PATH" CV_STM_ENV_FILE="$TEST_DIR/config.env" \
    CV_STM_ENV_DIR="$TEST_DIR" "$REPO_ROOT/services/stm-stack.sh" "$mode"
  [ "$(stat -c %a "$TEST_DIR/config.env")" = 660 ]
  [ "$(stat -c %u:%g "$TEST_DIR/config.env")" = "$BEFORE_OWNER" ]
  grep -q 'transmit_enable:=false' "$TEST_DIR/config.env"
  grep -q 'enable_fire_recommendation:=false' "$TEST_DIR/config.env"
done
grep -q '^CV_STM_SHOW=web$' "$TEST_DIR/config.env"
printf 'PASS: preview helpers preserve edit permissions and disabled TX/fire settings\n'
