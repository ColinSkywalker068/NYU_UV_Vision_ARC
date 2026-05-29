#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the older setup command.
# The operator CLI owns install/start/status/log behavior now.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/stm-stack.sh" install
"${SCRIPT_DIR}/stm-stack.sh" restart
"${SCRIPT_DIR}/stm-stack.sh" logs
