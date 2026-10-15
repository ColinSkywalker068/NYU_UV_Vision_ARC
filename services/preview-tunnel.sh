#!/usr/bin/env bash
# Run on the host device, not on the Jetson. This does not start or restart CV.
set -euo pipefail
host="${1:-ultraviolet@192.168.55.1}"
local_port="${2:-18080}"
remote_port="${3:-8080}"
for port in "$local_port" "$remote_port"; do
  [[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || {
    echo "Ports must be between 1 and 65535" >&2; exit 2;
  }
done
printf 'Preview: http://127.0.0.1:%s/\nKeep this terminal open; rerun after USB/network disconnect. Ctrl+C closes only the tunnel.\n' "$local_port"
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=10 \
  -o ServerAliveCountMax=3 -o ControlMaster=no -o ControlPath=none \
  -L "127.0.0.1:${local_port}:127.0.0.1:${remote_port}" "$host"
