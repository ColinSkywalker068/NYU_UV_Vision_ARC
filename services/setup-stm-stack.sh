#!/bin/bash

# Setup and enable the stm-stack.service on the Jetson
# This script copies the systemd file, reloads the daemon, enables the service on boot,
# starts it, and drops into the live log viewer.

set -e

SERVICE_NAME="stm-stack.service"
SERVICE_SRC="$(dirname "$0")/${SERVICE_NAME}"
SERVICE_DEST="/etc/systemd/system/${SERVICE_NAME}"

echo "🚀 Installing ${SERVICE_NAME} to ${SERVICE_DEST}..."
sudo cp "${SERVICE_SRC}" "${SERVICE_DEST}"

echo "🔄 Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "✅ Enabling ${SERVICE_NAME} to start on boot..."
sudo systemctl enable "${SERVICE_NAME}"

echo "▶️ Restarting ${SERVICE_NAME}..."
sudo systemctl restart "${SERVICE_NAME}"

echo ""
echo "==========================================================="
echo "Service is running! Now tracking live logs..."
echo "(Press Ctrl+C to stop viewing logs. The service will keep running.)"
echo "==========================================================="

# Watch the logs live
sudo journalctl -u "${SERVICE_NAME}" -f
