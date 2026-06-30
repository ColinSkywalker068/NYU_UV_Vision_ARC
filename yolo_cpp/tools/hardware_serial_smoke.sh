#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-}"
BAUDRATE="${AUTOAIM_SERIAL_BAUDRATE:-115200}"
COUNT="${2:-10}"

if [ -z "$PORT" ]; then
  for candidate in /dev/ttyACM0 /dev/ttyACM1 /dev/ttyACM2 /dev/ttyACM3; do
    if [ -e "$candidate" ]; then
      PORT="$candidate"
      break
    fi
  done
fi

if [ -z "$PORT" ] || [ ! -e "$PORT" ]; then
  echo "No serial device found; tried /dev/ttyACM0-3" >&2
  exit 2
fi

if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY' "$PORT" "$BAUDRATE" "$COUNT"
import os, sys, time
port, baudrate, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
print(f"opening {port} @ {baudrate} for {count} writes")
fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
try:
    import termios
    tty = termios.tcgetattr(fd)
    tty[0] = termios.IGNBRK
    tty[2] &= ~termios.CSTOPB
    tty[2] &= ~termios.PARENB
    tty[2] &= ~termios.CRTSCTS
    tty[2] |= termios.CREAD | termios.CLOCAL
    tty[3] &= ~termios.ICANON
    tty[3] &= ~termios.ECHO
    tty[3] &= ~termios.ISIG
    tty[5] = b'\x00'
    tty[6] = b'\x00'
    termios.tcsetattr(fd, termios.TCSANOW, tty)
    for i in range(count):
        payload = b'\x7e\x00\x00\x00\x00\x00\x00'
        os.write(fd, payload)
        time.sleep(0.05)
        print(f"write {i+1}/{count}")
finally:
    os.close(fd)
PY
else
  echo "python3 is required for this hardware smoke test" >&2
  exit 3
fi
