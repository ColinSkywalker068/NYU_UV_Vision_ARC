#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  hardware_serial_smoke.sh [options]

Open a Type-C board serial port and send valid SP VisionToGimbal frames.
Defaults are safe: mode=no-target, yaw=0, pitch=0.

Options:
  --port PATH             Serial device. Default scans /dev/ttyACM0-3.
  --baud RATE             Baud rate. Default: AUTOAIM_SERIAL_BAUDRATE or 115200.
  --mode no-target|aim|fire|0|1|2
                          Raw SP mode: 0=no target, 1=auto aim, 2=auto fire.
  --yaw-rad VALUE         Yaw command in radians. Default: 0.0.
  --pitch-rad VALUE       Pitch command in radians. Default: 0.0.
  --count N               Frames to send. Default: 10.
  --interval-ms N         Delay between frames. Default: 50.
  --read-window-ms N      RX read window after each write. Default: 20.
  --allow-large-angle     Allow |yaw| or |pitch| > 0.35 rad.
  -h, --help              Show this help.

Examples:
  ./yolo_cpp/tools/hardware_serial_smoke.sh --port /dev/ttyACM0
  ./yolo_cpp/tools/hardware_serial_smoke.sh --port /dev/ttyACM0 --mode aim --yaw-rad 0.05
EOF
}

PORT=""
BAUDRATE="${AUTOAIM_SERIAL_BAUDRATE:-115200}"
MODE="no-target"
YAW_RAD="0.0"
PITCH_RAD="0.0"
COUNT="10"
INTERVAL_MS="50"
READ_WINDOW_MS="20"
ALLOW_LARGE_ANGLE="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --baud)
      BAUDRATE="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --yaw-rad)
      YAW_RAD="${2:-}"
      shift 2
      ;;
    --pitch-rad)
      PITCH_RAD="${2:-}"
      shift 2
      ;;
    --count)
      COUNT="${2:-}"
      shift 2
      ;;
    --interval-ms)
      INTERVAL_MS="${2:-}"
      shift 2
      ;;
    --read-window-ms)
      READ_WINDOW_MS="${2:-}"
      shift 2
      ;;
    --allow-large-angle)
      ALLOW_LARGE_ANGLE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'hardware_serial_smoke.sh: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [ -z "$PORT" ]; then
  for candidate in /dev/ttyACM0 /dev/ttyACM1 /dev/ttyACM2 /dev/ttyACM3; do
    if [ -e "$candidate" ]; then
      PORT="$candidate"
      break
    fi
  done
fi

if [ -z "$PORT" ] || [ ! -e "$PORT" ]; then
  echo "No serial device found; tried /dev/ttyACM0-3. Pass --port PATH." >&2
  exit 2
fi

command -v python3 >/dev/null 2>&1 || {
  echo "python3 is required for this hardware smoke test" >&2
  exit 3
}

python3 - <<'PY' "$PORT" "$BAUDRATE" "$MODE" "$YAW_RAD" "$PITCH_RAD" "$COUNT" "$INTERVAL_MS" "$READ_WINDOW_MS" "$ALLOW_LARGE_ANGLE"
import argparse
import os
import struct
import sys
import termios
import time

port, baudrate_s, mode_s, yaw_s, pitch_s, count_s, interval_s, read_window_s, allow_large_s = sys.argv[1:]

BAUDS = {
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
}
for optional in (230400, 460800, 921600):
    attr = f"B{optional}"
    if hasattr(termios, attr):
        BAUDS[optional] = getattr(termios, attr)

MODE_MAP = {
    "no-target": 0,
    "none": 0,
    "0": 0,
    "aim": 1,
    "auto-aim": 1,
    "1": 1,
    "fire": 2,
    "auto-fire": 2,
    "2": 2,
}


def parse_int(value: str, name: str) -> int:
    try:
        return int(value, 0)
    except ValueError as exc:
        raise SystemExit(f"Invalid {name}: {value}") from exc


def parse_float(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid {name}: {value}") from exc


def crc16_sp(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0x8408
            else:
                crc >>= 1
            crc &= 0xFFFF
    return crc


def build_sp_frame(mode: int, yaw: float, pitch: float) -> bytes:
    frame = bytearray(b"SP")
    frame.append(mode & 0xFF)
    for value in (yaw, 0.0, 0.0, pitch, 0.0, 0.0):
        frame.extend(struct.pack("<f", value))
    frame.extend(struct.pack("<H", crc16_sp(frame)))
    if len(frame) != 29:
        raise RuntimeError(f"internal SP frame size bug: {len(frame)}")
    return bytes(frame)


def hex_bytes(data: bytes) -> str:
    return " ".join(f"{byte:02x}" for byte in data)


def configure_serial(fd: int, baudrate: int) -> None:
    if baudrate not in BAUDS:
        raise SystemExit(f"Unsupported baud rate {baudrate}; supported: {sorted(BAUDS)}")
    tty = termios.tcgetattr(fd)
    tty[0] = termios.IGNBRK
    tty[1] = 0
    tty[2] = (tty[2] & ~termios.CSIZE) | termios.CS8 | termios.CLOCAL | termios.CREAD
    tty[2] &= ~termios.PARENB
    tty[2] &= ~termios.CSTOPB
    if hasattr(termios, "CRTSCTS"):
        tty[2] &= ~termios.CRTSCTS
    tty[3] = 0
    tty[4] = BAUDS[baudrate]
    tty[5] = BAUDS[baudrate]
    tty[6][termios.VMIN] = 0
    tty[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, tty)


def decode_rx(data: bytes) -> list[str]:
    lines = []
    i = 0
    while i + 2 <= len(data):
        if data[i:i + 1] != b"S":
            i += 1
            continue
        kind = data[i + 1:i + 2]
        if kind == b"P" and i + 43 <= len(data):
            frame = data[i:i + 43]
            ok = crc16_sp(frame[:-2]) == struct.unpack_from("<H", frame, 41)[0]
            yaw = struct.unpack_from("<f", frame, 19)[0]
            pitch = struct.unpack_from("<f", frame, 27)[0]
            bullet_speed = struct.unpack_from("<f", frame, 35)[0]
            lines.append(
                f"RX SP GimbalToVision crc_ok={int(ok)} mode={frame[2]} "
                f"yaw={yaw:.6f} pitch={pitch:.6f} bullet_speed={bullet_speed:.2f}"
            )
            i += 43
        elif kind == b"T" and i + 39 <= len(data):
            frame = data[i:i + 39]
            ok = crc16_sp(frame[:-2]) == struct.unpack_from("<H", frame, 37)[0]
            robot_id = frame[30]
            bullet_speed = struct.unpack_from("<H", frame, 33)[0]
            team_color = frame[35]
            lines.append(
                f"RX ST SentryTelemetry crc_ok={int(ok)} robot_id={robot_id} "
                f"bullet_speed={bullet_speed} team_color={team_color}"
            )
            i += 39
        else:
            i += 1
    return lines


baudrate = parse_int(baudrate_s, "baud")
count = parse_int(count_s, "count")
interval_ms = parse_int(interval_s, "interval-ms")
read_window_ms = parse_int(read_window_s, "read-window-ms")
yaw = parse_float(yaw_s, "yaw-rad")
pitch = parse_float(pitch_s, "pitch-rad")
allow_large = allow_large_s == "1"

mode_key = mode_s.strip().lower()
if mode_key not in MODE_MAP:
    raise SystemExit(f"Invalid mode {mode_s!r}; use no-target, aim, or fire")
mode = MODE_MAP[mode_key]

if count < 1:
    raise SystemExit("count must be at least 1")
if interval_ms < 0 or read_window_ms < 0:
    raise SystemExit("interval-ms and read-window-ms must be non-negative")
if not allow_large and (abs(yaw) > 0.35 or abs(pitch) > 0.35):
    raise SystemExit("Refusing angle magnitude > 0.35 rad without --allow-large-angle")

packet = build_sp_frame(mode, yaw, pitch)
print(
    f"opening {port} @ {baudrate}; mode={mode} yaw_rad={yaw:.6f} "
    f"pitch_rad={pitch:.6f} count={count}"
)
print(f"SP TX frame[{len(packet)}]={hex_bytes(packet)}")

fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
rx = bytearray()
try:
    configure_serial(fd, baudrate)
    for index in range(count):
        written = os.write(fd, packet)
        termios.tcdrain(fd)
        print(f"write {index + 1}/{count}: {written} bytes")

        deadline = time.monotonic() + (read_window_ms / 1000.0)
        while time.monotonic() < deadline:
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                chunk = b""
            if chunk:
                rx.extend(chunk)
            else:
                time.sleep(0.002)

        if index + 1 < count:
            time.sleep(interval_ms / 1000.0)
finally:
    os.close(fd)

if rx:
    print(f"RX raw[{len(rx)}]={hex_bytes(rx)}")
    for line in decode_rx(bytes(rx)):
        print(line)
else:
    print("RX raw[0]=<none>")
PY
