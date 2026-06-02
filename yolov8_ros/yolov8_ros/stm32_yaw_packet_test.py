import argparse
import logging
import math
import time
from dataclasses import dataclass

import serial

from .stm32_serial_bridge import (
    DEFAULT_FIRE_MODE,
    DEFAULT_TARGET_TYPE,
    SEASKY_COMMAND_VISION_RECV,
    TARGET_STATE_CONVERGING,
    build_seasky_frame,
    build_vision_flags,
)


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class Options:
    port: str
    baudrate: int
    yaw_rad: float
    count: int
    interval_ms: int
    fire_mode: int
    target_state: int
    target_type: int
    allow_large_yaw: bool


def parse_args() -> Options:
    parser = argparse.ArgumentParser(
        description="Send a yaw-only Seasky vision packet to the STM32"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial device path")
    parser.add_argument(
        "--baud",
        "--baudrate",
        dest="baudrate",
        type=int,
        default=115200,
        help="Serial baudrate",
    )
    parser.add_argument(
        "--yaw-rad",
        type=float,
        default=0.05,
        help="Yaw error in radians. Flip sign if the gimbal moves the wrong way.",
    )
    parser.add_argument("--count", type=int, default=1, help="Packets to send")
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=50,
        help="Delay between packets when count > 1",
    )
    parser.add_argument(
        "--fire-mode",
        type=int,
        default=DEFAULT_FIRE_MODE,
        help="Seasky fire mode bits",
    )
    parser.add_argument(
        "--target-state",
        type=int,
        default=TARGET_STATE_CONVERGING,
        help="1=converging, 2=ready-to-fire",
    )
    parser.add_argument(
        "--target-type",
        type=int,
        default=DEFAULT_TARGET_TYPE,
        help="Controls target type enum value",
    )
    parser.add_argument(
        "--allow-large-yaw",
        action="store_true",
        help="Allow |yaw| > 0.35 rad",
    )
    parsed = parser.parse_args()

    if parsed.count < 1:
        parser.error("--count must be at least 1")
    if parsed.interval_ms < 0:
        parser.error("--interval-ms must be non-negative")
    if not parsed.allow_large_yaw and math.fabs(parsed.yaw_rad) > 0.35:
        parser.error("refusing |yaw| > 0.35 rad without --allow-large-yaw")

    return Options(
        port=str(parsed.port),
        baudrate=int(parsed.baudrate),
        yaw_rad=float(parsed.yaw_rad),
        count=int(parsed.count),
        interval_ms=int(parsed.interval_ms),
        fire_mode=int(parsed.fire_mode),
        target_state=int(parsed.target_state),
        target_type=int(parsed.target_type),
        allow_large_yaw=bool(parsed.allow_large_yaw),
    )


def build_yaw_packet(options: Options) -> tuple[int, bytes]:
    flags = build_vision_flags(
        fire_mode=options.fire_mode,
        target_state=options.target_state,
        target_type=options.target_type,
    )
    packet = build_seasky_frame(
        command_id=SEASKY_COMMAND_VISION_RECV,
        flags_register=flags,
        float_payload=[0.0, options.yaw_rad],
    )
    return flags, packet


def main() -> None:
    options = parse_args()
    flags, packet = build_yaw_packet(options)
    LOGGER.info(
        "Sending yaw-only STM packet: port=%s baud=%d pitch_rad=0 yaw_rad=%.6f "
        "count=%d flags=0x%04X packet[%d]=%s",
        options.port,
        options.baudrate,
        options.yaw_rad,
        options.count,
        flags,
        len(packet),
        packet.hex(" "),
    )

    try:
        with serial.Serial(
            port=options.port,
            baudrate=options.baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=0.5,
        ) as serial_device:
            for packet_index in range(options.count):
                serial_device.write(packet)
                serial_device.flush()
                if packet_index + 1 < options.count:
                    time.sleep(options.interval_ms / 1000.0)
    except serial.SerialException as serial_error:
        LOGGER.error("Failed to use serial device: %s", serial_error)
        raise SystemExit(1) from serial_error

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
