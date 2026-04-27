# fixed angles
# calculate change in yaw change in pitch move from pixels to agnles.
# RX chooses the color
import argparse
import logging
import random
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import serial

CURRENT_FILE_PATH: Path = Path(__file__).resolve()
PACKAGE_ROOT_PATH: Path = CURRENT_FILE_PATH.parents[2]
MODULE_ROOT_PATH: Path = PACKAGE_ROOT_PATH / "yolov8_ros"
if str(PACKAGE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT_PATH))
if str(MODULE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT_PATH))

try:
    from yolov8_ros.stm32_serial_bridge import (
        SEASKY_COMMAND_VISION_RECV,
        TARGET_STATE_CONVERGING,
        TARGET_STATE_NO_TARGET,
        build_seasky_frame,
        build_vision_flags,
        parse_attitude_flags,
        SEASKY_SOF,
        SEASKY_HEADER_SIZE_BYTES,
        SEASKY_TAIL_SIZE_BYTES,
        SEASKY_MIN_DATA_LENGTH,
        SEASKY_MAX_DATA_LENGTH,
        SEASKY_COMMAND_ATTITUDE_SEND,
        SEASKY_ATTITUDE_FLOAT_COUNT,
        SEASKY_FLOAT_SIZE_BYTES,
        compute_crc8,
        compute_crc16,
    )
except ModuleNotFoundError:
    from stm32_serial_bridge import (  # type: ignore
        SEASKY_COMMAND_VISION_RECV,
        TARGET_STATE_CONVERGING,
        TARGET_STATE_NO_TARGET,
        build_seasky_frame,
        build_vision_flags,
        parse_attitude_flags,
        SEASKY_SOF,
        SEASKY_HEADER_SIZE_BYTES,
        SEASKY_TAIL_SIZE_BYTES,
        SEASKY_MIN_DATA_LENGTH,
        SEASKY_MAX_DATA_LENGTH,
        SEASKY_COMMAND_ATTITUDE_SEND,
        SEASKY_ATTITUDE_FLOAT_COUNT,
        SEASKY_FLOAT_SIZE_BYTES,
        compute_crc8,
        compute_crc16,
    )


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_PORT: str = "/dev/ttyUSB0"
DEFAULT_BAUDRATE: int = 115200
DEFAULT_RATE_HZ: float = 20.0
DEFAULT_MESSAGE_COUNT: int = 0
DEFAULT_MIN_ANGLE_DEGREES: float = -30.0
DEFAULT_MAX_ANGLE_DEGREES: float = 30.0


@dataclass(frozen=True)
class VisionTestSample:
    yaw: float
    pitch: float
    fire_mode: int
    target_state: int
    target_type: int


def parse_positive_rate(rate_hz: float) -> float:
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be greater than 0")
    return rate_hz


def generate_random_sample(min_angle: float, max_angle: float) -> VisionTestSample:
    has_target: bool = bool(random.getrandbits(1))
    target_state: int = (
        TARGET_STATE_CONVERGING if has_target else TARGET_STATE_NO_TARGET
    )
    return VisionTestSample(
        yaw=random.uniform(min_angle, max_angle),
        pitch=random.uniform(min_angle, max_angle),
        fire_mode=random.randint(0, 3),
        target_state=target_state,
        target_type=random.randint(0, 15),
    )


def build_packet(sample: VisionTestSample) -> bytes:
    flags_register: int = build_vision_flags(
        fire_mode=sample.fire_mode,
        target_state=sample.target_state,
        target_type=sample.target_type,
    )
    return build_seasky_frame(
        command_id=SEASKY_COMMAND_VISION_RECV,
        flags_register=flags_register,
        float_payload=[float(sample.pitch), float(sample.yaw)],
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send randomized Seasky vision packets to STM32 over serial"
    )
    parser.add_argument(
        "--port", type=str, default=DEFAULT_PORT, help="Serial device path"
    )
    parser.add_argument(
        "--baudrate", type=int, default=DEFAULT_BAUDRATE, help="Serial baudrate"
    )
    parser.add_argument(
        "--rate_hz",
        type=float,
        default=DEFAULT_RATE_HZ,
        help="Packet send rate in Hz",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_MESSAGE_COUNT,
        help="Number of packets to send (0 means infinite)",
    )
    parser.add_argument(
        "--min_angle",
        type=float,
        default=DEFAULT_MIN_ANGLE_DEGREES,
        help="Minimum random angle in degrees",
    )
    parser.add_argument(
        "--max_angle",
        type=float,
        default=DEFAULT_MAX_ANGLE_DEGREES,
        help="Maximum random angle in degrees",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible data",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print packet hex bytes for each sent frame",
    )
    return parser.parse_args()


def drain_attitude_frames(
    rx_buffer: bytearray, verbose: bool, rx_count: list[int]
) -> None:
    while True:
        start_index: int = rx_buffer.find(bytes([SEASKY_SOF]))
        if start_index < 0:
            rx_buffer.clear()
            break
        if start_index > 0:
            del rx_buffer[:start_index]
        if len(rx_buffer) < SEASKY_HEADER_SIZE_BYTES:
            break
        header: bytes = bytes(rx_buffer[:SEASKY_HEADER_SIZE_BYTES])
        if compute_crc8(header[:3]) != header[3]:
            del rx_buffer[0]
            continue
        data_length: int = int(header[1]) | (int(header[2]) << 8)
        if data_length < SEASKY_MIN_DATA_LENGTH or data_length > SEASKY_MAX_DATA_LENGTH:
            del rx_buffer[0]
            continue
        full_frame_size: int = (
            SEASKY_HEADER_SIZE_BYTES + data_length + SEASKY_TAIL_SIZE_BYTES
        )
        if len(rx_buffer) < full_frame_size:
            break
        frame: bytes = bytes(rx_buffer[:full_frame_size])
        if compute_crc16(frame[:-2]) != struct.unpack("<H", frame[-2:])[0]:
            del rx_buffer[0]
            continue
        command_id: int = struct.unpack("<H", frame[4:6])[0]
        flags_register: int = struct.unpack("<H", frame[6:8])[0]
        payload: bytes = frame[8:-2]
        del rx_buffer[:full_frame_size]
        if command_id != SEASKY_COMMAND_ATTITUDE_SEND:
            continue
        expected: int = SEASKY_ATTITUDE_FLOAT_COUNT * SEASKY_FLOAT_SIZE_BYTES
        if len(payload) != expected:
            continue
        yaw, pitch, roll = struct.unpack("<fff", payload)
        enemy_color, work_mode, bullet_speed = parse_attitude_flags(flags_register)
        rx_count[0] += 1
        LOGGER.info(
            "[RX #%d] cmd=0x%04X flags=0x%04X yaw=%.4f pitch=%.4f roll=%.4f "
            "enemy_color=%d work_mode=%d bullet_speed=%d%s",
            rx_count[0],
            command_id,
            flags_register,
            yaw,
            pitch,
            roll,
            enemy_color,
            work_mode,
            bullet_speed,
            f" hex={frame.hex(' ')}" if verbose else "",
        )


def execute_sender(
    port: str,
    baudrate: int,
    rate_hz: float,
    count: int,
    min_angle: float,
    max_angle: float,
    seed: Optional[int],
    verbose: bool,
) -> None:
    if min_angle > max_angle:
        raise ValueError("min_angle must be less than or equal to max_angle")
    interval_seconds: float = 1.0 / parse_positive_rate(rate_hz)
    if count < 0:
        raise ValueError("count must be greater than or equal to 0")
    if seed is not None:
        random.seed(seed)
        LOGGER.info("Using deterministic random seed: %d", seed)
    rx_buffer: bytearray = bytearray()
    rx_count: list[int] = [0]
    with serial.Serial(
        port=port,
        baudrate=baudrate,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1.0,
    ) as serial_device:
        LOGGER.info(
            "Connected to serial device %s @ %d baud. rate_hz=%.2f count=%d",
            port,
            baudrate,
            rate_hz,
            count,
        )
        message_index: int = 0
        while True:
            if count > 0 and message_index >= count:
                break
            sample: VisionTestSample = generate_random_sample(
                min_angle=min_angle, max_angle=max_angle
            )
            packet: bytes = build_packet(sample)
            bytes_written: int = serial_device.write(packet)
            serial_device.flush()
            message_index += 1
            if verbose:
                LOGGER.info(
                    "[TX #%d] bytes=%d yaw=%.3f pitch=%.3f fire_mode=%d target_state=%d target_type=%d hex=%s",
                    message_index,
                    bytes_written,
                    sample.yaw,
                    sample.pitch,
                    sample.fire_mode,
                    sample.target_state,
                    sample.target_type,
                    packet.hex(" "),
                )
            else:
                LOGGER.info(
                    "[TX #%d] yaw=%.3f pitch=%.3f fire_mode=%d target_state=%d target_type=%d",
                    message_index,
                    sample.yaw,
                    sample.pitch,
                    sample.fire_mode,
                    sample.target_state,
                    sample.target_type,
                )
            available: int = serial_device.in_waiting
            if available > 0:
                rx_buffer.extend(serial_device.read(available))
                drain_attitude_frames(rx_buffer, verbose, rx_count)
            time.sleep(interval_seconds)
        time.sleep(0.1)
        available = serial_device.in_waiting
        if available > 0:
            rx_buffer.extend(serial_device.read(available))
            drain_attitude_frames(rx_buffer, verbose, rx_count)
    LOGGER.info("Done. TX=%d RX=%d attitude frames", message_index, rx_count[0])


def main() -> None:
    arguments = parse_arguments()
    try:
        execute_sender(
            port=arguments.port,
            baudrate=int(arguments.baudrate),
            rate_hz=float(arguments.rate_hz),
            count=int(arguments.count),
            min_angle=float(arguments.min_angle),
            max_angle=float(arguments.max_angle),
            seed=arguments.seed,
            verbose=bool(arguments.verbose),
        )
    except KeyboardInterrupt:
        LOGGER.info("Sender interrupted by user")
    except serial.SerialException as serial_error:
        LOGGER.error("Failed to use serial device: %s", serial_error)
    except ValueError as value_error:
        LOGGER.error("Invalid sender configuration: %s", value_error)


if __name__ == "__main__":
    main()
