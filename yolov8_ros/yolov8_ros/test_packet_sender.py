import argparse
import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

import serial

try:
    from yolov8_ros.stm32_serial_bridge import (
        SEASKY_COMMAND_VISION_RECV,
        TARGET_STATE_CONVERGING,
        TARGET_STATE_NO_TARGET,
        build_seasky_frame,
        build_vision_flags,
    )
except ModuleNotFoundError:
    from stm32_serial_bridge import (  # type: ignore
        SEASKY_COMMAND_VISION_RECV,
        TARGET_STATE_CONVERGING,
        TARGET_STATE_NO_TARGET,
        build_seasky_frame,
        build_vision_flags,
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
            time.sleep(interval_seconds)


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
