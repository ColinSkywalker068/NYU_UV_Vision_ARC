import argparse
import logging
import socket
import time
from dataclasses import dataclass
from typing import Optional

try:
    from yolov8_ros.stm32_serial_bridge import (
        SEASKY_COMMAND_VISION_RECV,
        TARGET_STATE_NO_TARGET,
        build_seasky_frame,
        build_vision_flags,
    )
except ModuleNotFoundError:
    from stm32_serial_bridge import (  # type: ignore[import-not-found]
        SEASKY_COMMAND_VISION_RECV,
        TARGET_STATE_NO_TARGET,
        build_seasky_frame,
        build_vision_flags,
    )


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 5000
DEFAULT_PITCH: float = 0.0
DEFAULT_YAW: float = 0.0
DEFAULT_FIRE_MODE: int = 2
DEFAULT_TARGET_STATE: int = TARGET_STATE_NO_TARGET
DEFAULT_TARGET_TYPE: int = 0
DEFAULT_COUNT: int = 1
DEFAULT_RATE_HZ: float = 5.0
DEFAULT_TIMEOUT_SECONDS: float = 3.0


@dataclass(frozen=True)
class VisionPacketSample:
    pitch: float
    yaw: float
    fire_mode: int
    target_state: int
    target_type: int


def build_vision_packet(sample: VisionPacketSample) -> bytes:
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


def parse_positive_rate(rate_hz: float) -> float:
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be greater than 0")
    return rate_hz


def parse_non_negative_count(count: int) -> int:
    if count < 0:
        raise ValueError("count must be greater than or equal to 0")
    return count


def parse_target_state(target_state: int) -> int:
    if target_state < 0 or target_state > 3:
        raise ValueError("target_state must be in range [0, 3]")
    return target_state


def parse_fire_mode(fire_mode: int) -> int:
    if fire_mode < 0 or fire_mode > 3:
        raise ValueError("fire_mode must be in range [0, 3]")
    return fire_mode


def parse_target_type(target_type: int) -> int:
    if target_type < 0 or target_type > 15:
        raise ValueError("target_type must be in range [0, 15]")
    return target_type


def parse_timeout(timeout_seconds: float) -> float:
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be greater than 0")
    return timeout_seconds


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send Seasky vision packets over TCP Ethernet"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Remote TCP host or IP address",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Remote TCP port",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=DEFAULT_PITCH,
        help="Pitch angle in degrees",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=DEFAULT_YAW,
        help="Yaw angle in degrees",
    )
    parser.add_argument(
        "--fire_mode",
        type=int,
        default=DEFAULT_FIRE_MODE,
        help="Vision fire mode bits [0..3]",
    )
    parser.add_argument(
        "--target_state",
        type=int,
        default=DEFAULT_TARGET_STATE,
        help="Vision target state bits [0..3]",
    )
    parser.add_argument(
        "--target_type",
        type=int,
        default=DEFAULT_TARGET_TYPE,
        help="Vision target type bits [0..15]",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of packets to send (0 means infinite)",
    )
    parser.add_argument(
        "--rate_hz",
        type=float,
        default=DEFAULT_RATE_HZ,
        help="Packet send rate in Hz",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="TCP connection timeout in seconds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full packet hex on each send",
    )
    return parser.parse_args()


def send_packets(
    host: str,
    port: int,
    sample: VisionPacketSample,
    count: int,
    rate_hz: float,
    timeout_seconds: float,
    verbose: bool,
) -> None:
    packet_count: int = parse_non_negative_count(count)
    interval_seconds: float = 1.0 / parse_positive_rate(rate_hz)
    connection_timeout_seconds: float = parse_timeout(timeout_seconds)
    packet: bytes = build_vision_packet(sample)
    with socket.create_connection(
        (host, port), timeout=connection_timeout_seconds
    ) as tcp_socket:
        tcp_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        LOGGER.info(
            "Connected to %s:%d. count=%d rate_hz=%.2f payload_bytes=%d",
            host,
            port,
            packet_count,
            rate_hz,
            len(packet),
        )
        message_index: int = 0
        while True:
            if packet_count > 0 and message_index >= packet_count:
                break
            tcp_socket.sendall(packet)
            message_index += 1
            if verbose:
                LOGGER.info(
                    "[TCP TX #%d] pitch=%.6f yaw=%.6f fire_mode=%d target_state=%d target_type=%d hex=%s",
                    message_index,
                    sample.pitch,
                    sample.yaw,
                    sample.fire_mode,
                    sample.target_state,
                    sample.target_type,
                    packet.hex(" "),
                )
            else:
                LOGGER.info(
                    "[TCP TX #%d] pitch=%.6f yaw=%.6f fire_mode=%d target_state=%d target_type=%d",
                    message_index,
                    sample.pitch,
                    sample.yaw,
                    sample.fire_mode,
                    sample.target_state,
                    sample.target_type,
                )
            time.sleep(interval_seconds)
        LOGGER.info("Finished sending %d packet(s)", message_index)


def main() -> None:
    arguments = parse_arguments()
    try:
        sample = VisionPacketSample(
            pitch=float(arguments.pitch),
            yaw=float(arguments.yaw),
            fire_mode=parse_fire_mode(int(arguments.fire_mode)),
            target_state=parse_target_state(int(arguments.target_state)),
            target_type=parse_target_type(int(arguments.target_type)),
        )
        send_packets(
            host=str(arguments.host),
            port=int(arguments.port),
            sample=sample,
            count=int(arguments.count),
            rate_hz=float(arguments.rate_hz),
            timeout_seconds=float(arguments.timeout_seconds),
            verbose=bool(arguments.verbose),
        )
    except KeyboardInterrupt:
        LOGGER.info("TCP sender interrupted by user")
    except (OSError, ValueError) as sender_error:
        LOGGER.error("Failed to send Seasky TCP packet(s): %s", sender_error)


if __name__ == "__main__":
    main()
