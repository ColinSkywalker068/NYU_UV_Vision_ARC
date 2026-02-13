import argparse
import logging
import struct
import sys
from typing import Optional

import rclpy
import serial
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32MultiArray, Int32MultiArray


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# region Seasky Protocol Constants
# Protocol frame structure
SEASKY_SOF: int = 0xA5
SEASKY_COMMAND_VISION_RECV: int = 0x0001
SEASKY_COMMAND_ATTITUDE_SEND: int = 0x0002
SEASKY_FLAGS_SIZE_BYTES: int = 2
SEASKY_COMMAND_SIZE_BYTES: int = 2
SEASKY_HEADER_SIZE_BYTES: int = 4
SEASKY_TAIL_SIZE_BYTES: int = 2
SEASKY_ATTITUDE_FLOAT_COUNT: int = 3
SEASKY_FLOAT_SIZE_BYTES: int = 4
SEASKY_MIN_DATA_LENGTH: int = 4
SEASKY_MAX_DATA_LENGTH: int = 128
# CRC configuration
SEASKY_CRC8_INIT: int = 0xFF
SEASKY_CRC8_POLY: int = 0x31
SEASKY_CRC16_INIT: int = 0xFFFF
SEASKY_CRC16_POLY: int = 0x1021
# Flag register bit masks and shifts (TX: vision -> control board)
FLAG_FIRE_MODE_MASK: int = 0x0003
FLAG_TARGET_STATE_MASK: int = 0x0003
FLAG_TARGET_STATE_SHIFT: int = 2
FLAG_TARGET_TYPE_MASK: int = 0x000F
FLAG_TARGET_TYPE_SHIFT: int = 4
# Flag register bit masks and shifts (RX: control board -> vision)
FLAG_ENEMY_COLOR_MASK: int = 0x0003
FLAG_WORK_MODE_MASK: int = 0x0003
FLAG_WORK_MODE_SHIFT: int = 2
FLAG_BULLET_SPEED_MASK: int = 0x00FF
FLAG_BULLET_SPEED_SHIFT: int = 4
# Default values
DEFAULT_FIRE_MODE: int = 2
DEFAULT_TARGET_TYPE: int = 0
TARGET_STATE_NO_TARGET: int = 0
TARGET_STATE_CONVERGING: int = 1
DEFAULT_RX_POLL_PERIOD_SECONDS: float = 0.01
# endregion


def compute_crc8(data: bytes) -> int:
    crc: int = SEASKY_CRC8_INIT
    for byte_value in data:
        crc ^= byte_value
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ SEASKY_CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def compute_crc16(data: bytes) -> int:
    crc: int = SEASKY_CRC16_INIT
    for byte_value in data:
        crc ^= byte_value << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ SEASKY_CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_vision_flags(fire_mode: int, target_state: int, target_type: int) -> int:
    return (
        (fire_mode & FLAG_FIRE_MODE_MASK)
        | ((target_state & FLAG_TARGET_STATE_MASK) << FLAG_TARGET_STATE_SHIFT)
        | ((target_type & FLAG_TARGET_TYPE_MASK) << FLAG_TARGET_TYPE_SHIFT)
    )


def parse_attitude_flags(flags_register: int) -> tuple[int, int, int]:
    enemy_color: int = flags_register & FLAG_ENEMY_COLOR_MASK
    work_mode: int = (flags_register >> FLAG_WORK_MODE_SHIFT) & FLAG_WORK_MODE_MASK
    bullet_speed: int = (
        flags_register >> FLAG_BULLET_SPEED_SHIFT
    ) & FLAG_BULLET_SPEED_MASK
    return enemy_color, work_mode, bullet_speed


def build_seasky_frame(
    command_id: int, flags_register: int, float_payload: list[float]
) -> bytes:
    payload_bytes: bytes = b"".join(
        struct.pack("<f", float(value)) for value in float_payload
    )
    data_length: int = (
        SEASKY_COMMAND_SIZE_BYTES + SEASKY_FLAGS_SIZE_BYTES + len(payload_bytes)
    )
    header_without_crc: bytes = bytes(
        [SEASKY_SOF, data_length & 0xFF, (data_length >> 8) & 0xFF]
    )
    header_crc8: int = compute_crc8(header_without_crc)
    header: bytes = header_without_crc + bytes([header_crc8])
    body: bytes = (
        struct.pack("<H", int(command_id))
        + struct.pack("<H", int(flags_register))
        + payload_bytes
    )
    frame_without_crc16: bytes = header + body
    frame_crc16: int = compute_crc16(frame_without_crc16)
    return frame_without_crc16 + struct.pack("<H", frame_crc16)


class STM32SerialTransmitter:
    """Transmit and receive Seasky protocol packets over serial."""

    def __init__(self, port: str, baudrate: int, verbose: bool) -> None:
        self.port: str = port
        self.baudrate: int = baudrate
        self.verbose: bool = verbose
        self.serial_device: Optional[serial.Serial] = None
        self.is_connected: bool = False
        self.message_count: int = 0
        self.rx_buffer: bytearray = bytearray()
        self.connect_serial()

    def connect_serial(self) -> None:
        try:
            self.serial_device = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=0.0,
            )
            self.is_connected = True
            LOGGER.info(
                "STM32 serial connected: %s @ %d baud", self.port, self.baudrate
            )
        except serial.SerialException as serial_error:
            self.is_connected = False
            LOGGER.warning(
                "Failed to open serial port %s: %s. Bridge stays alive without TX.",
                self.port,
                serial_error,
            )

    def send_vision_data(
        self,
        pitch: float,
        yaw: float,
        fire_mode: int,
        target_state: int,
        target_type: int,
    ) -> bool:
        if not self.is_connected or self.serial_device is None:
            return False
        try:
            flags_register: int = build_vision_flags(
                fire_mode=fire_mode, target_state=target_state, target_type=target_type
            )
            packet: bytes = build_seasky_frame(
                command_id=SEASKY_COMMAND_VISION_RECV,
                flags_register=flags_register,
                float_payload=[float(pitch), float(yaw)],
            )
            bytes_written: int = self.serial_device.write(packet)
            self.serial_device.flush()
            self.message_count += 1
            # DEBUG: TX logging disabled; uncomment below to log each sent packet.
            # if self.verbose:
            #     LOGGER.info(
            #         "[Seasky TX #%d] cmd=0x%04X flags=0x%04X pitch=%.6f yaw=%.6f bytes=%d hex=%s",
            #         self.message_count, SEASKY_COMMAND_VISION_RECV, flags_register,
            #         float(pitch), float(yaw), bytes_written, packet.hex(" "),
            #     )
            return True
        except serial.SerialException as serial_error:
            self.is_connected = False
            LOGGER.warning("Serial write error: %s", serial_error)
            return False
        except Exception as packet_error:
            LOGGER.warning("Packet build/send error: %s", packet_error)
            return False

    def read_attitude_frames(self) -> list[dict[str, int | float]]:
        if not self.is_connected or self.serial_device is None:
            return []
        available_bytes: int = int(self.serial_device.in_waiting)
        if available_bytes > 0:
            self.rx_buffer.extend(self.serial_device.read(available_bytes))
        parsed_frames: list[dict[str, int | float]] = []
        while True:
            start_index: int = self.rx_buffer.find(bytes([SEASKY_SOF]))
            if start_index < 0:
                self.rx_buffer.clear()
                break
            if start_index > 0:
                del self.rx_buffer[:start_index]
            if len(self.rx_buffer) < SEASKY_HEADER_SIZE_BYTES:
                break
            header: bytes = bytes(self.rx_buffer[:SEASKY_HEADER_SIZE_BYTES])
            header_crc8: int = compute_crc8(header[:3])
            if header_crc8 != header[3]:
                del self.rx_buffer[0]
                continue
            data_length: int = int(header[1]) | (int(header[2]) << 8)
            if (
                data_length < SEASKY_MIN_DATA_LENGTH
                or data_length > SEASKY_MAX_DATA_LENGTH
            ):
                del self.rx_buffer[0]
                continue
            full_frame_size: int = (
                SEASKY_HEADER_SIZE_BYTES + data_length + SEASKY_TAIL_SIZE_BYTES
            )
            if len(self.rx_buffer) < full_frame_size:
                break
            frame: bytes = bytes(self.rx_buffer[:full_frame_size])
            calculated_crc16: int = compute_crc16(frame[:-2])
            received_crc16: int = struct.unpack("<H", frame[-2:])[0]
            if calculated_crc16 != received_crc16:
                del self.rx_buffer[0]
                continue
            command_id: int = struct.unpack("<H", frame[4:6])[0]
            flags_register: int = struct.unpack("<H", frame[6:8])[0]
            payload: bytes = frame[8:-2]
            if command_id == SEASKY_COMMAND_ATTITUDE_SEND:
                expected_payload_size: int = (
                    SEASKY_ATTITUDE_FLOAT_COUNT * SEASKY_FLOAT_SIZE_BYTES
                )
                if len(payload) == expected_payload_size:
                    yaw, pitch, roll = struct.unpack("<fff", payload)
                    enemy_color, work_mode, bullet_speed = parse_attitude_flags(
                        flags_register
                    )
                    parsed_frames.append(
                        {
                            "yaw": float(yaw),
                            "pitch": float(pitch),
                            "roll": float(roll),
                            "enemy_color": int(enemy_color),
                            "work_mode": int(work_mode),
                            "bullet_speed": int(bullet_speed),
                            "flags_register": int(flags_register),
                            "command_id": int(command_id),
                        }
                    )
                    # DEBUG: RX always logged; to gate behind serial_verbose, wrap in: if self.verbose:
                    LOGGER.info(
                        "[Seasky RX] cmd=0x%04X flags=0x%04X yaw=%.6f pitch=%.6f roll=%.6f",
                        command_id,
                        flags_register,
                        float(yaw),
                        float(pitch),
                        float(roll),
                    )
            del self.rx_buffer[:full_frame_size]
        return parsed_frames

    def close(self) -> None:
        if self.serial_device is None:
            return
        if not self.serial_device.is_open:
            return
        self.serial_device.close()
        self.is_connected = False
        LOGGER.info("STM32 serial closed after %d messages", self.message_count)


class STM32SerialBridgeNode(Node):
    """Subscribe to YOLO topics and forward data to STM32 serial."""

    def __init__(self, parameter_overrides: Optional[list[Parameter]] = None) -> None:
        super().__init__(
            "stm32_serial_bridge_node", parameter_overrides=parameter_overrides
        )
        self.declare_parameter("transmit_enable", True)
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("serial_baudrate", 115200)
        self.declare_parameter("serial_verbose", False)
        self.transmit_enable: bool = bool(self.get_parameter("transmit_enable").value)
        self.serial_port: str = str(self.get_parameter("serial_port").value)
        self.serial_baudrate: int = int(self.get_parameter("serial_baudrate").value)
        self.serial_verbose: bool = bool(self.get_parameter("serial_verbose").value)
        self.declare_parameter("fire_mode", DEFAULT_FIRE_MODE)
        self.declare_parameter("target_type", DEFAULT_TARGET_TYPE)
        self.fire_mode: int = int(self.get_parameter("fire_mode").value)
        self.target_type: int = int(self.get_parameter("target_type").value)
        self.target_state: int = TARGET_STATE_NO_TARGET
        self.has_target: bool = False
        self.transmitter: Optional[STM32SerialTransmitter] = None
        if self.transmit_enable:
            self.transmitter = STM32SerialTransmitter(
                port=self.serial_port,
                baudrate=self.serial_baudrate,
                verbose=self.serial_verbose,
            )
        else:
            self.get_logger().info(
                "Transmit disabled. Set transmit_enable:=true to enable TX."
            )
        self.create_subscription(Bool, "override", self.handle_override, 10)
        self.create_subscription(
            Float32MultiArray, "bounding_box_angles", self.handle_angles, 10
        )
        self.attitude_publisher = self.create_publisher(
            Float32MultiArray, "attitude_data", 10
        )
        self.attitude_flags_publisher = self.create_publisher(
            Int32MultiArray, "attitude_flags", 10
        )
        self.rx_timer = self.create_timer(
            DEFAULT_RX_POLL_PERIOD_SECONDS, self.handle_rx_poll
        )
        self.get_logger().info(
            "STM32 serial bridge ready. transmit_enable=%s, port=%s, baud=%d"
            % (
                str(self.transmit_enable).lower(),
                self.serial_port,
                self.serial_baudrate,
            )
        )

    def handle_override(self, msg: Bool) -> None:
        self.has_target = bool(msg.data)
        self.target_state = (
            TARGET_STATE_CONVERGING if self.has_target else TARGET_STATE_NO_TARGET
        )

    def handle_angles(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            self.get_logger().warning(
                f"Received malformed angle message: {str(msg.data)}"
            )
            return
        if not self.transmit_enable:
            return
        if self.transmitter is None:
            return
        yaw: float = float(msg.data[0])
        pitch: float = float(msg.data[1])
        self.transmitter.send_vision_data(
            pitch=pitch,
            yaw=yaw,
            fire_mode=self.fire_mode,
            target_state=self.target_state,
            target_type=self.target_type,
        )

    def handle_rx_poll(self) -> None:
        if not self.transmit_enable:
            return
        if self.transmitter is None:
            return
        parsed_frames: list[
            dict[str, int | float]
        ] = self.transmitter.read_attitude_frames()
        for frame_data in parsed_frames:
            attitude_msg = Float32MultiArray()
            attitude_msg.data = [
                float(frame_data["yaw"]),
                float(frame_data["pitch"]),
                float(frame_data["roll"]),
            ]
            self.attitude_publisher.publish(attitude_msg)
            flags_msg = Int32MultiArray()
            flags_msg.data = [
                int(frame_data["enemy_color"]),
                int(frame_data["work_mode"]),
                int(frame_data["bullet_speed"]),
                int(frame_data["flags_register"]),
                int(frame_data["command_id"]),
            ]
            self.attitude_flags_publisher.publish(flags_msg)

    def destroy_node(self) -> bool:
        if self.transmitter is not None:
            self.transmitter.close()
        return super().destroy_node()


def parse_boolean_flag(flag_value: str) -> bool:
    normalized_value: str = flag_value.strip().lower()
    return normalized_value in ("1", "true", "yes", "on")


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="STM32 serial bridge node")
    parser.add_argument(
        "--enable_transmit",
        type=str,
        default=None,
        help="Enable serial transmit (true/false)",
    )
    parser.add_argument(
        "--serial_port",
        type=str,
        default=None,
        help="Serial device path, e.g. /dev/ttyUSB0",
    )
    parser.add_argument(
        "--serial_baudrate",
        type=int,
        default=None,
        help="Serial baudrate, e.g. 115200",
    )
    parser.add_argument(
        "--serial_verbose",
        type=str,
        default=None,
        help="Verbose transmit logging (true/false)",
    )
    parser.add_argument(
        "--fire_mode",
        type=int,
        default=None,
        help="Vision fire mode bits [0..3] for Seasky cmd 0x0001",
    )
    parser.add_argument(
        "--target_type",
        type=int,
        default=None,
        help="Vision target type bits [0..15] for Seasky cmd 0x0001",
    )
    source_args: list[str] = args if args is not None else sys.argv[1:]
    parsed_args, remaining_args = parser.parse_known_args(source_args)
    rclpy.init(args=remaining_args)
    parameter_overrides: list[Parameter] = []
    if parsed_args.enable_transmit is not None:
        is_transmit_enabled: bool = parse_boolean_flag(parsed_args.enable_transmit)
        parameter_overrides.append(
            Parameter("transmit_enable", Parameter.Type.BOOL, is_transmit_enabled)
        )
    if parsed_args.serial_port is not None:
        parameter_overrides.append(
            Parameter("serial_port", Parameter.Type.STRING, parsed_args.serial_port)
        )
    if parsed_args.serial_baudrate is not None:
        parameter_overrides.append(
            Parameter(
                "serial_baudrate", Parameter.Type.INTEGER, parsed_args.serial_baudrate
            )
        )
    if parsed_args.serial_verbose is not None:
        is_serial_verbose: bool = parse_boolean_flag(parsed_args.serial_verbose)
        parameter_overrides.append(
            Parameter("serial_verbose", Parameter.Type.BOOL, is_serial_verbose)
        )
    if parsed_args.fire_mode is not None:
        parameter_overrides.append(
            Parameter("fire_mode", Parameter.Type.INTEGER, parsed_args.fire_mode)
        )
    if parsed_args.target_type is not None:
        parameter_overrides.append(
            Parameter("target_type", Parameter.Type.INTEGER, parsed_args.target_type)
        )
    node = STM32SerialBridgeNode(parameter_overrides=parameter_overrides)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
