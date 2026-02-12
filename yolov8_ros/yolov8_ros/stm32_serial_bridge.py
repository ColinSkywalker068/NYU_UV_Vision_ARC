import argparse
import logging
import struct
import sys
from typing import Optional

import rclpy
import serial
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32MultiArray


LOGGER = logging.getLogger(__name__)
ANGLE_SCALE_FACTOR: int = 1_000_000
PACKET_HEADER: int = 0xE7
PACKET_LENGTH: int = 11
PACKET_TYPE_AUTO_AIM: int = 1
PACKET_RESERVED_1: int = 0
PACKET_RESERVED_2: int = 0
PACKET_FOOTER: int = 0


class STM32SerialTransmitter:
    """Transmit auto-aim packets to STM32 over serial."""

    def __init__(self, port: str, baudrate: int, verbose: bool) -> None:
        self.port: str = port
        self.baudrate: int = baudrate
        self.verbose: bool = verbose
        self.serial_device: Optional[serial.Serial] = None
        self.is_connected: bool = False
        self.message_count: int = 0
        self.connect_serial()

    def connect_serial(self) -> None:
        try:
            self.serial_device = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1.0,
            )
            self.is_connected = True
            LOGGER.info("STM32 serial connected: %s @ %d baud", self.port, self.baudrate)
        except serial.SerialException as serial_error:
            self.is_connected = False
            LOGGER.warning(
                "Failed to open serial port %s: %s. Bridge stays alive without TX.",
                self.port,
                serial_error,
            )

    def send_auto_aim(self, pitch: float, yaw: float, has_target: bool) -> bool:
        if not self.is_connected or self.serial_device is None:
            return False
        try:
            header: bytes = struct.pack(
                "BHBBH",
                PACKET_HEADER,
                PACKET_LENGTH,
                PACKET_RESERVED_1,
                PACKET_RESERVED_2,
                PACKET_TYPE_AUTO_AIM,
            )
            pitch_integer: int = int(pitch * ANGLE_SCALE_FACTOR)
            yaw_integer: int = int(yaw * ANGLE_SCALE_FACTOR)
            body: bytes = struct.pack("iiB", pitch_integer, yaw_integer, int(has_target))
            footer: bytes = struct.pack("H", PACKET_FOOTER)
            packet: bytes = header + body + footer
            bytes_written: int = self.serial_device.write(packet)
            self.serial_device.flush()
            self.message_count += 1
            if self.verbose:
                LOGGER.info(
                    "[STM32 TX #%d] pitch=%d yaw=%d has_target=%d bytes=%d",
                    self.message_count,
                    pitch_integer,
                    yaw_integer,
                    int(has_target),
                    bytes_written,
                )
            return True
        except serial.SerialException as serial_error:
            self.is_connected = False
            LOGGER.warning("Serial write error: %s", serial_error)
            return False

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
        super().__init__("stm32_serial_bridge_node", parameter_overrides=parameter_overrides)
        self.declare_parameter("transmit_enable", True)
        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("serial_baudrate", 115200)
        self.declare_parameter("serial_verbose", False)
        self.transmit_enable: bool = bool(self.get_parameter("transmit_enable").value)
        self.serial_port: str = str(self.get_parameter("serial_port").value)
        self.serial_baudrate: int = int(self.get_parameter("serial_baudrate").value)
        self.serial_verbose: bool = bool(self.get_parameter("serial_verbose").value)
        self.has_target: bool = False
        self.transmitter: Optional[STM32SerialTransmitter] = None
        if self.transmit_enable:
            self.transmitter = STM32SerialTransmitter(
                port=self.serial_port,
                baudrate=self.serial_baudrate,
                verbose=self.serial_verbose,
            )
        else:
            self.get_logger().info("Transmit disabled. Set transmit_enable:=true to enable TX.")
        self.create_subscription(Bool, "override", self.handle_override, 10)
        self.create_subscription(
            Float32MultiArray, "bounding_box_angles", self.handle_angles, 10
        )
        self.get_logger().info(
            "STM32 serial bridge ready. transmit_enable=%s, port=%s, baud=%d",
            str(self.transmit_enable).lower(),
            self.serial_port,
            self.serial_baudrate,
        )

    def handle_override(self, msg: Bool) -> None:
        self.has_target = bool(msg.data)

    def handle_angles(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 2:
            self.get_logger().warning("Received malformed angle message: %s", str(msg.data))
            return
        if not self.transmit_enable:
            return
        if self.transmitter is None:
            return
        yaw: float = float(msg.data[0])
        pitch: float = float(msg.data[1])
        self.transmitter.send_auto_aim(pitch=pitch, yaw=yaw, has_target=self.has_target)

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
            Parameter("serial_baudrate", Parameter.Type.INTEGER, parsed_args.serial_baudrate)
        )
    if parsed_args.serial_verbose is not None:
        is_serial_verbose: bool = parse_boolean_flag(parsed_args.serial_verbose)
        parameter_overrides.append(
            Parameter("serial_verbose", Parameter.Type.BOOL, is_serial_verbose)
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
