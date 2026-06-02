#!/usr/bin/env python3
"""
STM32 Serial Communication Module
Sends pitch/yaw angles and target status to STM32 microcontroller via USB-C serial.
"""

try:
    import serial
    import struct
    import logging
    from typing import Optional

    # Verify serial module has required attributes
    if not hasattr(serial, "Serial"):
        raise ImportError(
            "serial module does not have 'Serial' class. Wrong module imported?"
        )
    if not hasattr(serial, "SerialException"):
        raise ImportError(
            "serial module does not have 'SerialException'. Is pyserial installed correctly?"
        )
except ImportError as e:
    print(f"ERROR: Failed to import required modules for STM32Serial: {e}")
    print("Please install pyserial: pip3 install pyserial")
    raise

logger = logging.getLogger(__name__)


class STM32Serial:
    """Serial communication class for STM32 microcontroller."""

    def __init__(
        self, port: str = "/dev/serial", baudrate: int = 115200, verbose: bool = False
    ):
        """
        Initialize STM32 serial connection.

        Args:
            port: Serial port path (e.g., "/dev/serial", "/dev/ttyUSB0", "/dev/ttyACM0")
            baudrate: Serial baud rate (default: 115200)
            verbose: Enable verbose logging of transmitted data (default: False)
        """
        self.port = port
        self.baudrate = baudrate
        self.verbose = verbose
        self.ser: Optional[serial.Serial] = None
        self.HEADER = 0xE7
        self.connected = False
        self.message_count = 0
        self.last_sent_data = None

        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=1.0,
            )
            self.connected = True
            logger.info(f"STM32 serial connected: {port} @ {baudrate} baud")
        except serial.SerialException as e:
            logger.warning(f"Failed to open serial port {port}: {e}")
            logger.warning("Continuing without serial communication")
            self.connected = False
        except Exception as e:
            logger.error(f"Unexpected error opening serial port: {e}")
            self.connected = False

    def send_auto_aim(self, pitch: float, yaw: float, has_target: int) -> bool:
        """
        Send auto-aim data to STM32.

        Args:
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            has_target: 1 if target detected, 0 otherwise

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.connected or self.ser is None:
            if self.verbose:
                logger.debug("STM32 serial not connected, skipping send")
            return False

        try:
            # Packet format matching legacy uart_comms.py
            # msg = [pitch, yaw, has_target]
            length = 11
            msg_type = 1

            # Header: struct.pack("BHBBH", header, length, 0x00, 0x00, msg_type)
            header = struct.pack("BHBBH", self.HEADER, length, 0x00, 0x00, msg_type)

            # Body: struct.pack("iiB", int(pitch * 1000000), int(yaw * 1000000), int(has_target))
            pitch_int = int(pitch * 1000000)
            yaw_int = int(yaw * 1000000)
            body = struct.pack("iiB", pitch_int, yaw_int, int(has_target))

            # Footer: struct.pack("H", 0x0)
            footer = struct.pack("H", 0x0)

            msg = header + body + footer
            bytes_written = self.ser.write(msg)
            self.ser.flush()  # Ensure data is sent immediately

            # Store last sent data for debugging
            self.last_sent_data = {
                "pitch": pitch,
                "yaw": yaw,
                "has_target": has_target,
                "pitch_int": pitch_int,
                "yaw_int": yaw_int,
                "bytes_written": bytes_written,
            }
            self.message_count += 1

            if self.verbose:
                logger.info(
                    f"[STM32 TX #{self.message_count}] pitch={pitch:.6f} rad ({pitch_int}), "
                    f"yaw={yaw:.6f} rad ({yaw_int}), has_target={has_target}, "
                    f"bytes={bytes_written}/{len(msg)}"
                )
                logger.debug(
                    f"[STM32 TX] Raw bytes: {msg.hex(' ')} " f"({len(msg)} bytes)"
                )

            return True

        except serial.SerialException as e:
            logger.warning(f"Serial write error: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending serial data: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Test the serial connection by sending a test message.

        Returns:
            True if test message sent successfully, False otherwise
        """
        if not self.connected or self.ser is None:
            logger.warning("Cannot test: serial not connected")
            return False

        logger.info("Testing STM32 serial connection...")
        test_result = self.send_auto_aim(0.0, 0.0, 0)
        if test_result:
            logger.info("✓ Test message sent successfully")
        else:
            logger.error("✗ Failed to send test message")
        return test_result

    def get_status(self) -> dict:
        """
        Get current status of the serial connection.

        Returns:
            Dictionary with connection status information
        """
        status = {
            "connected": self.connected,
            "port": self.port,
            "baudrate": self.baudrate,
            "message_count": self.message_count,
            "last_sent": self.last_sent_data,
        }
        if self.ser is not None:
            status["port_open"] = self.ser.is_open
            status["in_waiting"] = self.ser.in_waiting
        return status

    def close(self):
        """Close serial connection."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            self.connected = False
            logger.info(
                f"STM32 serial connection closed (sent {self.message_count} messages total)"
            )

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
