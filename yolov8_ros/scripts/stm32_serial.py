#!/usr/bin/env python3
"""
STM32 Serial Communication Module
Sends pitch/yaw angles and target status to STM32 microcontroller via USB-C serial.
"""

import serial
import struct
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class STM32Serial:
    """Serial communication class for STM32 microcontroller."""

    def __init__(self, port: str = "/dev/serial", baudrate: int = 115200):
        """
        Initialize STM32 serial connection.

        Args:
            port: Serial port path (e.g., "/dev/serial", "/dev/ttyUSB0", "/dev/ttyACM0")
            baudrate: Serial baud rate (default: 115200)
        """
        self.port = port
        self.baudrate = baudrate
        self.ser: Optional[serial.Serial] = None
        self.HEADER = 0xE7
        self.connected = False

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
            return False

        try:
            # Packet format matching legacy uart_comms.py
            # msg = [pitch, yaw, has_target]
            length = 11
            msg_type = 1

            # Header: struct.pack("BHBBH", header, length, 0x00, 0x00, msg_type)
            header = struct.pack("BHBBH", self.HEADER, length, 0x00, 0x00, msg_type)

            # Body: struct.pack("iiB", int(pitch * 1000000), int(yaw * 1000000), int(has_target))
            body = struct.pack(
                "iiB", int(pitch * 1000000), int(yaw * 1000000), int(has_target)
            )

            # Footer: struct.pack("H", 0x0)
            footer = struct.pack("H", 0x0)

            msg = header + body + footer
            self.ser.write(msg)
            return True

        except serial.SerialException as e:
            logger.warning(f"Serial write error: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending serial data: {e}")
            return False

    def close(self):
        """Close serial connection."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
            self.connected = False
            logger.info("STM32 serial connection closed")

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

