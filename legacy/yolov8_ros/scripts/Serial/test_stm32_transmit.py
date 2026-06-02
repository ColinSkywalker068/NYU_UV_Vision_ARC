#!/usr/bin/env python3
"""
Test script to verify STM32 serial transmission.
This script sends test data to verify the serial connection is working.
"""

import sys
import time
import struct
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

try:
    from stm32_serial import STM32Serial
except ImportError as e:
    print(f"ERROR: Failed to import STM32Serial: {e}")
    sys.exit(1)


def build_packet(pitch: float, yaw: float, has_target: int) -> bytes:
    """
    Build the STM32 packet and return it as bytes.
    This replicates the packet construction from STM32Serial.send_auto_aim()
    """
    HEADER = 0xE7
    length = 11
    msg_type = 1

    # Header: struct.pack("BHBBH", header, length, 0x00, 0x00, msg_type)
    header = struct.pack("BHBBH", HEADER, length, 0x00, 0x00, msg_type)

    # Body: struct.pack("iiB", int(pitch * 1000000), int(yaw * 1000000), int(has_target))
    pitch_int = int(pitch * 1000000)
    yaw_int = int(yaw * 1000000)
    body = struct.pack("iiB", pitch_int, yaw_int, int(has_target))

    # Footer: struct.pack("H", 0x0)
    footer = struct.pack("H", 0x0)

    return header + body + footer


def format_packet_hex(packet: bytes) -> str:
    """
    Format packet bytes as hexadecimal with nice formatting.
    Groups by Header, Body, and Footer sections for readability.
    """
    if len(packet) == 15:  # Expected packet size
        # Header (7 bytes), Body (9 bytes), Footer (2 bytes)
        header_hex = " ".join(f"{b:02X}" for b in packet[0:7])
        body_hex = " ".join(f"{b:02X}" for b in packet[7:16])
        footer_hex = " ".join(f"{b:02X}" for b in packet[16:])
        return f"{header_hex} | {body_hex} | {footer_hex}"
    else:
        # Fallback for unexpected sizes
        return " ".join(f"{b:02X}" for b in packet)


def print_packet_details(pitch: float, yaw: float, has_target: int):
    """
    Print detailed packet information including hex representation.
    """
    packet = build_packet(pitch, yaw, has_target)
    pitch_int = int(pitch * 1000000)
    yaw_int = int(yaw * 1000000)

    print(f"    Packet Details:")
    print(f"      Pitch: {pitch:.6f} rad → {pitch_int} micro-rad")
    print(f"      Yaw:   {yaw:.6f} rad → {yaw_int} micro-rad")
    print(f"      Target: {has_target}")
    print(f"      Packet (hex): {format_packet_hex(packet)}")
    print(f"      Packet size: {len(packet)} bytes")

    # Show byte breakdown
    print(f"      Breakdown:")
    print(f"        Header (7 bytes): {' '.join(f'{b:02X}' for b in packet[0:7])}")
    print(f"        Body (9 bytes):   {' '.join(f'{b:02X}' for b in packet[7:16])}")
    print(f"        Footer (2 bytes): {' '.join(f'{b:02X}' for b in packet[16:])}")


def main():
    # Default port - update this to match your device
    port = "/dev/serial/by-path/platform-3610000.usb-usb-0:2.1:1.0"
    baudrate = 115200

    if len(sys.argv) > 1:
        port = sys.argv[1]
    if len(sys.argv) > 2:
        baudrate = int(sys.argv[2])

    print("=" * 70)
    print("STM32 Serial Transmission Test")
    print("=" * 70)
    print(f"Port: {port}")
    print(f"Baudrate: {baudrate}")
    print()

    # Initialize serial connection
    print("[1] Initializing serial connection...")
    stm32 = STM32Serial(port=port, baudrate=baudrate, verbose=True)

    if not stm32.connected:
        print("✗ Failed to connect to serial port")
        print("\nTroubleshooting:")
        print("  1. Check if device exists: ls -la /dev/serial*")
        print("  2. Check permissions: ls -la /dev/serial/by-path/...")
        print("  3. Try: sudo chmod 666 /dev/serial/by-path/...")
        sys.exit(1)

    print("✓ Serial connection established")
    print()

    # Test connection
    print("[2] Testing connection with zero values...")
    print("    Sending test packet (pitch=0.0, yaw=0.0, target=0)...")
    print_packet_details(0.0, 0.0, 0)
    if stm32.test_connection():
        print("    ✓ Test message sent successfully")
    else:
        print("    ✗ Test message failed")
        sys.exit(1)
    print()

    # Send test data
    print("[3] Sending test data sequences...")
    test_cases = [
        (0.0, 0.0, 0, "Zero angles, no target"),
        (0.1, 0.2, 1, "Small angles, target present"),
        (-0.5, 0.5, 1, "Medium angles, target present"),
        (1.0, -1.0, 1, "Large angles, target present"),
    ]

    for i, (pitch, yaw, has_target, description) in enumerate(test_cases, 1):
        print(f"\n  Test {i}: {description}")
        print(
            f"    Input: Pitch={pitch:.6f} rad, Yaw={yaw:.6f} rad, Target={has_target}"
        )
        print_packet_details(pitch, yaw, has_target)
        success = stm32.send_auto_aim(pitch, yaw, has_target)
        if success:
            print(f"    ✓ Sent successfully (message #{stm32.message_count})")
        else:
            print(f"    ✗ Failed to send")
        time.sleep(0.1)  # Small delay between messages

    print()

    # Get status
    print("[4] Final status:")
    status = stm32.get_status()
    print(f"  Messages sent: {status['message_count']}")
    print(f"  Connection: {'✓ Connected' if status['connected'] else '✗ Disconnected'}")
    if status["last_sent"]:
        last = status["last_sent"]
        print(
            f"  Last sent: pitch={last['pitch']:.6f}, yaw={last['yaw']:.6f}, "
            f"target={last['has_target']}, bytes={last['bytes_written']}"
        )

    print()
    print("=" * 70)
    print("Test complete!")
    print("=" * 70)
    print("\nIf you see messages being sent above, transmission is working.")
    print("Check your STM32 to verify it's receiving the data.")

    # Cleanup
    stm32.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
