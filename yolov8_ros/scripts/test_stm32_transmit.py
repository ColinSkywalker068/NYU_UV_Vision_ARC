#!/usr/bin/env python3
"""
Test script to verify STM32 serial transmission.
This script sends test data to verify the serial connection is working.
"""

import sys
import time
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
    if stm32.test_connection():
        print("✓ Test message sent successfully")
    else:
        print("✗ Test message failed")
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
        print(f"    Pitch: {pitch:.6f} rad, Yaw: {yaw:.6f} rad, Target: {has_target}")
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
