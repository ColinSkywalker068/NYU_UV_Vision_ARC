#!/usr/bin/env python3
"""
Test script to verify pyserial installation and serial port availability.
Run this to diagnose serial communication issues.
"""

import sys

print("=" * 60)
print("Serial Module Test")
print("=" * 60)

# Test 1: Import serial module
print("\n[1] Testing serial module import...")
try:
    import serial
    print("✓ serial module imported successfully")
    print(f"  Module path: {serial.__file__}")
    print(f"  Module version: {getattr(serial, 'VERSION', 'unknown')}")
except ImportError as e:
    print(f"✗ Failed to import serial: {e}")
    print("  Solution: pip3 install pyserial")
    sys.exit(1)

# Test 2: Check for required attributes
print("\n[2] Checking serial module attributes...")
required_attrs = ['Serial', 'SerialException', 'PARITY_NONE', 'STOPBITS_ONE', 'EIGHTBITS']
for attr in required_attrs:
    if hasattr(serial, attr):
        print(f"✓ serial.{attr} exists")
    else:
        print(f"✗ serial.{attr} NOT FOUND")
        print(f"  This might indicate a wrong 'serial' module is imported")

# Test 3: List available serial ports
print("\n[3] Listing available serial ports...")
try:
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    if ports:
        print(f"✓ Found {len(ports)} serial port(s):")
        for port in ports:
            print(f"  - {port.device}")
            print(f"    Description: {port.description}")
            print(f"    Hardware ID: {port.hwid}")
    else:
        print("⚠ No serial ports found")
        print("  This is normal if no USB serial device is connected")
except Exception as e:
    print(f"✗ Error listing ports: {e}")

# Test 4: Check if /dev/serial exists
print("\n[4] Checking /dev/serial...")
try:
    from pathlib import Path
    serial_path = Path("/dev/serial")
    if serial_path.exists():
        print(f"✓ /dev/serial exists")
        if serial_path.is_dir():
            print("  It's a directory. Contents:")
            for item in serial_path.iterdir():
                print(f"    - {item}")
        elif serial_path.is_char_device():
            print("  It's a character device")
        else:
            print(f"  It's a {serial_path.stat().st_mode}")
    else:
        print("✗ /dev/serial does NOT exist")
        print("  Common alternatives:")
        print("    - /dev/ttyUSB0, /dev/ttyUSB1, ...")
        print("    - /dev/ttyACM0, /dev/ttyACM1, ...")
        print("    - /dev/cu.usbmodem* (macOS)")
        print("    - /dev/cu.usbserial* (macOS)")
except Exception as e:
    print(f"✗ Error checking /dev/serial: {e}")

# Test 5: Try importing STM32Serial
print("\n[5] Testing STM32Serial import...")
try:
    from stm32_serial import STM32Serial
    print("✓ STM32Serial imported successfully")
except ImportError as e:
    print(f"✗ Failed to import STM32Serial: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Error importing STM32Serial: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)

