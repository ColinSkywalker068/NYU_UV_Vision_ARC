#!/usr/bin/env python3
"""Exercise the production ROS bridge using only disposable virtual serial ports."""

import argparse
import errno
import os
import signal
import struct
import subprocess
import tempfile
import time
import tty
import unittest
from pathlib import Path

import rclpy
from std_msgs.msg import Bool, Float32MultiArray


def attitude_packet(yaw, pitch):
    payload = b"SP\x01" + struct.pack("<4f5fH", 1, 0, 0, 0, yaw, 0, pitch, 0, 15, 0)
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
    return payload + struct.pack("<H", crc)


class SerialReconnectTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cv-serial-reconnect-")
        self.root = Path(self.temp.name)
        self.port = self.root / "controller"
        self.master = None
        self.outbound = bytearray()
        self.samples = []
        self.log_path = self.root / "bridge.log"
        self.log_file = self.log_path.open("w")
        # A separate ROS graph and a PTY path prevent access to the real controller.
        os.environ["ROS_DOMAIN_ID"] = str(100 + os.getpid() % 100)
        os.environ["ROS_LOCALHOST_ONLY"] = "1"
        rclpy.init(args=[])
        self.node = rclpy.create_node("serial_reconnect_test")
        self.sub = self.node.create_subscription(
            Float32MultiArray, "attitude_data", lambda msg: self.samples.append(msg.data), 10
        )
        self.angles = self.node.create_publisher(Float32MultiArray, "bounding_box_angles", 10)
        self.override = self.node.create_publisher(Bool, "override", 10)
        self.bridge = subprocess.Popen(
            [BRIDGE, "--ros-args", "-p", f"serial_port:={self.port}",
             "-p", "receive_enable:=true", "-p", "transmit_enable:=false",
             "-p", f"runtime_log_dir:={self.root}"],
            stdout=self.log_file, stderr=subprocess.STDOUT,
        )

    def tearDown(self):
        self.bridge.send_signal(signal.SIGINT)
        try:
            self.bridge.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.bridge.kill()
            self.bridge.wait()
        if self.master is not None:
            os.close(self.master)
        self.node.destroy_node()
        rclpy.shutdown()
        self.log_file.close()
        self.temp.cleanup()

    def attach(self):
        master, slave = os.openpty()
        tty.setraw(slave)  # Disable echo so bytes read at the master are actual bridge TX.
        os.set_blocking(master, False)
        self.port.symlink_to(os.ttyname(slave))
        os.close(slave)
        self.master = master

    def detach(self):
        os.close(self.master)
        self.master = None
        self.port.unlink()

    def step(self, packet=None):
        self.assertIsNone(self.bridge.poll(), "bridge exited during disconnect")
        self.override.publish(Bool(data=True))
        self.angles.publish(Float32MultiArray(data=[0.2, 0.1]))
        if self.master is not None:
            try:
                if packet is not None:
                    os.write(self.master, packet)
                self.outbound.extend(os.read(self.master, 4096))
            except OSError as exc:
                if exc.errno not in (errno.EIO, errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
        rclpy.spin_once(self.node, timeout_sec=0.02)
        time.sleep(0.01)

    def spin_for(self, seconds, packet=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.step(packet)

    def await_attitude(self, yaw, pitch, timeout=6):
        self.samples.clear()
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.step(attitude_packet(yaw, pitch))
            if any(abs(s[0] - yaw) < 1e-5 and abs(s[1] - pitch) < 1e-5
                   for s in self.samples):
                return
        self.fail(f"no telemetry after attaching virtual controller yaw={yaw}\n"
                  + self.log_path.read_text()[-5000:])

    def test_missing_startup_repeated_unplug_and_no_tx(self):
        self.spin_for(1.3)  # Start with no device at all, as at Jetson boot.
        self.attach()
        self.await_attitude(0.4, -0.1)

        bad = bytearray(attitude_packet(1.11, 0.5))
        bad[-1] ^= 0x80
        self.spin_for(0.3, bytes(bad))
        self.assertFalse(any(abs(s[0] - 1.11) < 1e-5 for s in self.samples))

        for i in range(3):
            # Leave half a packet buffered; it must not survive the new session.
            stale = attitude_packet(-1.0, -0.5)
            os.write(self.master, stale[:13])
            self.spin_for(0.15)
            self.detach()
            self.spin_for(1.4)
            self.attach()
            self.spin_for(1.25)
            self.step(stale[13:])
            self.spin_for(0.15)
            self.assertFalse(any(abs(s[0] + 1.0) < 1e-5 for s in self.samples),
                             "partial telemetry survived a USB session change")
            self.await_attitude(0.6 + i * 0.1, 0.2)
            self.assertFalse(any(abs(s[0] + 1.0) < 1e-5 for s in self.samples))

        self.spin_for(2.3)  # An enumerated device with no stream must be diagnosed.
        text = self.log_path.read_text()
        self.assertIn("Serial disconnected", text)
        self.assertIn("Serial open but no valid SP/ST telemetry", text)
        self.assertGreaterEqual(text.count("STM32 serial connected"), 4)
        self.assertEqual(self.outbound, b"", "TX disabled must remain silent after reconnect")
        for log in self.root.glob("vision_serial_runtime_*.log"):
            self.assertNotIn("kind=serial_tx", log.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True)
    args = parser.parse_args()
    BRIDGE = str(Path(args.bridge).resolve(strict=True))
    unittest.main(argv=["serial_reconnect"], verbosity=2)
