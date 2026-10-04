#!/usr/bin/env python3
"""Verify Type C quaternion decoding through the production bridge and a PTY."""

import argparse
import math
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
from cv_auto_aim_interfaces.msg import GimbalAttitude
from std_msgs.msg import Float32MultiArray


def typec_quaternion(yaw, pitch, roll):
    """Type C modules/imu/ins_task.c EularAngleToQuaternion, angles in degrees."""
    cy, cp, cr = (math.cos(math.radians(v) / 2) for v in (yaw, pitch, roll))
    sy, sp, sr = (math.sin(math.radians(v) / 2) for v in (yaw, pitch, roll))
    return (cp * cr * cy + sp * sr * sy,
            sp * cr * cy - cp * sr * sy,
            sp * cr * sy + cp * sr * cy,
            cp * cr * sy - sp * sr * cy)


def packet(quaternion, yaw, pitch=-0.43):
    payload = b"SP\x01" + struct.pack(
        "<9fH", *quaternion, yaw, 0.12, pitch, -0.21, 15, 0)
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
    return payload + struct.pack("<H", crc)


class SerialQuaternionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cv-serial-quaternion-")
        self.root = Path(self.temp.name)
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)
        port = os.ttyname(self.slave)
        self.assertTrue(port.startswith("/dev/pts/"), "only virtual serial ports allowed")
        os.environ["ROS_DOMAIN_ID"] = "196"
        os.environ["ROS_LOCALHOST_ONLY"] = "1"
        rclpy.init(args=[])
        self.node = rclpy.create_node("serial_quaternion_test")
        self.samples = []
        self.legacy = []
        self.sub = self.node.create_subscription(
            GimbalAttitude, "gimbal/attitude", self.samples.append, 100)
        self.legacy_sub = self.node.create_subscription(
            Float32MultiArray, "attitude_data", lambda msg: self.legacy.append(msg.data), 100)
        self.log_path = self.root / "bridge.log"
        self.log_file = self.log_path.open("w")
        self.bridge = subprocess.Popen(
            [BRIDGE, "--ros-args", "-p", f"serial_port:={port}",
             "-p", "receive_enable:=true", "-p", "transmit_enable:=false",
             "-p", f"runtime_log_dir:={self.root}"],
            stdout=self.log_file, stderr=subprocess.STDOUT)

    def tearDown(self):
        self.bridge.send_signal(signal.SIGINT)
        try:
            self.bridge.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.bridge.kill()
            self.bridge.wait()
        self.node.destroy_node()
        rclpy.shutdown()
        os.close(self.master)
        os.close(self.slave)
        self.log_file.close()
        self.temp.cleanup()

    def await_yaw(self, yaw, data):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.assertIsNone(self.bridge.poll(), self.log_path.read_text())
            os.write(self.master, data)
            rclpy.spin_once(self.node, timeout_sec=0.02)
            typed = next((s for s in reversed(self.samples) if abs(s.yaw - yaw) < 1e-5), None)
            legacy = next((s for s in reversed(self.legacy) if abs(s[0] - yaw) < 1e-5), None)
            if typed is not None and legacy is not None:
                return typed, legacy
        self.fail(f"no decoded packet yaw={yaw}\n{self.log_path.read_text()[-5000:]}")

    def test_typec_roll_and_validation(self):
        # The old X-angle decoder returns +/-20 degrees for the first two cases.
        cases = [(0, pitch, 0) for pitch in (-20, 20)]
        cases += [(0, 0, roll) for roll in (-10, 10)]
        cases += [(yaw, pitch, roll) for yaw in (-170, -90, 90, 170)
                  for pitch in (-20, 20) for roll in (-5, 5)]
        valid = []
        for angles in cases:
            q = typec_quaternion(*angles)
            for scale in (1, -1, 0.75, 1.5):
                valid.append((f"angles={angles}, scale={scale}",
                              tuple(scale * v for v in q), math.radians(angles[2])))
        valid += [("zero fallback", (0, 0, 0, 0), 0),
                  ("near-zero fallback", (1e-7, 0, 0, 0), 0),
                  ("lower norm boundary", (0.5, 0, 0, 0), 0),
                  ("upper norm boundary", (2, 0, 0, 0), 0)]
        # A rounded float32 sqrt(1/2) with raw 2*w*y > 1 must stay finite at
        # both asin endpoints after normalization/clamping.
        endpoint = struct.unpack("<f", struct.pack("<f", 0.7071068))[0]
        self.assertGreater(2 * endpoint * endpoint, 1)
        valid += [("positive asin endpoint", (endpoint, 0, endpoint, 0), math.pi / 2),
                  ("negative asin endpoint", (endpoint, 0, -endpoint, 0), -math.pi / 2)]
        for index, (label, q, expected_roll) in enumerate(valid):
            with self.subTest(label=label):
                # Deliberately disagree with quaternion yaw/pitch: the scalar
                # fields must remain authoritative, even with valid quaternions.
                yaw = 0.5 + index * 0.01
                typed, legacy = self.await_yaw(yaw, packet(q, yaw))
                self.assertTrue(math.isfinite(typed.roll))
                self.assertAlmostEqual(typed.roll, expected_roll, delta=2e-6)
                self.assertAlmostEqual(typed.pitch, -0.43, delta=1e-6)
                self.assertAlmostEqual(typed.yaw_rate, 0.12, delta=1e-6)
                self.assertAlmostEqual(typed.pitch_rate, -0.21, delta=1e-6)
                self.assertAlmostEqual(legacy[1], -0.43, delta=1e-6)
                self.assertAlmostEqual(legacy[2], expected_roll, delta=2e-6)

        invalid = [(0.49, 0, 0, 0), (2.01, 0, 0, 0), (1e-5, 0, 0, 0)]
        for component in range(4):
            for value in (float("nan"), float("inf"), -float("inf")):
                q = [1, 0, 0, 0]
                q[component] = value
                invalid.append(tuple(q))
        for index, q in enumerate(invalid):
            with self.subTest(invalid_quaternion=q):
                bad_yaw, good_yaw = -1 - index * 0.02, -1.01 - index * 0.02
                # A following valid marker proves the stream advanced past the
                # rejected packet. Check RX logs below too: publication alone
                # selects only the newest accepted SP packet in a read batch.
                self.await_yaw(good_yaw, packet(q, bad_yaw) + packet((1, 0, 0, 0), good_yaw))
                self.assertFalse(any(abs(s.yaw - bad_yaw) < 1e-5 for s in self.samples))
                self.assertFalse(any(abs(s[0] - bad_yaw) < 1e-5 for s in self.legacy))

        rclpy.spin_once(self.node, timeout_sec=0.02)
        logs = list(self.root.glob("vision_serial_runtime_*.log"))
        self.assertTrue(logs, "bridge did not create its decoded RX log")
        decoded_yaws = [float(dict(field.split("=", 1) for field in line.split())['yaw'])
                        for log in logs for line in log.read_text().splitlines()
                        if "kind=serial_rx " in line]
        self.assertTrue(decoded_yaws)
        for index in range(len(invalid)):
            self.assertFalse(any(abs(yaw - (-1 - index * 0.02)) < 1e-5
                                 for yaw in decoded_yaws), "invalid quaternion was decoded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True)
    args, remaining = parser.parse_known_args()
    BRIDGE = args.bridge
    unittest.main(argv=[__file__, *remaining])
