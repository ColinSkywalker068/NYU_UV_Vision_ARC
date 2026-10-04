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
from std_msgs.msg import Bool, Float32MultiArray, Int8
from cv_auto_aim_interfaces.msg import AimSolution, Target, TrackedArmor, TrackedArmors


def with_crc(payload):
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
    return payload + struct.pack("<H", crc)


def attitude_packet(yaw, pitch):
    return with_crc(b"SP\x01" + struct.pack("<4f5fH", 1, 0, 0, 0, yaw, 0, pitch, 0, 15, 0))


def status_packet(robot_status):
    payload = bytearray(37)
    payload[:2] = b"ST"
    payload[26] = robot_status
    payload[30] = 7
    return with_crc(payload)


class SerialReconnectTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cv-serial-reconnect-")
        self.root = Path(self.temp.name)
        self.port = self.root / "controller"
        self.master = None
        self.send_valid_solution = True
        self.provisional = self._testMethodName.startswith("test_provisional_calibration_")
        self.tx_enabled = self._testMethodName in ("test_provisional_calibration_respects_enabled_tx", "test_atomic_solution_rejects_stale_offset_and_replay", "test_atomic_angular_rates_and_safe_zero_fallback", "test_controller_status_independent_of_enabled_tx")
        self.outbound = bytearray()
        self.samples = []
        self.controller_status = []
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
        self.status_sub = self.node.create_subscription(
            Int8, "controller_auto_aim", lambda msg: self.controller_status.append(msg.data), 10
        )
        self.angles = self.node.create_publisher(Float32MultiArray, "bounding_box_angles", 10)
        self.solutions = self.node.create_publisher(AimSolution, "aim_solution", 10)
        self.targets = self.node.create_publisher(Target, "processor/target", 10)
        self.associations = self.node.create_publisher(TrackedArmors, "tracker/tracked_armors", 10)
        self.override = self.node.create_publisher(Bool, "override", 10)
        self.bridge = subprocess.Popen(
            [BRIDGE, "--ros-args", "-p", f"serial_port:={self.port}",
             "-p", "receive_enable:=true", "-p", f"transmit_enable:={str(self.tx_enabled).lower()}",
             "-p", f"calibration_provisional:={str(self.provisional).lower()}",
             "-p", f"runtime_log_dir:={self.root}",
             "-p", f"runtime_log_max_bytes:={4096 if self._testMethodName == 'test_alignment_logging_without_tx_and_rotation' else 67108864}",
             "-p", "runtime_log_backups:=2"],
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
        solution = AimSolution()
        solution.header.stamp = self.node.get_clock().now().to_msg()
        solution.header.frame_id = "gimbal_inertial"
        solution.output_mode = "absolute"
        solution.has_solution = True
        solution.used_3d = True
        solution.has_angular_velocity = True
        solution.yaw_velocity = 0.35
        solution.pitch_velocity = -0.2
        solution.yaw = 0.2
        solution.pitch = 0.1
        if self.send_valid_solution:
            target = Target()
            target.header = solution.header
            target.measurement_stamp_ns = self.node.get_clock().now().nanoseconds
            target.has_target = True
            target.track_id = 7
            self.targets.publish(target)
            self.solutions.publish(solution)
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

    def await_controller_status(self, expected, packet=None, timeout=3):
        self.controller_status.clear()
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.step(packet)
            if self.controller_status and self.controller_status[-1] == expected:
                return
        self.fail(f"controller status did not become {expected}: {self.controller_status}")

    def check_controller_status(self):
        self.await_controller_status(-1)  # Missing device at startup.
        self.attach()
        self.await_attitude(0.4, -0.1)
        self.await_controller_status(1, status_packet(0x8f))
        self.await_controller_status(0, status_packet(0x87))
        self.assertEqual(bool(self.outbound), self.tx_enabled)
        self.await_controller_status(-1, status_packet(0x08))  # Legacy; bit3 alone is not ON.
        self.await_controller_status(1, status_packet(0x88))
        corrupt = bytearray(status_packet(0x88))
        corrupt[-1] ^= 0x80
        # Valid SP mode1 plus bad ST cannot refresh the acceptance clock.
        self.spin_for(.75, attitude_packet(0.4, -0.1) + bytes(corrupt))
        self.assertEqual(self.controller_status[-1], -1)
        self.await_controller_status(0, status_packet(0x80))
        self.controller_status.clear()
        self.spin_for(.2, bytes(corrupt))
        self.assertNotIn(1, self.controller_status, "bad CRC revived ON")
        for _ in range(2):
            self.await_controller_status(1, status_packet(0x88))
            self.detach()
            self.await_controller_status(-1)
            self.attach()
            self.await_attitude(0.5, -.2)
            self.assertEqual(self.controller_status[-1], -1, "SP mode1 revived acceptance")
            self.await_controller_status(0, status_packet(0x80))
        self.assertEqual(bool(self.outbound), self.tx_enabled)
        log = "\n".join(path.read_text() for path in self.root.glob("vision_serial_runtime_*.log"))
        self.assertIn("cmd=0x5354 controller_auto_aim=1", log)
        self.assertIn("cmd=0x5354 controller_auto_aim=0", log)
        self.assertIn("cmd=0x5354 controller_auto_aim=-1", log)
        self.assertIn("cmd=0x5350 controller_auto_aim=-1", log)

    def test_controller_status_independent_of_disabled_tx(self):
        self.check_controller_status()

    def test_controller_status_independent_of_enabled_tx(self):
        self.check_controller_status()

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

    def check_provisional_tx(self):
        self.attach()
        self.await_attitude(0.4, -0.1)
        self.spin_for(0.5, attitude_packet(0.4, -0.1))
        if self.tx_enabled:
            self.assertGreaterEqual(len(self.outbound), 29)
            packets = [self.outbound[i:i+29] for i in range(0, len(self.outbound)-28, 29)]
            self.assertTrue(any(packet[2] == 1 for packet in packets))
        else:
            self.assertEqual(self.outbound, b"", "TX false must always stay silent")
        self.outbound.clear()
        self.detach()
        self.spin_for(1.4)
        self.attach()
        self.await_attitude(0.5, -0.2)
        self.spin_for(0.2)
        self.assertEqual(bool(self.outbound), self.tx_enabled)

    def test_provisional_calibration_respects_enabled_tx(self):
        self.check_provisional_tx()

    def test_provisional_calibration_respects_disabled_tx(self):
        self.check_provisional_tx()

    def test_alignment_logging_without_tx_and_rotation(self):
        self.attach()
        self.await_attitude(0.4, -0.1)
        self.spin_for(1.5, attitude_packet(0.4, -0.1))
        logs = sorted(self.root.glob("vision_serial_runtime_*"))
        self.assertEqual(len(logs), 3)
        content = "\n".join(p.read_text() for p in logs)
        self.assertIn("kind=tracking_alignment", content)
        self.assertIn("kind=tracking_target", content)
        self.assertIn("track_id=7", content)
        self.assertIn("tx_enabled=0", content)
        self.assertIn("attitude_fresh=1", content)
        self.assertIn("source_timestamped=0", content)
        self.assertIn("yaw_error=", content)
        self.assertIn("ros_ns=", content)
        self.assertTrue(all(p.stat().st_size < 6000 for p in logs))
        self.assertNotIn("kind=serial_tx", content)
        self.assertEqual(self.outbound, b"")

    def test_association_logging_without_tx(self):
        self.attach()
        self.await_attitude(0.4, -0.1)
        frame = TrackedArmors()
        frame.source_id = "test_camera"
        frame.selected_track_id = 99
        frame.raw_count = 3
        frame.enemy_count = 2
        frame.posed_count = 1
        frame.matched_count = 0
        frame.innovation_rejected_count = 1
        for index in range(9):
            track = TrackedArmor()
            track.track_id = 99 if index == 8 else index
            track.association_reason = "innovation_pending"
            track.lost_frames = 3
            track.real_measurement_age_s = 0.05
            track.innovation_m = 0.3
            track.armor.reprojection_rms_px = 1.2
            frame.tracks.append(track)
        end = time.monotonic() + 0.8
        while time.monotonic() < end:
            frame.header.stamp = self.node.get_clock().now().to_msg()
            self.associations.publish(frame)
            self.step(attitude_packet(0.4, -0.1))
        content = "\n".join(
            p.read_text() for p in self.root.glob("vision_serial_runtime_*")
        )
        rows = [
            line
            for line in content.splitlines()
            if "kind=tracking_association " in line
        ]
        self.assertTrue(rows, content[-4000:])
        self.assertIn(
            "raw_count=3 enemy_count=2 posed_count=1 matched_count=0", rows[-1]
        )
        self.assertIn("innovation_rejected_count=1", rows[-1])
        self.assertIn("track4_id=99", rows[-1])
        self.assertIn('track4_reason="innovation_pending"', rows[-1])
        self.assertIn("logged_tracks=5", rows[-1])
        self.assertNotIn("track5_", rows[-1])
        self.assertEqual(
            self.outbound, b"", "diagnostic subscriptions must not enable TX"
        )

    def test_atomic_angular_rates_and_safe_zero_fallback(self):
        self.attach()
        self.await_attitude(0.4, -0.1)
        self.spin_for(0.2)
        self.send_valid_solution = False
        self.spin_for(0.05)

        def send_case(**fields):
            solution = AimSolution()
            solution.header.stamp = self.node.get_clock().now().to_msg()
            solution.header.frame_id = "gimbal_inertial"
            solution.output_mode = "absolute"
            solution.has_solution = True
            solution.used_3d = True
            solution.has_angular_velocity = True
            solution.yaw, solution.pitch = 0.6, -0.15
            solution.yaw_velocity, solution.pitch_velocity = 0.7, -0.4
            for key, value in fields.items():
                setattr(solution, key, value)
            self.outbound.clear()
            self.solutions.publish(solution)
            self.spin_for(0.06)
            packets = [self.outbound[i:i+29] for i in range(0, len(self.outbound)-28, 29)]
            self.assertTrue(packets)
            return solution, packets[-1], struct.unpack("<6f", packets[-1][3:27])

        _, packet, values = send_case()
        self.assertEqual(packet[2], 1)
        for actual, expected in zip(values, (0.6, 0.7, 0, -0.15, -0.4, 0)):
            self.assertAlmostEqual(actual, expected, places=5)
        for fields in ({"has_angular_velocity": False}, {"used_3d": False},
                       {"yaw_velocity": float("nan")}, {"pitch_velocity": 100.0}):
            _, packet, values = send_case(**fields)
            self.assertEqual(packet[2], 1, "invalid rate must retain valid position aiming")
            self.assertEqual((values[1], values[2], values[4], values[5]), (0, 0, 0, 0))
        solution, _, _ = send_case()
        self.outbound.clear()
        self.solutions.publish(solution)  # Replayed atomic solution clears rates and control.
        self.spin_for(.06)
        packets = [self.outbound[i:i+29] for i in range(0, len(self.outbound)-28, 29)]
        self.assertTrue(packets)
        self.assertEqual(packets[-1][2], 0)
        values = struct.unpack("<6f", packets[-1][3:27])
        self.assertEqual((values[1], values[4]), (0, 0))

    def test_atomic_solution_rejects_stale_offset_and_replay(self):
        self.attach()
        self.await_attitude(0.4, -0.1)
        self.spin_for(0.2)
        self.assertGreaterEqual(len(self.outbound), 29)
        valid = [self.outbound[i:i+29] for i in range(0, len(self.outbound)-28, 29)]
        self.assertTrue(any(packet[2] == 1 for packet in valid), "no active auto-aim packet")
        self.send_valid_solution = False
        self.spin_for(0.35)  # Expire the active command before independent cases.
        for kind in ("stale", "future", "offset", "nan", "replay"):
            solution = AimSolution()
            stamp = self.node.get_clock().now().nanoseconds
            if kind == "stale": stamp -= 1_000_000_000
            if kind == "future": stamp += 1_000_000_000
            solution.header.stamp.sec = stamp // 1_000_000_000
            solution.header.stamp.nanosec = stamp % 1_000_000_000
            solution.header.frame_id = "gimbal_inertial"
            solution.output_mode = "offset" if kind == "offset" else "absolute"
            solution.has_solution = True
            solution.yaw = float("nan") if kind == "nan" else 0.3
            solution.pitch = 0.1
            if kind == "replay":
                self.solutions.publish(solution)
                self.spin_for(0.1)
            self.outbound.clear()
            self.solutions.publish(solution)
            self.spin_for(0.1)
            packets = [self.outbound[i:i+29] for i in range(0, len(self.outbound)-28, 29)]
            self.assertTrue(packets, f"{kind}: missing explicit no-target packet")
            self.assertTrue(all(packet[2] == 0 for packet in packets),
                            f"{kind}: invalid solution enabled control")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", required=True)
    args = parser.parse_args()
    BRIDGE = str(Path(args.bridge).resolve(strict=True))
    unittest.main(argv=["serial_reconnect"], verbosity=2)
