#!/usr/bin/env python3
"""Synthetic identification tests; no hardware, ROS, or controller access."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("diagnostic", Path(__file__).with_name("diagnose-camera-motion.py"))
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def synthetic(axis, multiplier, delay):
    t = np.arange(0, 4, .01)
    angle = lambda x: .10 * np.sin(2.3 * x) + .025 * np.sin(5.7 * x)
    actual, reported = angle(t), angle(t - delay)
    data = np.zeros((len(t), 8))
    data[:, 0], data[:, 7] = t, 1
    data[:, 1], data[:, 2] = 960, 600
    if axis == "yaw":
        data[:, 1] -= 1160 * multiplier * np.tan(actual)
        data[:, 4] = np.sin(reported / 2)
    else:
        data[:, 2] += 1160 * multiplier * np.tan(actual)
        data[:, 3] = np.sin(reported / 2)
    data[:, 6] = np.cos(reported / 2)
    return data


class Identification(unittest.TestCase):
    def test_recovers_focal_and_delay_separately_for_both_axes(self):
        for axis in ("yaw", "pitch"):
            data = synthetic(axis, 3.15, .03)
            result = diagnostic.fit_window(data, (1160, 1160), (960, 600), axis,
                                           [1, 2, 3, 3.15, 3.3, 4], [-.06, -.03, 0, .03, .06])
            best = result["focal_and_time_shift"]
            self.assertEqual(best["multiplier"], 3.15)
            self.assertAlmostEqual(best["attitude_shift_ms"], 30)
            self.assertLess(best["held_out_rms_deg"], 1e-7)
            self.assertGreater(result["time_shift_only"]["held_out_rms_deg"], 1)

    def test_correct_calibration_does_not_request_gain_change(self):
        result = diagnostic.fit_window(synthetic("yaw", 1, 0), (1160, 1160), (960, 600),
                                       "yaw", [.5, 1, 1.5], [-.03, 0, .03])
        self.assertEqual(result["focal_and_time_shift"]["multiplier"], 1)
        self.assertEqual(result["focal_and_time_shift"]["attitude_shift_ms"], 0)

    def test_equivalent_quaternion_signs_preserve_ray(self):
        data = synthetic("yaw", 3.15, 0)
        expected = diagnostic.bearings(data, (1160, 1160), (960, 600), 3.15, .015)
        data[1::2, 3:7] *= -1
        np.testing.assert_allclose(diagnostic.bearings(data, (1160, 1160), (960, 600), 3.15, .015), expected)

    def test_rejects_mixed_identity_and_short_windows(self):
        data = synthetic("yaw", 1, 0)
        with self.assertRaises(ValueError):
            diagnostic.fit_window(data[:10], (1160, 1160), (960, 600), "yaw", [1], [0])
        data[-1, 7] = 2
        with self.assertRaises(ValueError):
            diagnostic.fit_window(data, (1160, 1160), (960, 600), "yaw", [1], [0])

    def test_parser_excludes_coasting_and_invalid_quaternions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge.log"
            base = "kind=tracking_target has_target=1 camera_pose_valid=1 center_x=960 center_y=600 camera_qx=0 camera_qy=0 camera_qz=0 track_id=1 "
            path.write_text(base + "frame_ns=1000000000 coasting=0 camera_qw=1\n" +
                            base + "frame_ns=2000000000 coasting=1 camera_qw=1\n" +
                            base + "frame_ns=3000000000 coasting=0 camera_qw=0\n")
            self.assertEqual(diagnostic.read_targets(path, 0).shape, (1, 8))


if __name__ == "__main__":
    unittest.main()
