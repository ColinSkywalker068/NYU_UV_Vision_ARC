"""Evaluate the field launch parameters without executing any nodes/devices."""
import importlib.util
from pathlib import Path
import unittest

from launch import LaunchContext
from launch_ros.utilities import evaluate_parameters


class LaunchParameters(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "launch" / "basler.launch.py"
        spec = importlib.util.spec_from_file_location("basler_launch", path)
        self.launch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.launch)
        self.context = LaunchContext()
        for argument in self.launch.declare_arguments():
            argument.execute(self.context)

    def parameters(self, factory):
        node = factory()
        # ROS exposes no public accessor for an unexecuted Node's normalized
        # parameters. Evaluate the actual stored mapping; never execute Node.
        return evaluate_parameters(self.context, node._Node__parameters)[0]

    def test_aruw_and_source_calibration_reach_detector(self):
        self.context.launch_configurations.update({
            "model_type": "aruw_2026_pose", "task": "pose",
            "basler_width": "640", "basler_height": "384",
            "pre_mux_resize": "false", "confidence_threshold": "0.5",
            "camera_calibration_width": "1920", "camera_calibration_height": "1200",
            "camera_distortion": "[0.1,-0.2,0.0,0.0,0.0]",
        })
        p = self.parameters(self.launch.make_detector_node)
        self.assertEqual(p["type"], "aruw_2026_pose")
        self.assertEqual(p["task"], "pose")
        self.assertEqual((p["imgsz_width"], p["imgsz_height"]), (640, 384))
        self.assertEqual((p["camera_calibration_width"], p["camera_calibration_height"]), (1920, 1200))
        self.assertEqual(p["camera_distortion"], (0.1, -0.2, 0.0, 0.0, 0.0))
        self.assertEqual(p["conf"], 0.5)
        self.assertFalse(p["pre_mux_resize"])

    def test_provisional_calibration_is_independent_of_transmit_enable(self):
        self.context.launch_configurations["calibration_provisional"] = "true"
        for factory in (self.launch.make_detector_node, self.launch.make_solver_node,
                        self.launch.make_stm_bridge_node):
            self.assertTrue(self.parameters(factory)["calibration_provisional"])
        self.context.launch_configurations["transmit_enable"] = "true"
        self.assertTrue(self.parameters(self.launch.make_stm_bridge_node)["transmit_enable"])
        self.context.launch_configurations["transmit_enable"] = "false"
        self.assertFalse(self.parameters(self.launch.make_stm_bridge_node)["transmit_enable"])

    def test_preview_defaults_and_controller_limits(self):
        bridge = self.parameters(self.launch.make_stm_bridge_node)
        solver = self.parameters(self.launch.make_solver_node)
        tracker = self.parameters(self.launch.make_tracker_node)
        self.assertFalse(bridge["transmit_enable"])
        self.assertFalse(bridge["transmit_angles_in_degrees"])
        self.assertFalse(solver["enable_fire_recommendation"])
        self.assertEqual(solver["output_mode"], "absolute")
        self.assertEqual(solver["yaw_max_slew_deg_s"], 60.0)
        self.assertEqual(solver["pitch_max_accel_deg_s2"], 240.0)
        self.assertEqual(tracker["camera_translation_m"], (0.0, 0.0, 0.0))
        self.assertEqual(tracker["track_max_coast_frames"], 6)
        self.assertEqual(tracker["track_max_coast_s"], 0.12)
        self.assertEqual(tracker["track_drop_frames"], 16)
        self.assertEqual(tracker["velocity_filter_tau_s"], 0.10)


if __name__ == "__main__":
    unittest.main()
