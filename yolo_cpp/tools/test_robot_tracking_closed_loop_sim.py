#!/usr/bin/env python3
"""Native closed-loop tracking and gimbal simulation tests."""

import unittest

from robot_tracking_closed_loop_sim import run_closed_loop
from robot_tracking_scenario import scenarios


class ClosedLoopSimulationTest(unittest.TestCase):
    def test_fire_recommendation_defaults_off(self):
        robot = scenarios()["nonspin_translate"].robots[0]
        trace, summary = run_closed_loop(robot, fps=60.0, seed=2)
        self.assertTrue(trace)
        self.assertEqual(summary["fire_recommendation_frames"], 0)
        self.assertFalse(any(sample["fire_recommendation"] for sample in trace))

    def test_opt_in_fire_is_boolean_trace_only(self):
        robot = scenarios()["nonspin_translate"].robots[0]
        _, summary = run_closed_loop(
            robot,
            fps=60.0,
            seed=2,
            enable_fire_recommendation=True,
        )
        self.assertGreater(summary["fire_recommendation_frames"], 0)
        self.assertIsNotNone(summary["fire_recommendation_truth_alignment_fraction"])
        self.assertIn("not simulated", summary["note"])

    def test_gimbal_converges_on_nonspinning_target(self):
        robot = scenarios()["nonspin_translate"].robots[0]
        trace, _ = run_closed_loop(robot, fps=60.0, seed=3, position_noise_std_m=0.0)
        initial_error = abs(trace[0]["aim_error_deg"])
        settled_error = abs(trace[min(len(trace) - 1, 60)]["aim_error_deg"])
        self.assertLess(settled_error, initial_error)

    def test_stationary_single_plate_converges_without_fire(self):
        robot = scenarios()["stationary_single_plate"].robots[0]
        trace, summary = run_closed_loop(
            robot, fps=60.0, seed=5, position_noise_std_m=0.0
        )
        self.assertTrue(trace)
        self.assertLess(abs(trace[-1]["aim_error_deg"]), 0.1)
        self.assertLess(summary["center_rmse_m"], 1e-9)
        self.assertEqual(summary["fire_recommendation_frames"], 0)

    def test_spin_stop_reverse_runs_with_bounded_center_error(self):
        robot = scenarios()["spin_stop_reverse"].robots[0]
        trace, summary = run_closed_loop(robot, fps=60.0, seed=4)
        self.assertTrue(trace)
        self.assertLess(summary["center_rmse_m"], 0.05)


if __name__ == "__main__":
    unittest.main()
