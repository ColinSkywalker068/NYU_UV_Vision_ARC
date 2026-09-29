#!/usr/bin/env python3
"""Native tests for deterministic robot/armor scenario generation."""

import math
import unittest

from robot_tracking_scenario import generate_frames, robot_state, scenarios


class RobotTrackingScenarioTest(unittest.TestCase):
    def test_generation_is_deterministic(self):
        scenario = scenarios()["spin_translate"]
        self.assertEqual(
            generate_frames(scenario, fps=20.0, seed=7),
            generate_frames(scenario, fps=20.0, seed=7),
        )

    def test_nonspinning_robot_keeps_phase(self):
        robot = scenarios()["nonspin_translate"].robots[0]
        x0, _, phase0, omega0 = robot_state(robot, 0.0)
        x1, _, phase1, omega1 = robot_state(robot, 2.0)
        self.assertGreater(x1, x0)
        self.assertAlmostEqual(phase1, phase0)
        self.assertEqual(omega0, 0.0)
        self.assertEqual(omega1, 0.0)

    def test_spin_stop_reverse_covers_all_motion_modes(self):
        robot = scenarios()["spin_stop_reverse"].robots[0]
        self.assertGreater(robot_state(robot, 1.0)[3], 0.0)
        self.assertEqual(robot_state(robot, 2.0)[3], 0.0)
        self.assertLess(robot_state(robot, 3.0)[3], 0.0)

    def test_observation_does_not_leak_identity(self):
        frame = generate_frames(scenarios()["two_robot_crossing"], fps=10.0, seed=3)[0]
        self.assertTrue(frame["observations"])
        for observation in frame["observations"]:
            self.assertNotIn("robot_id", observation)
            self.assertNotIn("plate_slot", observation)
        self.assertEqual(len(frame["truth"]["robots"]), 2)

    def test_spinning_phase_advances(self):
        robot = scenarios()["spin_stationary"].robots[0]
        phase_delta = math.remainder(
            robot_state(robot, 0.1)[2] - robot_state(robot, 0.0)[2],
            2.0 * math.pi,
        )
        self.assertAlmostEqual(phase_delta, 0.7)


if __name__ == "__main__":
    unittest.main()
