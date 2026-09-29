#!/usr/bin/env python3
"""Verify the manual observation profile has no command or serial path."""

import pathlib
import unittest


class ObservationLaunchContractTest(unittest.TestCase):
    def test_profile_omits_solver_serial_and_fire(self):
        launch_file = (
            pathlib.Path(__file__).parents[1] / "launch" / "tracking_observe.launch.py"
        )
        source = launch_file.read_text(encoding="utf-8")
        required_fail_closed_settings = {
            '"start_solver": "false"',
            '"start_serial_bridge": "false"',
            '"transmit_enable": "false"',
            '"receive_enable": "false"',
            '"enable_fire_recommendation": "false"',
        }
        for setting in required_fail_closed_settings:
            with self.subTest(setting=setting):
                self.assertIn(setting, source)


if __name__ == "__main__":
    unittest.main()
