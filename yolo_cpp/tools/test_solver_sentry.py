#!/usr/bin/env python3
"""Retired test entry point: the old unstamped attitude contract is unsupported."""
raise SystemExit(
    "This test used the retired attitude_data/unstamped-target interface. "
    "From the repository on a host with Docker, run tools/run-local-tracking-tests.sh. "
    "In a sourced, built ROS workspace, use: colcon test --packages-select yolo_cpp "
    "--ctest-args -R 'test_aim_geometry|test_pose_pipeline|test_closed_loop_pipeline' "
    "--return-code-on-test-failure. No hardware is needed."
)
