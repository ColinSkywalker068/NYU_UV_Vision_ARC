"""Compatibility alias for basler.launch.py.

New field deployments should launch basler.launch.py directly. This file stays
small so older scripts and operator muscle memory keep working during the
transition.
"""

import importlib.util
from pathlib import Path

from launch import LaunchDescription


def generate_launch_description() -> LaunchDescription:
    """Forward to the canonical Basler + STM32 launch description."""
    basler_launch_path = Path(__file__).with_name("basler.launch.py")
    spec = importlib.util.spec_from_file_location("basler_launch", basler_launch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load launch file: {basler_launch_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()
