"""Orin Nano visual bring-up profile.

This launch file keeps the canonical Basler graph from basler.launch.py, but
defaults it to the Orin-stable native TensorRT + full-sensor Bayer profile.
"""

import importlib.util
from pathlib import Path
from typing import Any

from launch import LaunchDescription
from launch.substitutions import EnvironmentVariable


ORIN_DEFAULTS = {
    "detector_backend": "native_tensorrt",
    # Let ArmorDetectorNode resolve the model from the current checkout when
    # CV_STM_MODEL_PATH is unset.  The former absolute path only worked for
    # the robot's historical /home/ultraviolet/CV_Detection checkout.
    "model_path": EnvironmentVariable("CV_STM_MODEL_PATH", default_value=""),
    "basler_pixel_format": "bayer_rggb8",
    "basler_use_full_sensor": "true",
    "basler_source_width": "1920",
    "basler_source_height": "1200",
    "basler_fps": "160",
    "basler_exposure_us": "3000.0",
    "native_tensorrt_cuda_bayer_preprocess": "true",
    "show": "gui",
}


def load_basler_launch() -> Any:
    """Load the canonical Basler launch module from the same directory."""
    basler_launch_path = Path(__file__).with_name("basler.launch.py")
    spec = importlib.util.spec_from_file_location("basler_launch", basler_launch_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load launch file: {basler_launch_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_launch_description() -> LaunchDescription:
    """Create the Orin profile by overriding Basler launch argument defaults."""
    basler_launch = load_basler_launch()
    base_argument_specs = basler_launch.argument_specs

    def orin_argument_specs() -> list[tuple[str, Any, str]]:
        return [
            (name, ORIN_DEFAULTS.get(name, default), description)
            for name, default, description in base_argument_specs()
        ]

    basler_launch.argument_specs = orin_argument_specs
    return basler_launch.generate_launch_description()
