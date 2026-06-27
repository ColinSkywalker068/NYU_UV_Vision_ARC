"""Launch the Basler auto-aim stack with the NYUSH sentry gimbal bridge."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_config(name: str) -> LaunchConfiguration:
    return LaunchConfiguration(name)


def load_basler_launch_module() -> Any:
    spec = spec_from_file_location(
        "yolo_cpp_basler_launch", Path(__file__).with_name("basler.launch.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load sibling basler.launch.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASLER_LAUNCH = load_basler_launch_module()


def basler_argument_specs() -> list[tuple[str, Any, str]]:
    """Mirror basler.launch.py args while disabling the SeaSky STM bridge here."""
    specs = []
    for name, default, description in BASLER_LAUNCH.argument_specs():
        if name == "stm32_serial_enable":
            specs.append(
                (name, "false", "Keep SeaSky STM32 serial disabled in NYUSH mode")
            )
        else:
            specs.append((name, default, description))
    return specs


def declare_basler_arguments() -> list[DeclareLaunchArgument]:
    return [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in basler_argument_specs()
    ]


def make_basler_stack() -> IncludeLaunchDescription:
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("yolo_cpp"), "launch", "basler.launch.py"]
            )
        ),
        launch_arguments={
            name: launch_config(name) for name, _, _ in basler_argument_specs()
        }.items(),
    )


def make_nyush_gimbal_bridge() -> Node:
    return Node(
        package="yolo_cpp",
        executable="nyush_gimbal_serial_bridge",
        name="nyush_gimbal_serial_bridge",
        output="screen",
        condition=IfCondition(launch_config("nyush_gimbal_enable")),
        parameters=[
            {
                "transmit_enable": launch_config("nyush_gimbal_enable"),
                "serial_port": launch_config("nyush_gimbal_port"),
                "serial_baudrate": launch_config("nyush_gimbal_baudrate"),
                "serial_verbose": launch_config("nyush_gimbal_verbose"),
                "enemy_color": launch_config("nyush_enemy_color"),
                "rx_yaw_sign": launch_config("nyush_rx_yaw_sign"),
                "rx_pitch_sign": launch_config("nyush_rx_pitch_sign"),
                "rx_roll_sign": launch_config("nyush_rx_roll_sign"),
                "tx_yaw_sign": launch_config("nyush_tx_yaw_sign"),
                "tx_pitch_sign": launch_config("nyush_tx_pitch_sign"),
            }
        ],
    )


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            *declare_basler_arguments(),
            DeclareLaunchArgument(
                "nyush_gimbal_enable",
                default_value="true",
                description="Start the NYUSH sentry gimbal serial bridge",
            ),
            DeclareLaunchArgument(
                "nyush_gimbal_port",
                default_value="/tmp/nyush-rm-sentry-vision",
                description="NYUSH gimbal serial device or pseudo-terminal",
            ),
            DeclareLaunchArgument(
                "nyush_gimbal_baudrate",
                default_value="115200",
                description="NYUSH gimbal serial baud rate",
            ),
            DeclareLaunchArgument(
                "nyush_gimbal_verbose",
                default_value="false",
                description="Enable verbose NYUSH gimbal serial logs",
            ),
            DeclareLaunchArgument(
                "nyush_enemy_color",
                default_value="1",
                description="Enemy color code published on attitude_flags: 0 red, 1 blue",
            ),
            DeclareLaunchArgument(
                "nyush_rx_yaw_sign",
                default_value="-1",
                description="Sign applied to NYUSH RX yaw before publishing attitude_data",
            ),
            DeclareLaunchArgument(
                "nyush_rx_pitch_sign",
                default_value="-1",
                description="Sign applied to NYUSH RX pitch before publishing attitude_data",
            ),
            DeclareLaunchArgument(
                "nyush_rx_roll_sign",
                default_value="1",
                description="Sign applied to NYUSH RX roll before publishing attitude_data",
            ),
            DeclareLaunchArgument(
                "nyush_tx_yaw_sign",
                default_value="-1",
                description="Sign applied to solver yaw before NYUSH TX",
            ),
            DeclareLaunchArgument(
                "nyush_tx_pitch_sign",
                default_value="-1",
                description="Sign applied to solver pitch before NYUSH TX",
            ),
            make_basler_stack(),
            make_nyush_gimbal_bridge(),
        ]
    )
