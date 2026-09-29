"""Launch perception and robot tracking with no command or serial path."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Run the Basler detector, plate tracker, and robot estimator only."""
    basler_launch = PathJoinSubstitution(
        [FindPackageShare("yolo_cpp"), "launch", "basler.launch.py"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("show", default_value="false"),
            DeclareLaunchArgument("target_color", default_value="blue"),
            DeclareLaunchArgument("robot_armor_radius_m", default_value="0.26"),
            DeclareLaunchArgument("robot_armor_count", default_value="4"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(basler_launch),
                launch_arguments={
                    "start_solver": "false",
                    "start_serial_bridge": "false",
                    "transmit_enable": "false",
                    "receive_enable": "false",
                    "enable_fire_recommendation": "false",
                    "show": LaunchConfiguration("show"),
                    "target_color": LaunchConfiguration("target_color"),
                    "robot_estimator_enable": "true",
                    "robot_armor_radius_m": LaunchConfiguration(
                        "robot_armor_radius_m"
                    ),
                    "robot_armor_count": LaunchConfiguration("robot_armor_count"),
                }.items(),
            ),
        ]
    )
