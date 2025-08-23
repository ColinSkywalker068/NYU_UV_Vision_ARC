from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the path to the RealSense2 camera launch file
    realsense_launch_file = os.path.join(
        get_package_share_directory("realsense2_camera"),
        "launch",
        "rs_launch.py",
    )

    return LaunchDescription(
        [
            # Frame Grabber Node
            Node(
                package="yolov8_ros",
                executable="frame_grabber",
                name="frame_grabber_node",
                output="screen",  # Show node logs in the terminal
                parameters=[
                    # Match the unusual topic name from your other launch file
                    {"image_topic": "/camera/camera/color/image_raw"}
                ],
            ),
            # RealSense2 Camera Launch
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments={
                    # Configure camera profile
                    "rgb_camera.color_profile": "640x480x60",
                    # Disable all streams except the color camera to save resources
                    "enable_depth": "false",
                    "enable_infra1": "false",
                    "enable_infra2": "false",
                    "enable_gyro": "false",
                    "enable_accel": "false",
                    "pointcloud.enable": "false",
                    # Add critical camera settings from your other launch file
                    "rgb_camera.enable_auto_exposure": "true",
                    "rgb_camera.enable_auto_white_balance": "true",
                    "rgb_camera.global_time_enabled": "true",
                }.items(),
            ),
        ]
    )
