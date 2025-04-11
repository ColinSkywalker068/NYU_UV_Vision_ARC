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
            # YOLOv8 Node
            Node(
                package="yolov8_ros",
                executable="yolov8_node",
                name="yolov8_node",
                remappings=[("video_frames", "/camera/camera/color/image_raw")],
                parameters=[
                    {
                        "model_path": "/home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb2.pt"
                    },
                    {"task": "obb"},
                    {"show": True},
                    {"conf": 0.2},
                ],
            ),
            # RealSense2 Camera with only two key parameters
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments={
                    "rgb_camera.color_profile": "640x480x30",
                    "rgb_camera.enable_auto_exposure": "false",  # Manual exposure
                    "enable_depth": "false",  # Turn off depth
                    "rgb_camera.exposure": "100",  # Exposure value
                    "rgb_camera.brightness": "50",  # Brightness tweak
                    "rgb_camera.enable_auto_white_balance": "false",  # Manual WB
                    "rgb_camera.global_time_enabled": "false",  # Disable global timing
                    "rgb_camera.gain": "0",  # Gain set to 0
                    "rgb_camera.gamma": "250",  # Gamma correction
                    "rgb_camera.white_balance": "4800",  # White balance in K
                }.items(),
            ),
        ]
    )
