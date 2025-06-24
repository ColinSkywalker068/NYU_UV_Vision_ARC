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
                        "model_path": "/home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb2.engine"
                    },
                    {"task": "obb"},
                    {"show": False},
                    {"conf": 0.25},
                ],
                
            ),
            # RealSense2 Camera with only two key parameters
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments={
                    # -- Core Performance Settings --

                    # Use a single profile to set resolution and FPS. This is more reliable.
                    "rgb_camera.profile": "640x480x60",

                    # Use a raw, uncompressed format to avoid CPU overhead from decompression.
                    # YUYV is a common and performant choice.
                    'rgb_camera.color_format': "YUYV",

                    # Disable all other streams to maximize bandwidth and performance for the color stream.
                    "enable_depth": "false",
                    'enable_infra1': "false",
                    'enable_infra2': "false",

                    # Disable sync; it's not needed when only one stream (color) is active.
                    "enable_sync": "false",
                    
                    # Disable pointcloud and align_depth, which are computationally expensive.
                    "pointcloud.enable": "false",
                    "align_depth.enable": "false",

                    # -- Image Quality Settings (with minimal performance impact) --
                    "rgb_camera.enable_auto_exposure": "true",
                    "rgb_camera.enable_auto_white_balance": "true",
                    
                    # This setting can sometimes help with timestamping consistency.
                    "rgb_camera.global_time_enabled": "true",

                    'rgb_camera.width': "640",
                    'rgb_camera.height': "480",
                    'rgb_camera.fps': "60",

                    #"rgb_camera.gain": "0",  # Gain set to 0
                    #"rgb_camera.gamma": "250",  # Gamma correction
                    #"rgb_camera.white_balance": "4800",  # White balance in K
                }.items(),
            ),
        ]
    )
