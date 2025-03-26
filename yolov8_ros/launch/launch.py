# Test Merge
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the path to the realsense2_camera launch file
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
                remappings=[
                    ("video_frames", "/camera/camera/color/image_raw"),
                ],
                parameters=[
                    {
                        "model_path": "/home/ultraviolet/CV_Detection/yolov8_ros/yolov8_ros/best.engine"
                    },
                    {"task": "detect"},
                    {"show": False},
                    {"conf": 0.5},
                ],
            ),
            # Include the Realsense2 Camera Launch File
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments={
                    "rgb_camera.profile": "640x480x30",
                    "depth_module.enable": False,
                    "pointcloud.enable": False,
                    "enable_debug": False,
                    "log_level": "debug",
                }.items(),
            ),
        ]
    )
