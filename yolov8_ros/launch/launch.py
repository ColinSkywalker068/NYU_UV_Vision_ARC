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
                        "model_path": "/home/ultraviolet/CV_Detection/yolov8_ros/yolov8_ros/bestNewTR.pt"
                    },
                    {"task": "detect"},
                    {"show": True},
                    {"conf": 0},
                ],
            ),
            # Include the Realsense2 Camera Launch File
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments={
                    "rgb_camera.profile": "640x480x30",
                    "depth_module.profile": "640x480x30",  # Setting a profile for the depth module
                    # "depth_module.enable": True,  # Re-enable the depth module
                    # "pointcloud.enable": False,
                    # "enable_debug": False,
                    # "log_level": "info",  # Changed log_level to "info" to reduce verbosity
                }.items(),
            ),
        ]
    )
