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
                executable="yolov8_noRos",
                name="yolov8_noRos",
                # remappings=[("video_frames", "/camera/camera/color/image_raw")],
                parameters=[
                    {
                        "model_path": "/home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb2.engine"
                    },
                    {"task": "obb"},
                    {"show": False},
                    {"conf": 0.05},
                ],
        
            ),
        ]
    )
