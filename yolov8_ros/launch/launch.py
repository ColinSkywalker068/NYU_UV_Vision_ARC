from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
import os


def generate_launch_description():
    actions = []

    # YOLOv8 Node
    actions.append(
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
                {"show_video": True},
                {"conf": 0.25},
            ],
        )
    )

    # Try to include RealSense camera launch if available; skip if package is missing
    try:
        realsense_launch_file = os.path.join(
            get_package_share_directory("realsense2_camera"),
            "launch",
            "rs_launch.py",
        )
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(realsense_launch_file),
                launch_arguments={
                    "rgb_camera.color_profile": "640x480x60",
                }.items(),
            )
        )
    except PackageNotFoundError:
        pass

    return LaunchDescription(actions)
