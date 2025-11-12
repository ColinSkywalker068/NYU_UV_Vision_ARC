from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import (
    get_package_share_directory,
    PackageNotFoundError,
)
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
                    "model_path": "/home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine"
                },
                {"task": "obb"},
                {"show": True},
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
                    # Stream profiles: keep only color at lower res/fps to reduce USB/CPU
                    "rgb_camera.profile": "640x480x60",
                    # Disable unused streams and features to cut overhead
                    "enable_depth": "false",
                    "enable_infra1": "false",
                    "enable_infra2": "false",
                    "enable_gyro": "false",
                    "enable_accel": "false",
                    "enable_sync": "false",
                    # Turn off filters and heavy processing
                    "pointcloud.enable": "false",
                    "align_depth.enable": "false",
                    "colorizer.enable": "false",
                    # Keep queue small to lower latency/memory
                    "frames_queue_size": "2",
                    # QoS optimized for sensor data
                    "color_qos": "SENSOR_DATA",
                    # Keep constant fps under AE (avoid dropping fps in low light)
                    "rgb_camera.enable_auto_exposure": "true",
                    "rgb_camera.auto_exposure_priority": "false",
                    # Minor overhead saving, not using TFs downstream
                    "publish_tf": "false",
                }.items(),
            )
        )
    except PackageNotFoundError:
        pass

    return LaunchDescription(actions)
