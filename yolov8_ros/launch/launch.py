from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
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
                    {"show": True},
                    {"conf": 0.5},
                ],
            ),
            Node()
        ]
    )
