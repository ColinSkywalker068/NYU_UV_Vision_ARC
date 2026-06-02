from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    actions = []

    # YOLOv8 Modular Node (Basler)
    actions.append(
        Node(
            package="yolov8_ros",
            executable="yolov8_modular_node",
            name="yolov8_modular_node",
            parameters=[
                {
                    "model_path": "/home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine"
                },
                {"task": "obb"},
                {"show": False},
                {"conf": 0.25},
                {"imgsz_width": 640},
                {"imgsz_height": 640},
                {"basler_fps": 30},
            ],
        )
    )

    return LaunchDescription(actions)
