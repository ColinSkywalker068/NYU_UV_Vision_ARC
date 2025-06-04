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
                    # --- Model and Basic YOLO Parameters ---
                    {"model_path": "/home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb.pt"}, # Your model
                    {"task": "obb"},          # Your task: "detect" or "obb"
                    {"type": "yolo"},         # Model type: "yolo", "rtdetr", "nas" (default: yolo)
                    {"imgsz": 640},           # Inference image size (default: 640)
                    {"conf": 0.05},           # Confidence threshold (default in code is 0.25, your launch overrides)
                    {"half": True},           # Use FP16 precision (default: True)
                    {"show": True},           # Enable OpenCV visualization windows (default: True)

                    # --- Kalman Filter and Tracking Parameters ---
                    {"tracker_timeout": 15},                   # Default: 15 frames
                    {"kalman_initial_dt": 0.0333},             # Default: 1.0/30.0 (for ~30 FPS)
                    {"prediction_lookahead_steps": 0.5},       # Default: 0.5 (predict 0.5 * dt ahead)
                    {"target_selection_strategy": "closest_to_center"}, # Options: "first", "biggest_area", "closest_to_center" (default: "first")

                    # --- SAHI (Slicing Aided Hyper Inference) Parameters ---
                    {"use_sahi": False},                        # Set to True to enable SAHI (default: False)
                    {"sahi_slice_height": 320},               # Slice height for SAHI (default: 512)
                    {"sahi_slice_width": 320},                # Slice width for SAHI (default: 512)
                    {"sahi_overlap_height_ratio": 0.1},       # Overlap ratio for height (default: 0.2)
                    {"sahi_overlap_width_ratio": 0.1},        # Overlap ratio for width (default: 0.2)
                    {"sahi_nms_iou_threshold": 0.4},          # NMS IoU threshold for SAHI merged detections (default: 0.5)
                ]
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
