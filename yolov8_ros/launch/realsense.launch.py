from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    package_share_dir = Path(get_package_share_directory("yolov8_ros"))
    default_model_path = str(package_share_dir / "models" / "best.engine")

    # --- Nodes ---
    camera_node = Node(
        package="yolov8_ros",
        executable="camera_node",
        name="camera_node",
        output="screen",
        parameters=[
            {
                "camera_type": "realsense",
                "imgsz_width": LaunchConfiguration("rs_width"),
                "imgsz_height": LaunchConfiguration("rs_height"),
                "realsense_width": LaunchConfiguration("rs_width"),
                "realsense_height": LaunchConfiguration("rs_height"),
                "realsense_fps": LaunchConfiguration("rs_fps"),
                "realsense_exposure": LaunchConfiguration("rs_exposure"),
                "realsense_gain": LaunchConfiguration("rs_gain"),
                "realsense_auto_exposure": LaunchConfiguration("rs_auto_exposure"),
                "realsense_auto_white_balance": LaunchConfiguration(
                    "rs_auto_white_balance"
                ),
                "realsense_white_balance": LaunchConfiguration("rs_white_balance"),
                "realsense_use_poll": LaunchConfiguration("rs_use_poll"),
            }
        ],
    )

    yolov8_node = Node(
        package="yolov8_ros",
        executable="yolov8_modular_node",
        name="yolov8_modular_node",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "task": LaunchConfiguration("task"),
                "conf": LaunchConfiguration("conf"),
                "show": LaunchConfiguration("show"),
                "use_kalman": LaunchConfiguration("use_kalman"),
                "enable_color_verification": LaunchConfiguration(
                    "enable_color_verification"
                ),
                "hfov_degrees": LaunchConfiguration("hfov_degrees"),
                "vfov_degrees": LaunchConfiguration("vfov_degrees"),
            }
        ],
    )

    return LaunchDescription(
        [
            # RealSense camera args
            DeclareLaunchArgument("rs_width", default_value="640", description="RealSense frame width"),
            DeclareLaunchArgument("rs_height", default_value="480", description="RealSense frame height"),
            DeclareLaunchArgument("rs_fps", default_value="60", description="RealSense frames per second"),
            DeclareLaunchArgument("rs_exposure", default_value="300.0", description="RealSense manual exposure (us)"),
            DeclareLaunchArgument("rs_gain", default_value="16.0", description="RealSense manual gain value"),
            DeclareLaunchArgument("rs_auto_exposure", default_value="false", description="Enable RealSense auto exposure"),
            DeclareLaunchArgument("rs_auto_white_balance", default_value="false", description="Enable RealSense auto white balance"),
            DeclareLaunchArgument("rs_white_balance", default_value="5800.0", description="RealSense manual white balance (K)"),
            DeclareLaunchArgument("rs_use_poll", default_value="false", description="Use poll_for_frames instead of wait_for_frames"),
            # YOLO args
            DeclareLaunchArgument("model_path", default_value=default_model_path, description="Path to YOLO model (.engine / .pt / .onnx)"),
            DeclareLaunchArgument("task", default_value="detect", description="YOLO task: detect, obb, segment"),
            DeclareLaunchArgument("conf", default_value="0.25", description="YOLO confidence threshold"),
            DeclareLaunchArgument("show", default_value="true", description="Show OpenCV visualization window"),
            DeclareLaunchArgument("use_kalman", default_value="false", description="Enable Kalman filter tracking"),
            DeclareLaunchArgument("enable_color_verification", default_value="false", description="Enable armor plate color verification"),
            # FOV — D435: hfov=69.4 vfov=42.5 | D455: hfov=87.0 vfov=58.0
            DeclareLaunchArgument("hfov_degrees", default_value="69.4", description="Camera horizontal FOV in degrees (D435=69.4, D455=87.0)"),
            DeclareLaunchArgument("vfov_degrees", default_value="42.5", description="Camera vertical FOV in degrees (D435=42.5, D455=58.0)"),
            # Nodes
            camera_node,
            yolov8_node,
        ]
    )
