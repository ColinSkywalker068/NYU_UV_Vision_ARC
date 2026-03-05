from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description() -> LaunchDescription:
    package_share_dir = Path(get_package_share_directory("yolov8_ros"))
    default_model_path = str(package_share_dir / "models" / "best.engine")

    # --- RealSense camera args ---
    rs_width_arg = DeclareLaunchArgument(
        "rs_width", default_value="640", description="RealSense frame width"
    )
    rs_height_arg = DeclareLaunchArgument(
        "rs_height", default_value="480", description="RealSense frame height"
    )
    rs_fps_arg = DeclareLaunchArgument(
        "rs_fps", default_value="30", description="RealSense frames per second"
    )
    rs_exposure_arg = DeclareLaunchArgument(
        "rs_exposure",
        default_value="120.0",
        description="RealSense manual exposure value",
    )
    rs_gain_arg = DeclareLaunchArgument(
        "rs_gain", default_value="64.0", description="RealSense manual gain value"
    )
    rs_auto_exposure_arg = DeclareLaunchArgument(
        "rs_auto_exposure",
        default_value="false",
        description="Enable RealSense auto exposure",
    )
    rs_auto_white_balance_arg = DeclareLaunchArgument(
        "rs_auto_white_balance",
        default_value="false",
        description="Enable RealSense auto white balance",
    )
    rs_white_balance_arg = DeclareLaunchArgument(
        "rs_white_balance",
        default_value="4600.0",
        description="RealSense manual white balance (K)",
    )
    rs_use_poll_arg = DeclareLaunchArgument(
        "rs_use_poll",
        default_value="false",
        description="Use poll_for_frames instead of wait_for_frames",
    )

    # --- YOLO args ---
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=default_model_path,
        description="Path to YOLO model (.engine / .pt / .onnx)",
    )
    task_arg = DeclareLaunchArgument(
        "task", default_value="detect", description="YOLO task: detect, obb, segment"
    )
    conf_arg = DeclareLaunchArgument(
        "conf", default_value="0.25", description="YOLO confidence threshold"
    )
    show_arg = DeclareLaunchArgument(
        "show", default_value="true", description="Show OpenCV visualization window"
    )
    use_kalman_arg = DeclareLaunchArgument(
        "use_kalman", default_value="false", description="Enable Kalman filter tracking"
    )
    enable_color_verification_arg = DeclareLaunchArgument(
        "enable_color_verification",
        default_value="false",
        description="Enable armor plate color verification",
    )

    # FOV — D435: hfov=69.4 vfov=42.5 | D455: hfov=87.0 vfov=58.0
    hfov_arg = DeclareLaunchArgument(
        "hfov_degrees",
        default_value="69.4",
        description="Camera horizontal FOV in degrees (D435=69.4, D455=87.0)",
    )
    vfov_arg = DeclareLaunchArgument(
        "vfov_degrees",
        default_value="42.5",
        description="Camera vertical FOV in degrees (D435=42.5, D455=58.0)",
    )

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
            rs_width_arg,
            rs_height_arg,
            rs_fps_arg,
            rs_exposure_arg,
            rs_gain_arg,
            rs_auto_exposure_arg,
            rs_auto_white_balance_arg,
            rs_white_balance_arg,
            rs_use_poll_arg,
            model_path_arg,
            task_arg,
            conf_arg,
            show_arg,
            use_kalman_arg,
            enable_color_verification_arg,
            hfov_arg,
            vfov_arg,
            camera_node,
            yolov8_node,
        ]
    )
