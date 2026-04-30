from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_model_path = "/home/ultraviolet/CV_Detection/model_test/bestmerge.engine"

    rs_width_arg = DeclareLaunchArgument(
        "rs_width", default_value="640", description="RealSense color width"
    )
    rs_height_arg = DeclareLaunchArgument(
        "rs_height", default_value="480", description="RealSense color height"
    )
    rs_fps_arg = DeclareLaunchArgument(
        "rs_fps", default_value="60", description="RealSense color FPS"
    )
    rs_exposure_arg = DeclareLaunchArgument(
        "rs_exposure", default_value="120.0", description="RealSense manual exposure"
    )
    rs_auto_exposure_arg = DeclareLaunchArgument(
        "rs_auto_exposure",
        default_value="true",
        description="Enable RealSense auto exposure",
    )
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=default_model_path,
        description="Path to YOLO weights (.pt / .engine / .onnx)",
    )
    task_arg = DeclareLaunchArgument(
        "task", default_value="obb", description="YOLO task: detect, obb, segment"
    )
    conf_arg = DeclareLaunchArgument(
        "conf", default_value="0.25", description="Confidence threshold"
    )
    show_arg = DeclareLaunchArgument(
        "show", default_value="false", description="OpenCV debug window"
    )
    use_kalman_arg = DeclareLaunchArgument(
        "use_kalman", default_value="false", description="Kalman-smoothed publishing"
    )
    hfov_arg = DeclareLaunchArgument(
        "hfov_degrees", default_value="69.4", description="Horizontal FOV deg"
    )
    vfov_arg = DeclareLaunchArgument(
        "vfov_degrees", default_value="42.5", description="Vertical FOV deg"
    )

    perception_node = Node(
        package="yolov8_ros_cpp",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[
            {
                "camera_type": "realsense",
                "realsense_width": LaunchConfiguration("rs_width"),
                "realsense_height": LaunchConfiguration("rs_height"),
                "realsense_fps": LaunchConfiguration("rs_fps"),
                "realsense_exposure": LaunchConfiguration("rs_exposure"),
                "realsense_auto_exposure": LaunchConfiguration("rs_auto_exposure"),
                "model_path": LaunchConfiguration("model_path"),
                "task": LaunchConfiguration("task"),
                "conf": LaunchConfiguration("conf"),
                "show": LaunchConfiguration("show"),
                "use_kalman": LaunchConfiguration("use_kalman"),
                "hfov_degrees": LaunchConfiguration("hfov_degrees"),
                "vfov_degrees": LaunchConfiguration("vfov_degrees"),
                "imgsz_width": LaunchConfiguration("rs_width"),
                "imgsz_height": LaunchConfiguration("rs_height"),
            }
        ],
    )

    return LaunchDescription(
        [
            rs_width_arg,
            rs_height_arg,
            rs_fps_arg,
            rs_exposure_arg,
            rs_auto_exposure_arg,
            model_path_arg,
            task_arg,
            conf_arg,
            show_arg,
            use_kalman_arg,
            hfov_arg,
            vfov_arg,
            perception_node,
        ]
    )
