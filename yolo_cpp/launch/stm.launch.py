from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    camera_type_arg = DeclareLaunchArgument(
        "camera_type",
        default_value="realsense",
        description="Camera backend: realsense or basler",
    )
    detector_backend_arg = DeclareLaunchArgument(
        "detector_backend",
        default_value="native_tensorrt",
        description="Detector backend: deepstream or native_tensorrt",
    )
    allow_native_fallback_arg = DeclareLaunchArgument(
        "allow_native_tensorrt_fallback",
        default_value="false",
        description="Allow DeepStream startup to fall back to native TensorRT",
    )
    deepstream_config_arg = DeclareLaunchArgument(
        "deepstream_config_path",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("yolo_cpp"),
                "config",
                "deepstream_bestmerge_nvinfer.txt",
            ]
        ),
        description="Optional DeepStream nvinfer config template path",
    )
    serial_port_arg = DeclareLaunchArgument(
        "serial_port",
        default_value="/dev/ttyUSB0",
        description="Serial port for STM32 connection",
    )
    serial_baudrate_arg = DeclareLaunchArgument(
        "serial_baudrate",
        default_value="115200",
        description="Baud rate for STM32 serial",
    )
    serial_verbose_arg = DeclareLaunchArgument(
        "serial_verbose",
        default_value="false",
        description="Enable verbose serial logging",
    )
    basler_width_arg = DeclareLaunchArgument(
        "basler_width",
        default_value="640",
        description="Basler frame width",
    )
    basler_height_arg = DeclareLaunchArgument(
        "basler_height",
        default_value="640",
        description="Basler frame height",
    )
    basler_fps_arg = DeclareLaunchArgument(
        "basler_fps",
        default_value="60",
        description="Basler frames per second",
    )
    basler_exposure_arg = DeclareLaunchArgument(
        "basler_exposure_us",
        default_value="5000.0",
        description="Basler exposure in microseconds",
    )
    basler_gain_arg = DeclareLaunchArgument(
        "basler_gain_db",
        default_value="5.0",
        description="Basler gain in dB",
    )
    basler_flip_arg = DeclareLaunchArgument(
        "basler_flip",
        default_value="true",
        description="Rotate Basler frames 180 degrees to compensate for upside-down mounting",
    )
    realsense_width_arg = DeclareLaunchArgument(
        "realsense_width",
        default_value="640",
        description="RealSense frame width",
    )
    realsense_height_arg = DeclareLaunchArgument(
        "realsense_height",
        default_value="480",
        description="RealSense frame height",
    )
    realsense_fps_arg = DeclareLaunchArgument(
        "realsense_fps",
        default_value="60",
        description="RealSense frames per second",
    )
    realsense_exposure_arg = DeclareLaunchArgument(
        "realsense_exposure",
        default_value="180.0",
        description="RealSense manual exposure",
    )
    realsense_auto_exposure_arg = DeclareLaunchArgument(
        "realsense_auto_exposure",
        default_value="false",
        description="Enable RealSense auto exposure",
    )
    realsense_use_poll_arg = DeclareLaunchArgument(
        "realsense_use_poll",
        default_value="false",
        description="Use non-blocking RealSense poll_for_frames() instead of wait_for_frames()",
    )
    realsense_use_max_native_fps_arg = DeclareLaunchArgument(
        "realsense_use_max_native_fps",
        default_value="true",
        description="Use the best native RealSense FPS for the requested resolution",
    )
    inference_mode_arg = DeclareLaunchArgument(
        "inference_mode",
        default_value="predict",
        description="Detector inference mode",
    )
    show_arg = DeclareLaunchArgument(
        "show",
        default_value="false",
        description="Show OpenCV visualization window",
    )
    track_min_hits_arg = DeclareLaunchArgument(
        "track_min_hits",
        default_value="2",
        description="Frames required to confirm a track",
    )
    track_max_coast_frames_arg = DeclareLaunchArgument(
        "track_max_coast_frames",
        default_value="3",
        description="Maximum published coast frames",
    )
    track_drop_frames_arg = DeclareLaunchArgument(
        "track_drop_frames",
        default_value="8",
        description="Frames before dropping an unmatched track",
    )
    association_max_center_distance_ratio_arg = DeclareLaunchArgument(
        "association_max_center_distance_ratio",
        default_value="0.20",
        description="Maximum normalized center distance for matching",
    )
    association_min_iou_arg = DeclareLaunchArgument(
        "association_min_iou",
        default_value="0.10",
        description="Minimum IoU required for matching",
    )
    target_switch_min_score_delta_arg = DeclareLaunchArgument(
        "target_switch_min_score_delta",
        default_value="0.15",
        description="Confidence delta required to steal target lock",
    )
    target_lock_hysteresis_frames_arg = DeclareLaunchArgument(
        "target_lock_hysteresis_frames",
        default_value="2",
        description="Coast frames allowed before target lock yields",
    )
    hfov_arg = DeclareLaunchArgument(
        "hfov_degrees", default_value="79.2", description="Horizontal FOV degrees"
    )
    vfov_arg = DeclareLaunchArgument(
        "vfov_degrees", default_value="63.78", description="Vertical FOV degrees"
    )
    publish_zero_angles_arg = DeclareLaunchArgument(
        "publish_zero_angles_when_no_target",
        default_value="true",
        description="Publish zero yaw/pitch when there is no selected target",
    )

    detector_node = Node(
        package="yolo_cpp",
        executable="armor_detector_node",
        name="armor_detector_node",
        output="screen",
        parameters=[
            {
                "camera_type": LaunchConfiguration("camera_type"),
                "detector_backend": LaunchConfiguration("detector_backend"),
                "allow_native_tensorrt_fallback": LaunchConfiguration(
                    "allow_native_tensorrt_fallback"
                ),
                "deepstream_config_path": LaunchConfiguration("deepstream_config_path"),
                "imgsz_width": LaunchConfiguration("realsense_width"),
                "imgsz_height": LaunchConfiguration("realsense_height"),
                "basler_fps": LaunchConfiguration("basler_fps"),
                "basler_exposure_us": LaunchConfiguration("basler_exposure_us"),
                "basler_gain_db": LaunchConfiguration("basler_gain_db"),
                "basler_flip": LaunchConfiguration("basler_flip"),
                "basler_use_full_sensor": True,
                "basler_centered_roi": True,
                "realsense_width": LaunchConfiguration("realsense_width"),
                "realsense_height": LaunchConfiguration("realsense_height"),
                "realsense_fps": LaunchConfiguration("realsense_fps"),
                "realsense_exposure": LaunchConfiguration("realsense_exposure"),
                "realsense_auto_exposure": LaunchConfiguration(
                    "realsense_auto_exposure"
                ),
                "realsense_use_poll": LaunchConfiguration("realsense_use_poll"),
                "realsense_use_max_native_fps": LaunchConfiguration(
                    "realsense_use_max_native_fps"
                ),
                "show": LaunchConfiguration("show"),
                "inference_mode": LaunchConfiguration("inference_mode"),
            }
        ],
    )

    tracker_node = Node(
        package="yolo_cpp",
        executable="armor_tracker_node",
        name="armor_tracker_node",
        output="screen",
        parameters=[
            {
                "track_min_hits": LaunchConfiguration("track_min_hits"),
                "track_max_coast_frames": LaunchConfiguration("track_max_coast_frames"),
                "track_drop_frames": LaunchConfiguration("track_drop_frames"),
                "association_max_center_distance_ratio": LaunchConfiguration(
                    "association_max_center_distance_ratio"
                ),
                "association_min_iou": LaunchConfiguration("association_min_iou"),
                "target_switch_min_score_delta": LaunchConfiguration(
                    "target_switch_min_score_delta"
                ),
                "target_lock_hysteresis_frames": LaunchConfiguration(
                    "target_lock_hysteresis_frames"
                ),
            }
        ],
    )

    solver_node = Node(
        package="yolo_cpp",
        executable="aim_solver_node",
        name="aim_solver_node",
        output="screen",
        parameters=[
            {
                "hfov_degrees": LaunchConfiguration("hfov_degrees"),
                "vfov_degrees": LaunchConfiguration("vfov_degrees"),
                "publish_zero_angles_when_no_target": LaunchConfiguration(
                    "publish_zero_angles_when_no_target"
                ),
                "default_image_width": LaunchConfiguration("realsense_width"),
                "default_image_height": LaunchConfiguration("realsense_height"),
            }
        ],
    )

    stm_bridge_node = Node(
        package="yolo_cpp",
        executable="stm32_serial_bridge",
        name="stm32_serial_bridge_node",
        output="screen",
        parameters=[
            {
                "transmit_enable": True,
                "serial_port": LaunchConfiguration("serial_port"),
                "serial_baudrate": LaunchConfiguration("serial_baudrate"),
                "serial_verbose": LaunchConfiguration("serial_verbose"),
            }
        ],
    )

    return LaunchDescription(
        [
            camera_type_arg,
            detector_backend_arg,
            allow_native_fallback_arg,
            deepstream_config_arg,
            serial_port_arg,
            serial_baudrate_arg,
            serial_verbose_arg,
            basler_width_arg,
            basler_height_arg,
            basler_fps_arg,
            basler_exposure_arg,
            basler_gain_arg,
            basler_flip_arg,
            realsense_width_arg,
            realsense_height_arg,
            realsense_fps_arg,
            realsense_exposure_arg,
            realsense_auto_exposure_arg,
            realsense_use_poll_arg,
            realsense_use_max_native_fps_arg,
            inference_mode_arg,
            show_arg,
            track_min_hits_arg,
            track_max_coast_frames_arg,
            track_drop_frames_arg,
            association_max_center_distance_ratio_arg,
            association_min_iou_arg,
            target_switch_min_score_delta_arg,
            target_lock_hysteresis_frames_arg,
            hfov_arg,
            vfov_arg,
            publish_zero_angles_arg,
            detector_node,
            tracker_node,
            solver_node,
            stm_bridge_node,
        ]
    )
