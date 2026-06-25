"""Launch the RM handoff-compatible detector, tracker, BT, and serial path."""

from typing import Any

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def package_config_path(file_name: str) -> PathJoinSubstitution:
    return PathJoinSubstitution([FindPackageShare("yolo_cpp"), "config", file_name])


def launch_config(name: str) -> LaunchConfiguration:
    return LaunchConfiguration(name)


def argument_specs() -> list[tuple[str, Any, str]]:
    return [
        ("camera_type", "realsense", "Camera backend: realsense or basler"),
        (
            "model_path",
            EnvironmentVariable("CV_STM_MODEL_PATH", default_value=""),
            "YOLO model or TensorRT engine path",
        ),
        ("detector_backend", "native_tensorrt", "Detector backend"),
        (
            "allow_native_tensorrt_fallback",
            "false",
            "Allow DeepStream startup to fall back to native TensorRT",
        ),
        (
            "deepstream_config_path",
            package_config_path("deepstream_bestmerge_nvinfer.txt"),
            "Optional DeepStream nvinfer config template path",
        ),
        ("task", "obb", "YOLO task"),
        ("conf", "0.30", "Detector confidence threshold"),
        (
            "use_model_color_classes",
            "true",
            "Use model class IDs directly for enemy color instead of HSV verification",
        ),
        (
            "enable_color_verification",
            "false",
            "Enable HSV enemy-color verification when use_model_color_classes is false",
        ),
        (
            "target_color",
            "blue",
            "Enemy color until STM/robot-id feedback overrides it",
        ),
        ("show", "false", "Preview mode: false, gui, or web"),
        ("preview_max_fps", "60.0", "Maximum preview render FPS"),
        ("debug_publish_image_raw", "false", "Publish debug image_raw frames"),
        ("save_target_frames", "false", "Save target-frame JPEGs"),
        ("target_frame_save_dir", "", "Optional target-frame output directory"),
        ("target_frame_save_max", "100", "Maximum target-frame images saved per run"),
        ("target_frame_jpeg_quality", "85", "JPEG quality for saved target frames"),
        ("enable_3d_pose", "true", "Solve detector-side PnP poses"),
        ("camera_frame_id", "camera_optical_frame", "Detector pose frame"),
        ("camera_info_topic", "camera_info", "CameraInfo topic"),
        ("camera_fx", "0.0", "Optional static camera fx"),
        ("camera_fy", "0.0", "Optional static camera fy"),
        ("camera_cx", "0.0", "Optional static camera cx"),
        ("camera_cy", "0.0", "Optional static camera cy"),
        (
            "rm_handoff_number_model_path",
            "/Users/Kushal/Projects/nyu_rm_sentry/rm_vision_ws/src/rm_auto_aim/armor_detector/model/mlp.onnx",
            "RM number classifier ONNX path",
        ),
        (
            "rm_handoff_number_label_path",
            "/Users/Kushal/Projects/nyu_rm_sentry/rm_vision_ws/src/rm_auto_aim/armor_detector/model/label.txt",
            "RM number classifier label path",
        ),
        (
            "rm_handoff_number_classifier_enabled",
            "true",
            "Enable the RM MLP armor-number classifier",
        ),
        (
            "rm_handoff_require_number_classifier",
            "true",
            "Drop detections when armor number cannot be classified",
        ),
        (
            "rm_handoff_classifier_threshold",
            "0.7",
            "RM number classifier confidence threshold",
        ),
        (
            "rm_handoff_default_number",
            "",
            "Fallback armor number when the classifier cannot produce one",
        ),
        ("target_frame", "gimbal_odom", "Tracker target frame"),
        ("tracker_max_match_distance", "0.15", "Maximum EKF match distance in meters"),
        ("tracker_max_match_yaw_diff", "1.0", "Maximum EKF match yaw diff in radians"),
        ("tracker_tracking_thres", "5", "Matched frames before TRACKING"),
        ("tracker_lost_time_thres", "0.3", "Seconds before TEMP_LOST becomes LOST"),
        ("serial_enable", "false", "Start the handoff serial bridge"),
        ("serial_port", "/dev/ttyUSB0", "Serial device"),
        ("serial_baudrate", "115200", "Serial baud rate"),
        ("bt_adapter_enable", "false", "Start the BT communication adapter"),
        ("realsense_width", "640", "RealSense width"),
        ("realsense_height", "480", "RealSense height"),
        ("realsense_fps", "60", "RealSense FPS"),
        ("realsense_exposure", "120.0", "RealSense exposure"),
        ("realsense_auto_exposure", "true", "RealSense auto exposure"),
        ("realsense_use_poll", "false", "Use RealSense poll_for_frames"),
        (
            "realsense_use_max_native_fps",
            "true",
            "Use best native RealSense FPS for the requested resolution",
        ),
        ("basler_width", "640", "Basler model/input width"),
        ("basler_height", "640", "Basler model/input height"),
        ("basler_pixel_format", "bayer_rggb8", "Basler pixel format"),
        ("basler_source_width", "1920", "Basler source width"),
        ("basler_source_height", "1200", "Basler source height"),
        ("basler_fps", "160", "Basler FPS"),
        ("basler_uncapped_fps", "false", "Disable Basler frame-rate cap"),
        ("basler_exposure_us", "5000.0", "Basler exposure in microseconds"),
        ("basler_gain_db", "5.0", "Basler gain in dB"),
        ("basler_flip", "true", "Rotate Basler frames 180 degrees"),
        ("basler_use_full_sensor", "true", "Use full Basler sensor"),
        ("basler_centered_roi", "true", "Center Basler ROI when cropped"),
        (
            "native_tensorrt_cuda_bayer_preprocess",
            "false",
            "Use CUDA Bayer preprocessing for native TensorRT Basler frames",
        ),
        ("pre_mux_resize", "true", "Resize full-FOV Basler frames before nvstreammux"),
        ("pre_mux_width", "0", "Optional pre-mux resize width"),
        ("pre_mux_height", "0", "Optional pre-mux resize height"),
        ("deepstream_profile_stages", "false", "Log DeepStream stage timings"),
        (
            "deepstream_profile_stage_interval",
            "300",
            "Frames between DeepStream timing summaries",
        ),
        (
            "bayer_conversion_backend",
            "cuda",
            "Bayer conversion backend for DeepStream",
        ),
    ]


def declare_arguments() -> list[DeclareLaunchArgument]:
    return [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in argument_specs()
    ]


def generate_launch_description() -> LaunchDescription:
    detector = Node(
        package="yolo_cpp",
        executable="armor_detector_node",
        name="armor_detector_node",
        output="screen",
        parameters=[
            {
                "camera_type": launch_config("camera_type"),
                "model_path": launch_config("model_path"),
                "detector_backend": launch_config("detector_backend"),
                "allow_native_tensorrt_fallback": launch_config(
                    "allow_native_tensorrt_fallback"
                ),
                "deepstream_config_path": launch_config("deepstream_config_path"),
                "task": launch_config("task"),
                "conf": launch_config("conf"),
                "use_model_color_classes": launch_config("use_model_color_classes"),
                "enable_color_verification": launch_config("enable_color_verification"),
                "target_color": launch_config("target_color"),
                "show": ParameterValue(launch_config("show"), value_type=str),
                "preview_max_fps": ParameterValue(
                    launch_config("preview_max_fps"),
                    value_type=float,
                ),
                "debug_publish_image_raw": launch_config("debug_publish_image_raw"),
                "save_target_frames": launch_config("save_target_frames"),
                "target_frame_save_dir": launch_config("target_frame_save_dir"),
                "target_frame_save_max": launch_config("target_frame_save_max"),
                "target_frame_jpeg_quality": launch_config("target_frame_jpeg_quality"),
                "enable_3d_pose": launch_config("enable_3d_pose"),
                "camera_frame_id": launch_config("camera_frame_id"),
                "camera_info_topic": launch_config("camera_info_topic"),
                "camera_fx": launch_config("camera_fx"),
                "camera_fy": launch_config("camera_fy"),
                "camera_cx": launch_config("camera_cx"),
                "camera_cy": launch_config("camera_cy"),
                "detector_output_contract": "rm_handoff",
                "rm_handoff_number_model_path": launch_config(
                    "rm_handoff_number_model_path"
                ),
                "rm_handoff_number_label_path": launch_config(
                    "rm_handoff_number_label_path"
                ),
                "rm_handoff_number_classifier_enabled": launch_config(
                    "rm_handoff_number_classifier_enabled"
                ),
                "rm_handoff_require_number_classifier": launch_config(
                    "rm_handoff_require_number_classifier"
                ),
                "rm_handoff_classifier_threshold": launch_config(
                    "rm_handoff_classifier_threshold"
                ),
                "rm_handoff_default_number": launch_config("rm_handoff_default_number"),
                "realsense_width": launch_config("realsense_width"),
                "realsense_height": launch_config("realsense_height"),
                "realsense_fps": launch_config("realsense_fps"),
                "realsense_exposure": launch_config("realsense_exposure"),
                "realsense_auto_exposure": launch_config("realsense_auto_exposure"),
                "realsense_use_poll": launch_config("realsense_use_poll"),
                "realsense_use_max_native_fps": launch_config(
                    "realsense_use_max_native_fps"
                ),
                "imgsz_width": launch_config("basler_width"),
                "imgsz_height": launch_config("basler_height"),
                "basler_pixel_format": launch_config("basler_pixel_format"),
                "basler_source_width": launch_config("basler_source_width"),
                "basler_source_height": launch_config("basler_source_height"),
                "basler_fps": launch_config("basler_fps"),
                "basler_uncapped_fps": launch_config("basler_uncapped_fps"),
                "basler_exposure_us": launch_config("basler_exposure_us"),
                "basler_gain_db": launch_config("basler_gain_db"),
                "basler_flip": launch_config("basler_flip"),
                "basler_use_full_sensor": launch_config("basler_use_full_sensor"),
                "basler_centered_roi": launch_config("basler_centered_roi"),
                "native_tensorrt_cuda_bayer_preprocess": launch_config(
                    "native_tensorrt_cuda_bayer_preprocess"
                ),
                "pre_mux_resize": launch_config("pre_mux_resize"),
                "pre_mux_width": launch_config("pre_mux_width"),
                "pre_mux_height": launch_config("pre_mux_height"),
                "deepstream_profile_stages": launch_config("deepstream_profile_stages"),
                "deepstream_profile_stage_interval": launch_config(
                    "deepstream_profile_stage_interval"
                ),
                "bayer_conversion_backend": launch_config("bayer_conversion_backend"),
            }
        ],
    )

    tracker = Node(
        package="yolo_cpp",
        executable="rm_handoff_tracker_node",
        name="armor_tracker",
        output="screen",
        parameters=[
            {
                "target_frame": launch_config("target_frame"),
                "tracker.max_match_distance": launch_config(
                    "tracker_max_match_distance"
                ),
                "tracker.max_match_yaw_diff": launch_config(
                    "tracker_max_match_yaw_diff"
                ),
                "tracker.tracking_thres": launch_config("tracker_tracking_thres"),
                "tracker.lost_time_thres": launch_config("tracker_lost_time_thres"),
            }
        ],
    )

    serial_bridge = Node(
        package="yolo_cpp",
        executable="rm_handoff_serial_bridge",
        name="rm_serial_driver",
        output="screen",
        condition=IfCondition(launch_config("serial_enable")),
        parameters=[
            {
                "device_name": launch_config("serial_port"),
                "baud_rate": launch_config("serial_baudrate"),
                "detect_color_nodes": ["armor_detector_node"],
            }
        ],
    )

    bt_adapter = Node(
        package="yolo_cpp",
        executable="bt_comm_adapter.py",
        name="bt_comm_adapter",
        output="screen",
        condition=IfCondition(launch_config("bt_adapter_enable")),
    )

    return LaunchDescription(
        [
            *declare_arguments(),
            detector,
            tracker,
            serial_bridge,
            bt_adapter,
        ]
    )
