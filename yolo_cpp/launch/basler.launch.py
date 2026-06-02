"""Launch the full Basler + DeepStream + STM32 auto-aim stack."""

from typing import Any

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


DEFAULT_MODEL_PATH = (
    "/home/ultraviolet/CV_Detection/model_test/"
    "best-roboflow-nitish-obb.onnx_b1_gpu0_fp16.engine"
)


def package_config_path(file_name: str) -> PathJoinSubstitution:
    """Return a path inside the installed yolo_cpp config directory."""
    return PathJoinSubstitution([FindPackageShare("yolo_cpp"), "config", file_name])


def launch_config(name: str) -> LaunchConfiguration:
    """Keep launch parameter dictionaries compact and readable."""
    return LaunchConfiguration(name)


def argument_specs() -> list[tuple[str, Any, str]]:
    """Declare launch arguments in operator-facing order."""
    return [
        (
            "camera_type",
            "basler",
            "Camera backend. Basler is the canonical default for this launch file.",
        ),
        (
            "detector_backend",
            "deepstream",
            "Detector backend: deepstream or native_tensorrt",
        ),
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
        (
            "model_path",
            DEFAULT_MODEL_PATH,
            "Path to YOLO weights (.pt / .engine / .onnx)",
        ),
        ("serial_port", "/dev/ttyUSB0", "Serial port for STM32 connection"),
        ("serial_baudrate", "115200", "Baud rate for STM32 serial"),
        ("serial_verbose", "false", "Enable verbose serial logging"),
        ("basler_width", "640", "Basler frame width"),
        ("basler_height", "640", "Basler frame height"),
        ("basler_fps", "110", "Basler frames per second"),
        (
            "basler_uncapped_fps",
            "true",
            "Disable Basler frame-rate limiting and let exposure set camera speed",
        ),
        (
            "basler_pixel_format",
            "bayer_rggb8",
            "Basler pixel format: bayer_rggb8, yuy2, mono8, bgr8, or rgb8",
        ),
        (
            "basler_source_width",
            "1920",
            "Basler source/camera width used when basler_use_full_sensor is true",
        ),
        (
            "basler_source_height",
            "1200",
            "Basler source/camera height used when basler_use_full_sensor is true",
        ),
        (
            "pre_mux_resize",
            "true",
            "Resize full-FOV Basler frames before nvstreammux pads them to the model input",
        ),
        (
            "pre_mux_width",
            "0",
            "Optional pre-mux resize bounding width; 0 uses the model width",
        ),
        (
            "pre_mux_height",
            "0",
            "Optional pre-mux resize bounding height; 0 uses the model height",
        ),
        (
            "deepstream_profile_stages",
            "false",
            "Log DeepStream per-stage pad timing for conversion bottleneck diagnosis",
        ),
        (
            "deepstream_profile_stage_interval",
            "300",
            "Frames between DeepStream stage-profile log summaries",
        ),
        (
            "bayer_conversion_backend",
            "cuda",
            "Bayer conversion backend for DeepStream: cuda/gpu or cpu/bayer2rgb",
        ),
        ("basler_exposure_us", "5000.0", "Basler exposure in microseconds"),
        ("basler_gain_db", "5.0", "Basler gain in dB"),
        ("basler_flip", "true", "Rotate Basler frames 180 degrees"),
        (
            "basler_use_full_sensor",
            "true",
            "Use the full Basler sensor instead of cropping the camera to the model input size",
        ),
        (
            "basler_centered_roi",
            "true",
            "Center the Basler ROI when basler_use_full_sensor is false",
        ),
        ("realsense_width", "640", "RealSense frame width"),
        ("realsense_height", "480", "RealSense frame height"),
        ("realsense_fps", "60", "RealSense frames per second"),
        ("realsense_exposure", "100.0", "RealSense manual exposure"),
        ("realsense_auto_exposure", "false", "Enable RealSense auto exposure"),
        (
            "realsense_use_poll",
            "false",
            "Use non-blocking RealSense poll_for_frames() instead of wait_for_frames()",
        ),
        (
            "realsense_use_max_native_fps",
            "true",
            "Use the best native RealSense FPS for the requested resolution",
        ),
        ("inference_mode", "predict", "Detector inference mode"),
        (
            "debug_publish_image_raw",
            "false",
            "Publish debug image_raw frame from detector",
        ),
        (
            "save_target_frames",
            "false",
            "Save up to target_frame_save_max JPEG frames when a target is found",
        ),
        (
            "target_frame_save_dir",
            "",
            "Optional directory for saved target frames; default is pictures next to vision_logs",
        ),
        ("target_frame_save_max", "100", "Maximum target-frame images saved per run"),
        ("target_frame_jpeg_quality", "85", "JPEG quality for saved target frames"),
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
            "Enemy armor color used until STM or robot_id feedback overrides it: blue or red",
        ),
        ("show", "true", "Show OpenCV visualization window"),
        (
            "preview_max_fps",
            "60.0",
            "Maximum OpenCV preview render FPS when show is enabled",
        ),
        ("track_min_hits", "1", "Frames required to confirm a track"),
        ("track_max_coast_frames", "1", "Maximum published coast frames"),
        ("track_drop_frames", "4", "Frames before dropping an unmatched track"),
        (
            "association_max_center_distance_ratio",
            "0.30",
            "Maximum normalized center distance for matching",
        ),
        ("association_min_iou", "0.0", "Minimum IoU required for matching"),
        (
            "target_switch_min_score_delta",
            "0.02",
            "Confidence delta required to steal target lock",
        ),
        (
            "target_lock_hysteresis_frames",
            "0",
            "Coast frames allowed before target lock yields",
        ),
        (
            "association_max_3d_distance_m",
            "0.35",
            "Maximum 3D association distance when PnP pose is available",
        ),
        ("hfov_degrees", "79.2", "Horizontal FOV degrees"),
        ("vfov_degrees", "63.78", "Vertical FOV degrees"),
        (
            "enable_3d_pose",
            "true",
            "Enable armor corner PnP pose solving in the detector",
        ),
        (
            "camera_frame_id",
            "camera_optical_frame",
            "Frame id attached to detector PnP poses",
        ),
        (
            "camera_info_topic",
            "camera_info",
            "CameraInfo topic used for calibrated PnP intrinsics",
        ),
        (
            "camera_fx",
            "0.0",
            "Optional static camera fx; FOV fallback is used when unset",
        ),
        (
            "camera_fy",
            "0.0",
            "Optional static camera fy; FOV fallback is used when unset",
        ),
        ("camera_cx", "0.0", "Optional static camera cx"),
        ("camera_cy", "0.0", "Optional static camera cy"),
        (
            "processor_target_frame",
            "camera_optical_frame",
            "Frame used for processor target pose output",
        ),
        (
            "use_3d_solution",
            "true",
            "Use processor pose/velocity when available for yaw/pitch solving",
        ),
        (
            "solver_output_mode",
            "absolute",
            "Aim output mode: offset or absolute. Absolute adds STM attitude yaw/pitch.",
        ),
        (
            "default_bullet_speed_mps",
            "18.0",
            "Fallback bullet speed when STM feedback does not provide one",
        ),
        ("solver_latency_s", "0.03", "Latency used by the 3D ballistic prediction"),
        (
            "compensate_gravity",
            "true",
            "Enable simple gravity drop compensation in the 3D solver",
        ),
        (
            "publish_zero_angles_when_no_target",
            "true",
            "Publish zero yaw/pitch when there is no selected target",
        ),
        (
            "transmit_angles_in_degrees",
            "false",
            "Convert solver radians to degrees before STM TX",
        ),
        (
            "hold_fire_mode",
            "0",
            "Fire mode sent while the solver suggests holding fire",
        ),
    ]


def declare_arguments() -> list[DeclareLaunchArgument]:
    """Build all launch argument actions from the table above."""
    return [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in argument_specs()
    ]


def make_detector_node() -> Node:
    """Create the detector node with Basler, DeepStream, and debug parameters."""
    return Node(
        package="yolo_cpp",
        executable="armor_detector_node",
        name="armor_detector_node",
        output="screen",
        parameters=[
            {
                "camera_type": launch_config("camera_type"),
                "detector_backend": launch_config("detector_backend"),
                "allow_native_tensorrt_fallback": launch_config(
                    "allow_native_tensorrt_fallback"
                ),
                "model_path": launch_config("model_path"),
                "deepstream_config_path": launch_config("deepstream_config_path"),
                "imgsz_width": launch_config("basler_width"),
                "imgsz_height": launch_config("basler_height"),
                "basler_fps": launch_config("basler_fps"),
                "basler_uncapped_fps": launch_config("basler_uncapped_fps"),
                "basler_pixel_format": launch_config("basler_pixel_format"),
                "basler_source_width": launch_config("basler_source_width"),
                "basler_source_height": launch_config("basler_source_height"),
                "pre_mux_resize": launch_config("pre_mux_resize"),
                "pre_mux_width": launch_config("pre_mux_width"),
                "pre_mux_height": launch_config("pre_mux_height"),
                "deepstream_profile_stages": launch_config("deepstream_profile_stages"),
                "deepstream_profile_stage_interval": launch_config(
                    "deepstream_profile_stage_interval"
                ),
                "bayer_conversion_backend": launch_config("bayer_conversion_backend"),
                "basler_exposure_us": launch_config("basler_exposure_us"),
                "basler_gain_db": launch_config("basler_gain_db"),
                "basler_flip": launch_config("basler_flip"),
                "basler_use_full_sensor": launch_config("basler_use_full_sensor"),
                "basler_centered_roi": launch_config("basler_centered_roi"),
                "realsense_width": launch_config("realsense_width"),
                "realsense_height": launch_config("realsense_height"),
                "realsense_fps": launch_config("realsense_fps"),
                "realsense_exposure": launch_config("realsense_exposure"),
                "realsense_auto_exposure": launch_config("realsense_auto_exposure"),
                "realsense_use_poll": launch_config("realsense_use_poll"),
                "realsense_use_max_native_fps": launch_config(
                    "realsense_use_max_native_fps"
                ),
                "hfov_degrees": launch_config("hfov_degrees"),
                "vfov_degrees": launch_config("vfov_degrees"),
                "enable_3d_pose": launch_config("enable_3d_pose"),
                "camera_frame_id": launch_config("camera_frame_id"),
                "camera_info_topic": launch_config("camera_info_topic"),
                "camera_fx": launch_config("camera_fx"),
                "camera_fy": launch_config("camera_fy"),
                "camera_cx": launch_config("camera_cx"),
                "camera_cy": launch_config("camera_cy"),
                "show": launch_config("show"),
                "preview_max_fps": ParameterValue(
                    launch_config("preview_max_fps"),
                    value_type=float,
                ),
                "debug_publish_image_raw": launch_config("debug_publish_image_raw"),
                "save_target_frames": launch_config("save_target_frames"),
                "target_frame_save_dir": launch_config("target_frame_save_dir"),
                "target_frame_save_max": launch_config("target_frame_save_max"),
                "target_frame_jpeg_quality": launch_config("target_frame_jpeg_quality"),
                "use_model_color_classes": launch_config("use_model_color_classes"),
                "enable_color_verification": launch_config("enable_color_verification"),
                "target_color": launch_config("target_color"),
                "inference_mode": launch_config("inference_mode"),
            }
        ],
    )


def make_tracker_node() -> Node:
    """Create the temporal tracker node."""
    return Node(
        package="yolo_cpp",
        executable="armor_tracker_node",
        name="armor_tracker_node",
        output="screen",
        parameters=[
            {
                "track_min_hits": launch_config("track_min_hits"),
                "track_max_coast_frames": launch_config("track_max_coast_frames"),
                "track_drop_frames": launch_config("track_drop_frames"),
                "association_max_center_distance_ratio": launch_config(
                    "association_max_center_distance_ratio"
                ),
                "association_min_iou": launch_config("association_min_iou"),
                "target_switch_min_score_delta": launch_config(
                    "target_switch_min_score_delta"
                ),
                "target_lock_hysteresis_frames": launch_config(
                    "target_lock_hysteresis_frames"
                ),
                "association_max_3d_distance_m": launch_config(
                    "association_max_3d_distance_m"
                ),
            }
        ],
    )


def make_processor_node() -> Node:
    """Create the target processor node."""
    return Node(
        package="yolo_cpp",
        executable="armor_processor_node",
        name="armor_processor_node",
        output="screen",
        parameters=[{"target_frame": launch_config("processor_target_frame")}],
    )


def make_solver_node() -> Node:
    """Create the aim solver node."""
    return Node(
        package="yolo_cpp",
        executable="aim_solver_node",
        name="aim_solver_node",
        output="screen",
        parameters=[
            {
                "hfov_degrees": launch_config("hfov_degrees"),
                "vfov_degrees": launch_config("vfov_degrees"),
                "publish_zero_angles_when_no_target": launch_config(
                    "publish_zero_angles_when_no_target"
                ),
                "default_image_width": launch_config("basler_width"),
                "default_image_height": launch_config("basler_height"),
                "target_topic": "processor/target",
                "use_3d_solution": launch_config("use_3d_solution"),
                "output_mode": launch_config("solver_output_mode"),
                "default_bullet_speed_mps": launch_config("default_bullet_speed_mps"),
                "solver_latency_s": launch_config("solver_latency_s"),
                "compensate_gravity": launch_config("compensate_gravity"),
            }
        ],
    )


def make_stm_bridge_node() -> Node:
    """Create the STM32 serial bridge node."""
    return Node(
        package="yolo_cpp",
        executable="stm32_serial_bridge",
        name="stm32_serial_bridge_node",
        output="screen",
        parameters=[
            {
                "transmit_enable": True,
                "serial_port": launch_config("serial_port"),
                "serial_baudrate": launch_config("serial_baudrate"),
                "serial_verbose": launch_config("serial_verbose"),
                "hold_fire_mode": launch_config("hold_fire_mode"),
                "transmit_angles_in_degrees": launch_config(
                    "transmit_angles_in_degrees"
                ),
            }
        ],
    )


def generate_launch_description() -> LaunchDescription:
    """Create the Basler field graph with detector, solver, and serial bridge."""
    return LaunchDescription(
        [
            *declare_arguments(),
            make_detector_node(),
            make_tracker_node(),
            make_processor_node(),
            make_solver_node(),
            make_stm_bridge_node(),
        ]
    )
