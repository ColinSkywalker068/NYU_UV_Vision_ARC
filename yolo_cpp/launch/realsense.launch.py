"""Launch a RealSense-only detector, tracker, processor, and solver stack."""

from typing import Any

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_config(name: str) -> LaunchConfiguration:
    """Keep launch parameter dictionaries compact and readable."""
    return LaunchConfiguration(name)


def argument_specs() -> list[tuple[str, Any, str]]:
    """Declare launch arguments in operator-facing order."""
    return [
        ("rs_width", "640", "RealSense color width"),
        ("rs_height", "480", "RealSense color height"),
        ("rs_fps", "60", "RealSense color FPS"),
        ("rs_exposure", "120.0", "RealSense manual exposure"),
        ("rs_auto_exposure", "true", "Enable RealSense auto exposure"),
        (
            "rs_use_poll",
            "false",
            "Use non-blocking RealSense poll_for_frames() instead of wait_for_frames()",
        ),
        (
            "rs_use_max_native_fps",
            "true",
            "Use the best native RealSense FPS for the requested resolution",
        ),
        (
            "model_path",
            EnvironmentVariable("CV_STM_MODEL_PATH", default_value=""),
            "Path to YOLO weights (.pt / .engine / .onnx); empty uses the package default resolver",
        ),
        ("task", "obb", "YOLO task: detect, obb, segment"),
        ("conf", "0.30", "Confidence threshold"),
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
            "Enemy armor color used until robot_id feedback overrides it: blue or red",
        ),
        ("show", "false", "Preview mode: false, gui, or web"),
        (
            "preview_max_fps",
            "60.0",
            "Maximum preview render FPS when show is gui or web",
        ),
        (
            "preview_http_enabled",
            "true",
            "Allow the browser preview server when show is web",
        ),
        (
            "preview_http_bind_address",
            "0.0.0.0",
            "Preview HTTP bind address; 0.0.0.0 allows other devices on the robot LAN",
        ),
        ("preview_http_port", "8080", "Preview HTTP port"),
        ("preview_http_jpeg_quality", "80", "Preview HTTP JPEG quality 1-100"),
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
        ("hfov_degrees", "69.4", "Horizontal FOV deg"),
        ("vfov_degrees", "42.5", "Vertical FOV deg"),
        ("track_min_hits", "2", "Frames required to confirm a track"),
        ("track_max_coast_frames", "6", "Maximum published coast frames"),
        ("track_drop_frames", "16", "Frames before dropping an unmatched track"),
        ("track_max_coast_s", "0.12", "Maximum seconds to publish a missed plate prediction"),
        ("track_drop_s", "0.35", "Maximum seconds to retain an unmatched identity"),
        ("position_filter_tau_s", "0.035", "3D position filter time constant; zero disables"),
        ("velocity_filter_tau_s", "0.10", "3D velocity filter time constant; zero disables"),
        (
            "association_max_center_distance_ratio",
            "0.30",
            "Maximum normalized center distance for matching",
        ),
        ("association_min_iou", "0.0", "Minimum IoU required for matching"),
        (
            "target_switch_min_score_delta",
            "0.08",
            "Confidence delta required to steal target lock",
        ),
        (
            "target_lock_hysteresis_frames",
            "6",
            "Coast frames allowed before target lock yields",
        ),
        (
            "association_max_3d_distance_m",
            "0.35",
            "Maximum 3D association distance when PnP pose is available",
        ),
        (
            "publish_zero_angles_when_no_target",
            "true",
            "Publish zero yaw/pitch when there is no selected target",
        ),
        (
            "enable_fire_recommendation",
            "false",
            "Allow solver fire recommendations; human opt-in for competition only",
        ),
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
            "gimbal_inertial",
            "Frame used for processor target pose output",
        ),
        (
            "use_3d_solution",
            "true",
            "Use processor pose/velocity when available for yaw/pitch solving",
        ),
        (
            "robot_estimator_enable",
            "true",
            "Track a shared robot center/phase; a lone plate still uses direct aiming",
        ),
        ("robot_armor_radius_m", "0.26", "Robot center-to-armor radius in meters"),
        ("robot_armor_count", "4", "Number of armor positions around the robot"),
        ("robot_center_gate_m", "0.45", "Reject plates this far from the tracked center"),
        (
            "robot_plate_cluster_gate_m",
            "0.18",
            "Maximum inferred-center separation for plates on the same robot",
        ),
        ("robot_estimator_min_updates", "5", "Updates required before robot prediction"),
        ("robot_estimator_use_ekf", "true", "Use uncertainty-aware center/phase/geometry estimation"),
        ("actuator_delay_s", "0.0", "Measured actuator response delay; do not guess from packet rate"),
        ("attitude_max_gap_ms", "50.0", "Maximum telemetry interpolation gap"),
        ("attitude_sync_tolerance_ms", "30.0", "Maximum telemetry edge extrapolation"),
        ("camera_translation_m", "[0.0,0.0,0.0]", "Camera offset from gimbal pivot, metres in zero-attitude optical axes"),
        ("camera_mount_yaw_pitch_roll", "[0.0,0.0,0.0]", "Camera mounting rotation in radians"),
        ("attitude_receive_delay_s", "0.0", "Measured receive-to-sample delay of SP telemetry"),
        ("robot_estimator_coast_frames", "8", "Frames to predict through missing detections"),
        ("solver_latency_s", "0.0", "Residual transport delay added to measured frame age"),
        ("default_bullet_speed_mps", "18.0", "Fallback bullet speed for 3D solving"),
    ]


def declare_arguments() -> list[DeclareLaunchArgument]:
    """Build all launch argument actions from the table above."""
    return [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in argument_specs()
    ]


def make_detector_node() -> Node:
    """Create the native TensorRT detector node for RealSense input."""
    return Node(
        package="yolo_cpp",
        executable="armor_detector_node",
        name="armor_detector_node",
        output="screen",
        parameters=[
            {
                "camera_type": "realsense",
                "detector_backend": "native_tensorrt",
                "allow_native_tensorrt_fallback": False,
                "realsense_width": launch_config("rs_width"),
                "realsense_height": launch_config("rs_height"),
                "realsense_fps": launch_config("rs_fps"),
                "realsense_exposure": launch_config("rs_exposure"),
                "realsense_auto_exposure": launch_config("rs_auto_exposure"),
                "realsense_use_poll": launch_config("rs_use_poll"),
                "realsense_use_max_native_fps": launch_config("rs_use_max_native_fps"),
                "model_path": launch_config("model_path"),
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
                "preview_http_enabled": launch_config("preview_http_enabled"),
                "preview_http_bind_address": launch_config("preview_http_bind_address"),
                "preview_http_port": launch_config("preview_http_port"),
                "preview_http_jpeg_quality": launch_config("preview_http_jpeg_quality"),
                "debug_publish_image_raw": launch_config("debug_publish_image_raw"),
                "save_target_frames": launch_config("save_target_frames"),
                "target_frame_save_dir": launch_config("target_frame_save_dir"),
                "target_frame_save_max": launch_config("target_frame_save_max"),
                "target_frame_jpeg_quality": launch_config("target_frame_jpeg_quality"),
                "hfov_degrees": launch_config("hfov_degrees"),
                "vfov_degrees": launch_config("vfov_degrees"),
                "enable_3d_pose": launch_config("enable_3d_pose"),
                "camera_frame_id": launch_config("camera_frame_id"),
                "camera_info_topic": launch_config("camera_info_topic"),
                "camera_fx": launch_config("camera_fx"),
                "camera_fy": launch_config("camera_fy"),
                "camera_cx": launch_config("camera_cx"),
                "camera_cy": launch_config("camera_cy"),
                "imgsz_width": launch_config("rs_width"),
                "imgsz_height": launch_config("rs_height"),
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
                "attitude_sync_tolerance_ms": launch_config("attitude_sync_tolerance_ms"),
                "attitude_max_gap_ms": launch_config("attitude_max_gap_ms"),
                "camera_translation_m": launch_config("camera_translation_m"),
                "camera_mount_yaw_pitch_roll": launch_config("camera_mount_yaw_pitch_roll"),
                "track_max_coast_frames": launch_config("track_max_coast_frames"),
                "track_drop_frames": launch_config("track_drop_frames"),
                "track_max_coast_s": launch_config("track_max_coast_s"),
                "track_drop_s": launch_config("track_drop_s"),
                "position_filter_tau_s": launch_config("position_filter_tau_s"),
                "velocity_filter_tau_s": launch_config("velocity_filter_tau_s"),
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
        parameters=[
            {
                "target_frame": launch_config("processor_target_frame"),
                "robot_estimator.enable": launch_config("robot_estimator_enable"),
                "robot_estimator.use_ekf": launch_config("robot_estimator_use_ekf"),
                "robot_estimator.armor_radius_m": launch_config("robot_armor_radius_m"),
                "robot_estimator.armor_count": launch_config("robot_armor_count"),
                "robot_estimator.center_gate_m": launch_config("robot_center_gate_m"),
                "robot_estimator.plate_cluster_gate_m": launch_config(
                    "robot_plate_cluster_gate_m"
                ),
                "robot_estimator.min_updates": launch_config("robot_estimator_min_updates"),
                "robot_estimator.coast_frames": launch_config("robot_estimator_coast_frames"),
            }
        ],
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
                "enable_fire_recommendation": launch_config(
                    "enable_fire_recommendation"
                ),
                "default_image_width": launch_config("rs_width"),
                "default_image_height": launch_config("rs_height"),
                "target_topic": "processor/target",
                "use_3d_solution": launch_config("use_3d_solution"),
                "output_mode": "offset",
                "solver_latency_s": launch_config("solver_latency_s"),
                "actuator_delay_s": launch_config("actuator_delay_s"),
                "attitude_max_gap_ms": launch_config("attitude_max_gap_ms"),
                "default_bullet_speed_mps": launch_config("default_bullet_speed_mps"),
            }
        ],
    )


def generate_launch_description() -> LaunchDescription:
    """Create the lightweight RealSense auto-aim launch graph."""
    return LaunchDescription(
        [
            *declare_arguments(),
            make_detector_node(),
            make_tracker_node(),
            make_processor_node(),
            make_solver_node(),
        ]
    )
