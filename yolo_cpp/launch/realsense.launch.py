"""Launch a RealSense-only detector, tracker, processor, and solver stack."""

from typing import Any

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


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
        ("rs_width", "640", "RealSense color width"),
        ("rs_height", "480", "RealSense color height"),
        ("rs_fps", "60", "RealSense color FPS"),
        ("rs_exposure", "120.0", "RealSense manual exposure"),
        ("rs_gain", "64.0", "RealSense manual gain"),
        ("rs_auto_exposure", "true", "Enable RealSense auto exposure"),
        (
            "rs_auto_white_balance",
            "true",
            "Enable RealSense auto white balance",
        ),
        ("rs_white_balance", "6500.0", "RealSense manual white balance"),
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
        ("imgsz_width", "640", "Detector model input width"),
        ("imgsz_height", "640", "Detector model input height"),
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
            "model_path",
            EnvironmentVariable("CV_STM_MODEL_PATH", default_value=""),
            "Path to YOLO weights (.pt / .engine / .onnx); empty uses the package default resolver",
        ),
        ("task", "obb", "YOLO task: detect, obb, segment"),
        ("inference_mode", "predict", "Detector inference mode"),
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
        (
            "use_robot_id_targeting",
            "false",
            "Allow robot_id or attitude_flags feedback to override target color",
        ),
        ("enemy_color_red_value", "0", "Controller/robot-id value for red enemy color"),
        ("enemy_color_blue_value", "1", "Controller/robot-id value for blue enemy color"),
        ("max_hsv_candidates", "3", "Maximum candidate boxes checked by HSV color verification"),
        ("small_armor_width_m", "0.135", "Small armor physical width for PnP"),
        ("small_armor_height_m", "0.055", "Small armor physical height for PnP"),
        ("large_armor_width_m", "0.225", "Large armor physical width for PnP"),
        ("large_armor_height_m", "0.055", "Large armor physical height for PnP"),
        (
            "large_armor_min_aspect_ratio",
            "3.2",
            "Aspect ratio threshold used to classify large armor plates",
        ),
        ("runtime_log_dir", "", "Optional directory for detector runtime logs"),
        ("stage_log_interval", "300", "Frames between detector/tracker timing summaries"),
        ("show", "false", "Preview mode: false, gui, or web"),
        ("vis_window_name", "Armor Detector", "OpenCV preview window name"),
        ("preview_target_topic", "tracker/target", "Target topic used for preview overlays"),
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
        (
            "publish_zero_angles_when_no_target",
            "true",
            "Publish zero yaw/pitch when there is no selected target",
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
            "camera_optical_frame",
            "Frame used for processor target pose output",
        ),
        ("processor_lost_threshold", "5", "Frames before processor target state is considered lost"),
        ("spin_observer_allow", "true", "Enable spin observation in the processor"),
        ("spin_observer_max_jump_angle", "0.2", "Maximum armor jump angle for spin observation"),
        ("spin_observer_max_jump_period", "0.8", "Maximum jump period for spin observation"),
        (
            "spin_observer_allow_following_range",
            "0.3",
            "Range threshold for allowing following during spin observation",
        ),
        ("filter_position_alpha", "0.7", "Processor position low-pass alpha"),
        ("filter_velocity_alpha", "0.5", "Processor velocity low-pass alpha"),
        (
            "use_3d_solution",
            "true",
            "Use processor pose/velocity when available for yaw/pitch solving",
        ),
        (
            "solver_output_mode",
            "offset",
            "Aim output mode: offset or absolute. Absolute adds STM attitude yaw/pitch.",
        ),
        ("solver_latency_s", "0.03", "Latency used by the 3D ballistic prediction"),
        ("default_bullet_speed_mps", "18.0", "Fallback bullet speed for 3D solving"),
        (
            "compensate_gravity",
            "true",
            "Enable simple gravity drop compensation in the 3D solver",
        ),
        ("gravity_mps2", "9.80665", "Gravity constant used by solver compensation"),
    ]


def declare_arguments() -> list[DeclareLaunchArgument]:
    """Build all launch argument actions from the table above."""
    return [
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in argument_specs()
    ]


def make_detector_node() -> Node:
    """Create the detector node for RealSense input."""
    return Node(
        package="yolo_cpp",
        executable="armor_detector_node",
        name="armor_detector_node",
        output="screen",
        parameters=[
            {
                "camera_type": "realsense",
                "detector_backend": launch_config("detector_backend"),
                "allow_native_tensorrt_fallback": launch_config(
                    "allow_native_tensorrt_fallback"
                ),
                "deepstream_config_path": launch_config("deepstream_config_path"),
                "realsense_width": launch_config("rs_width"),
                "realsense_height": launch_config("rs_height"),
                "realsense_fps": launch_config("rs_fps"),
                "realsense_exposure": launch_config("rs_exposure"),
                "realsense_gain": launch_config("rs_gain"),
                "realsense_auto_exposure": launch_config("rs_auto_exposure"),
                "realsense_auto_white_balance": launch_config("rs_auto_white_balance"),
                "realsense_white_balance": launch_config("rs_white_balance"),
                "realsense_use_poll": launch_config("rs_use_poll"),
                "realsense_use_max_native_fps": launch_config("rs_use_max_native_fps"),
                "deepstream_profile_stages": launch_config("deepstream_profile_stages"),
                "deepstream_profile_stage_interval": launch_config(
                    "deepstream_profile_stage_interval"
                ),
                "model_path": launch_config("model_path"),
                "task": launch_config("task"),
                "inference_mode": launch_config("inference_mode"),
                "conf": launch_config("conf"),
                "use_model_color_classes": launch_config("use_model_color_classes"),
                "enable_color_verification": launch_config("enable_color_verification"),
                "target_color": launch_config("target_color"),
                "use_robot_id_targeting": launch_config("use_robot_id_targeting"),
                "enemy_color_red_value": launch_config("enemy_color_red_value"),
                "enemy_color_blue_value": launch_config("enemy_color_blue_value"),
                "max_hsv_candidates": launch_config("max_hsv_candidates"),
                "small_armor_width_m": launch_config("small_armor_width_m"),
                "small_armor_height_m": launch_config("small_armor_height_m"),
                "large_armor_width_m": launch_config("large_armor_width_m"),
                "large_armor_height_m": launch_config("large_armor_height_m"),
                "large_armor_min_aspect_ratio": launch_config(
                    "large_armor_min_aspect_ratio"
                ),
                "runtime_log_dir": launch_config("runtime_log_dir"),
                "stage_log_interval": launch_config("stage_log_interval"),
                "show": ParameterValue(launch_config("show"), value_type=str),
                "vis_window_name": launch_config("vis_window_name"),
                "preview_target_topic": launch_config("preview_target_topic"),
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
                "imgsz_width": launch_config("imgsz_width"),
                "imgsz_height": launch_config("imgsz_height"),
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
                "use_robot_id_targeting": launch_config("use_robot_id_targeting"),
                "enemy_color_red_value": launch_config("enemy_color_red_value"),
                "enemy_color_blue_value": launch_config("enemy_color_blue_value"),
                "stage_log_interval": launch_config("stage_log_interval"),
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
                "processor_lost_threshold": launch_config("processor_lost_threshold"),
                "spin_observer.allow": launch_config("spin_observer_allow"),
                "spin_observer.max_jump_angle": launch_config(
                    "spin_observer_max_jump_angle"
                ),
                "spin_observer.max_jump_period": launch_config(
                    "spin_observer_max_jump_period"
                ),
                "spin_observer.allow_following_range": launch_config(
                    "spin_observer_allow_following_range"
                ),
                "filter.position_alpha": launch_config("filter_position_alpha"),
                "filter.velocity_alpha": launch_config("filter_velocity_alpha"),
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
                "default_image_width": launch_config("rs_width"),
                "default_image_height": launch_config("rs_height"),
                "target_topic": "processor/target",
                "use_3d_solution": launch_config("use_3d_solution"),
                "output_mode": launch_config("solver_output_mode"),
                "solver_latency_s": launch_config("solver_latency_s"),
                "default_bullet_speed_mps": launch_config("default_bullet_speed_mps"),
                "compensate_gravity": launch_config("compensate_gravity"),
                "gravity_mps2": launch_config("gravity_mps2"),
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
