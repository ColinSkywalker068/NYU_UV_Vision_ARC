"""Launch a RealSense-only detector, tracker, processor, and solver stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Create the lightweight RealSense auto-aim launch graph."""
    default_model_path = PathJoinSubstitution(
        [
            FindPackageShare("yolo_cpp"),
            "models",
            "best-roboflow-nitish-obb.onnx_b1_gpu0_fp16.engine",
        ]
    )

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
    rs_use_poll_arg = DeclareLaunchArgument(
        "rs_use_poll",
        default_value="false",
        description="Use non-blocking RealSense poll_for_frames() instead of wait_for_frames()",
    )
    rs_use_max_native_fps_arg = DeclareLaunchArgument(
        "rs_use_max_native_fps",
        default_value="true",
        description="Use the best native RealSense FPS for the requested resolution",
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
        "conf", default_value="0.30", description="Confidence threshold"
    )
    use_model_color_classes_arg = DeclareLaunchArgument(
        "use_model_color_classes",
        default_value="true",
        description="Use model class IDs directly for enemy color instead of HSV verification",
    )
    enable_color_verification_arg = DeclareLaunchArgument(
        "enable_color_verification",
        default_value="false",
        description="Enable HSV enemy-color verification when use_model_color_classes is false",
    )
    target_color_arg = DeclareLaunchArgument(
        "target_color",
        default_value="blue",
        description="Enemy armor color used until robot_id feedback overrides it: blue or red",
    )
    show_arg = DeclareLaunchArgument(
        "show", default_value="false", description="OpenCV debug window"
    )
    debug_publish_image_raw_arg = DeclareLaunchArgument(
        "debug_publish_image_raw",
        default_value="false",
        description="Publish debug image_raw frame from detector",
    )
    save_target_frames_arg = DeclareLaunchArgument(
        "save_target_frames",
        default_value="false",
        description="Save up to target_frame_save_max JPEG frames when a target is found",
    )
    target_frame_save_dir_arg = DeclareLaunchArgument(
        "target_frame_save_dir",
        default_value="",
        description="Optional directory for saved target frames; default is pictures next to vision_logs",
    )
    target_frame_save_max_arg = DeclareLaunchArgument(
        "target_frame_save_max",
        default_value="100",
        description="Maximum target-frame images saved per run",
    )
    target_frame_jpeg_quality_arg = DeclareLaunchArgument(
        "target_frame_jpeg_quality",
        default_value="85",
        description="JPEG quality for saved target frames",
    )
    hfov_arg = DeclareLaunchArgument(
        "hfov_degrees", default_value="69.4", description="Horizontal FOV deg"
    )
    vfov_arg = DeclareLaunchArgument(
        "vfov_degrees", default_value="42.5", description="Vertical FOV deg"
    )
    track_min_hits_arg = DeclareLaunchArgument(
        "track_min_hits",
        default_value="1",
        description="Frames required to confirm a track",
    )
    track_max_coast_frames_arg = DeclareLaunchArgument(
        "track_max_coast_frames",
        default_value="1",
        description="Maximum published coast frames",
    )
    track_drop_frames_arg = DeclareLaunchArgument(
        "track_drop_frames",
        default_value="4",
        description="Frames before dropping an unmatched track",
    )
    association_max_center_distance_ratio_arg = DeclareLaunchArgument(
        "association_max_center_distance_ratio",
        default_value="0.30",
        description="Maximum normalized center distance for matching",
    )
    association_min_iou_arg = DeclareLaunchArgument(
        "association_min_iou",
        default_value="0.0",
        description="Minimum IoU required for matching",
    )
    target_switch_min_score_delta_arg = DeclareLaunchArgument(
        "target_switch_min_score_delta",
        default_value="0.02",
        description="Confidence delta required to steal target lock",
    )
    target_lock_hysteresis_frames_arg = DeclareLaunchArgument(
        "target_lock_hysteresis_frames",
        default_value="0",
        description="Coast frames allowed before target lock yields",
    )
    association_max_3d_distance_arg = DeclareLaunchArgument(
        "association_max_3d_distance_m",
        default_value="0.35",
        description="Maximum 3D association distance when PnP pose is available",
    )
    publish_zero_angles_arg = DeclareLaunchArgument(
        "publish_zero_angles_when_no_target",
        default_value="true",
        description="Publish zero yaw/pitch when there is no selected target",
    )
    enable_3d_pose_arg = DeclareLaunchArgument(
        "enable_3d_pose",
        default_value="true",
        description="Enable armor corner PnP pose solving in the detector",
    )
    camera_frame_id_arg = DeclareLaunchArgument(
        "camera_frame_id",
        default_value="camera_optical_frame",
        description="Frame id attached to detector PnP poses",
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="camera_info",
        description="CameraInfo topic used for calibrated PnP intrinsics",
    )
    camera_fx_arg = DeclareLaunchArgument(
        "camera_fx",
        default_value="0.0",
        description="Optional static camera fx; FOV fallback is used when unset",
    )
    camera_fy_arg = DeclareLaunchArgument(
        "camera_fy",
        default_value="0.0",
        description="Optional static camera fy; FOV fallback is used when unset",
    )
    camera_cx_arg = DeclareLaunchArgument(
        "camera_cx",
        default_value="0.0",
        description="Optional static camera cx",
    )
    camera_cy_arg = DeclareLaunchArgument(
        "camera_cy",
        default_value="0.0",
        description="Optional static camera cy",
    )
    processor_target_frame_arg = DeclareLaunchArgument(
        "processor_target_frame",
        default_value="camera_optical_frame",
        description="Frame used for processor target pose output",
    )
    use_3d_solution_arg = DeclareLaunchArgument(
        "use_3d_solution",
        default_value="true",
        description="Use processor pose/velocity when available for yaw/pitch solving",
    )
    solver_latency_arg = DeclareLaunchArgument(
        "solver_latency_s",
        default_value="0.03",
        description="Latency used by the 3D ballistic prediction",
    )
    default_bullet_speed_arg = DeclareLaunchArgument(
        "default_bullet_speed_mps",
        default_value="18.0",
        description="Fallback bullet speed for 3D solving",
    )

    detector_node = Node(
        package="yolo_cpp",
        executable="armor_detector_node",
        name="armor_detector_node",
        output="screen",
        parameters=[
            {
                "camera_type": "realsense",
                "detector_backend": "native_tensorrt",
                "allow_native_tensorrt_fallback": False,
                "realsense_width": LaunchConfiguration("rs_width"),
                "realsense_height": LaunchConfiguration("rs_height"),
                "realsense_fps": LaunchConfiguration("rs_fps"),
                "realsense_exposure": LaunchConfiguration("rs_exposure"),
                "realsense_auto_exposure": LaunchConfiguration("rs_auto_exposure"),
                "realsense_use_poll": LaunchConfiguration("rs_use_poll"),
                "realsense_use_max_native_fps": LaunchConfiguration(
                    "rs_use_max_native_fps"
                ),
                "model_path": LaunchConfiguration("model_path"),
                "task": LaunchConfiguration("task"),
                "conf": LaunchConfiguration("conf"),
                "use_model_color_classes": LaunchConfiguration(
                    "use_model_color_classes"
                ),
                "enable_color_verification": LaunchConfiguration(
                    "enable_color_verification"
                ),
                "target_color": LaunchConfiguration("target_color"),
                "show": LaunchConfiguration("show"),
                "debug_publish_image_raw": LaunchConfiguration(
                    "debug_publish_image_raw"
                ),
                "save_target_frames": LaunchConfiguration("save_target_frames"),
                "target_frame_save_dir": LaunchConfiguration("target_frame_save_dir"),
                "target_frame_save_max": LaunchConfiguration("target_frame_save_max"),
                "target_frame_jpeg_quality": LaunchConfiguration(
                    "target_frame_jpeg_quality"
                ),
                "hfov_degrees": LaunchConfiguration("hfov_degrees"),
                "vfov_degrees": LaunchConfiguration("vfov_degrees"),
                "enable_3d_pose": LaunchConfiguration("enable_3d_pose"),
                "camera_frame_id": LaunchConfiguration("camera_frame_id"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "camera_fx": LaunchConfiguration("camera_fx"),
                "camera_fy": LaunchConfiguration("camera_fy"),
                "camera_cx": LaunchConfiguration("camera_cx"),
                "camera_cy": LaunchConfiguration("camera_cy"),
                "imgsz_width": LaunchConfiguration("rs_width"),
                "imgsz_height": LaunchConfiguration("rs_height"),
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
                "association_max_3d_distance_m": LaunchConfiguration(
                    "association_max_3d_distance_m"
                ),
            }
        ],
    )

    processor_node = Node(
        package="yolo_cpp",
        executable="armor_processor_node",
        name="armor_processor_node",
        output="screen",
        parameters=[
            {
                "target_frame": LaunchConfiguration("processor_target_frame"),
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
                "default_image_width": LaunchConfiguration("rs_width"),
                "default_image_height": LaunchConfiguration("rs_height"),
                "target_topic": "processor/target",
                "use_3d_solution": LaunchConfiguration("use_3d_solution"),
                "output_mode": "offset",
                "solver_latency_s": LaunchConfiguration("solver_latency_s"),
                "default_bullet_speed_mps": LaunchConfiguration(
                    "default_bullet_speed_mps"
                ),
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
            rs_use_poll_arg,
            rs_use_max_native_fps_arg,
            model_path_arg,
            task_arg,
            conf_arg,
            use_model_color_classes_arg,
            enable_color_verification_arg,
            target_color_arg,
            show_arg,
            debug_publish_image_raw_arg,
            save_target_frames_arg,
            target_frame_save_dir_arg,
            target_frame_save_max_arg,
            target_frame_jpeg_quality_arg,
            hfov_arg,
            vfov_arg,
            track_min_hits_arg,
            track_max_coast_frames_arg,
            track_drop_frames_arg,
            association_max_center_distance_ratio_arg,
            association_min_iou_arg,
            target_switch_min_score_delta_arg,
            target_lock_hysteresis_frames_arg,
            association_max_3d_distance_arg,
            publish_zero_angles_arg,
            enable_3d_pose_arg,
            camera_frame_id_arg,
            camera_info_topic_arg,
            camera_fx_arg,
            camera_fy_arg,
            camera_cx_arg,
            camera_cy_arg,
            processor_target_frame_arg,
            use_3d_solution_arg,
            solver_latency_arg,
            default_bullet_speed_arg,
            detector_node,
            tracker_node,
            processor_node,
            solver_node,
        ]
    )
