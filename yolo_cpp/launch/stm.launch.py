"""Launch the full camera-to-STM32 auto-aim stack."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Create the complete detector, tracker, processor, solver, and serial bridge graph."""
    default_model_path = (
        "/home/ultraviolet/CV_Detection/model_test/best-roboflow-nitish-obb.engine"
    )
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
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=default_model_path,
        description="Path to YOLO weights (.pt / .engine / .onnx)",
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
    basler_uncapped_fps_arg = DeclareLaunchArgument(
        "basler_uncapped_fps",
        default_value="true",
        description="Disable Basler frame-rate limiting and let the camera run as fast as configured exposure allows",
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
    basler_use_full_sensor_arg = DeclareLaunchArgument(
        "basler_use_full_sensor",
        default_value="true",
        description="Use the full Basler sensor instead of cropping the camera to the model input size",
    )
    basler_centered_roi_arg = DeclareLaunchArgument(
        "basler_centered_roi",
        default_value="true",
        description="Center the Basler ROI when basler_use_full_sensor is false",
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
        default_value="100.0",
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
    debug_publish_image_raw_arg = DeclareLaunchArgument(
        "debug_publish_image_raw",
        default_value="false",
        description="Publish debug image_raw frame from detector",
    )
    use_model_color_classes_arg = DeclareLaunchArgument(
        "use_model_color_classes",
        default_value="true",
        description="Use model class IDs directly for enemy color instead of HSV verification",
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
    association_max_3d_distance_arg = DeclareLaunchArgument(
        "association_max_3d_distance_m",
        default_value="0.35",
        description="Maximum 3D association distance when PnP pose is available",
    )
    hfov_arg = DeclareLaunchArgument(
        "hfov_degrees", default_value="79.2", description="Horizontal FOV degrees"
    )
    vfov_arg = DeclareLaunchArgument(
        "vfov_degrees", default_value="63.78", description="Vertical FOV degrees"
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
    solver_output_mode_arg = DeclareLaunchArgument(
        "solver_output_mode",
        default_value="absolute",
        description="Aim output mode: offset or absolute. Absolute adds STM attitude yaw/pitch.",
    )
    default_bullet_speed_arg = DeclareLaunchArgument(
        "default_bullet_speed_mps",
        default_value="18.0",
        description="Fallback bullet speed when STM feedback does not provide one",
    )
    solver_latency_arg = DeclareLaunchArgument(
        "solver_latency_s",
        default_value="0.03",
        description="Latency used by the 3D ballistic prediction",
    )
    compensate_gravity_arg = DeclareLaunchArgument(
        "compensate_gravity",
        default_value="true",
        description="Enable simple gravity drop compensation in the 3D solver",
    )
    publish_zero_angles_arg = DeclareLaunchArgument(
        "publish_zero_angles_when_no_target",
        default_value="true",
        description="Publish zero yaw/pitch when there is no selected target",
    )
    transmit_angles_in_degrees_arg = DeclareLaunchArgument(
        "transmit_angles_in_degrees",
        default_value="true",
        description="Convert solver radians to degrees before STM TX",
    )
    hold_fire_mode_arg = DeclareLaunchArgument(
        "hold_fire_mode",
        default_value="0",
        description="Fire mode sent while the solver suggests holding fire",
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
                "model_path": LaunchConfiguration("model_path"),
                "deepstream_config_path": LaunchConfiguration("deepstream_config_path"),
                "imgsz_width": LaunchConfiguration("basler_width"),
                "imgsz_height": LaunchConfiguration("basler_height"),
                "basler_fps": LaunchConfiguration("basler_fps"),
                "basler_uncapped_fps": LaunchConfiguration("basler_uncapped_fps"),
                "basler_exposure_us": LaunchConfiguration("basler_exposure_us"),
                "basler_gain_db": LaunchConfiguration("basler_gain_db"),
                "basler_flip": LaunchConfiguration("basler_flip"),
                "basler_use_full_sensor": LaunchConfiguration("basler_use_full_sensor"),
                "basler_centered_roi": LaunchConfiguration("basler_centered_roi"),
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
                "hfov_degrees": LaunchConfiguration("hfov_degrees"),
                "vfov_degrees": LaunchConfiguration("vfov_degrees"),
                "enable_3d_pose": LaunchConfiguration("enable_3d_pose"),
                "camera_frame_id": LaunchConfiguration("camera_frame_id"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "camera_fx": LaunchConfiguration("camera_fx"),
                "camera_fy": LaunchConfiguration("camera_fy"),
                "camera_cx": LaunchConfiguration("camera_cx"),
                "camera_cy": LaunchConfiguration("camera_cy"),
                "show": LaunchConfiguration("show"),
                "debug_publish_image_raw": LaunchConfiguration(
                    "debug_publish_image_raw"
                ),
                "use_model_color_classes": LaunchConfiguration(
                    "use_model_color_classes"
                ),
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
                "default_image_width": LaunchConfiguration("realsense_width"),
                "default_image_height": LaunchConfiguration("realsense_height"),
                "target_topic": "processor/target",
                "use_3d_solution": LaunchConfiguration("use_3d_solution"),
                "output_mode": LaunchConfiguration("solver_output_mode"),
                "default_bullet_speed_mps": LaunchConfiguration(
                    "default_bullet_speed_mps"
                ),
                "solver_latency_s": LaunchConfiguration("solver_latency_s"),
                "compensate_gravity": LaunchConfiguration("compensate_gravity"),
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
                "hold_fire_mode": LaunchConfiguration("hold_fire_mode"),
                "transmit_angles_in_degrees": LaunchConfiguration(
                    "transmit_angles_in_degrees"
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            camera_type_arg,
            detector_backend_arg,
            allow_native_fallback_arg,
            deepstream_config_arg,
            model_path_arg,
            serial_port_arg,
            serial_baudrate_arg,
            serial_verbose_arg,
            basler_width_arg,
            basler_height_arg,
            basler_fps_arg,
            basler_uncapped_fps_arg,
            basler_exposure_arg,
            basler_gain_arg,
            basler_flip_arg,
            basler_use_full_sensor_arg,
            basler_centered_roi_arg,
            realsense_width_arg,
            realsense_height_arg,
            realsense_fps_arg,
            realsense_exposure_arg,
            realsense_auto_exposure_arg,
            realsense_use_poll_arg,
            realsense_use_max_native_fps_arg,
            inference_mode_arg,
            debug_publish_image_raw_arg,
            use_model_color_classes_arg,
            show_arg,
            track_min_hits_arg,
            track_max_coast_frames_arg,
            track_drop_frames_arg,
            association_max_center_distance_ratio_arg,
            association_min_iou_arg,
            target_switch_min_score_delta_arg,
            target_lock_hysteresis_frames_arg,
            association_max_3d_distance_arg,
            hfov_arg,
            vfov_arg,
            enable_3d_pose_arg,
            camera_frame_id_arg,
            camera_info_topic_arg,
            camera_fx_arg,
            camera_fy_arg,
            camera_cx_arg,
            camera_cy_arg,
            processor_target_frame_arg,
            use_3d_solution_arg,
            solver_output_mode_arg,
            default_bullet_speed_arg,
            solver_latency_arg,
            compensate_gravity_arg,
            publish_zero_angles_arg,
            transmit_angles_in_degrees_arg,
            hold_fire_mode_arg,
            detector_node,
            tracker_node,
            processor_node,
            solver_node,
            stm_bridge_node,
        ]
    )
