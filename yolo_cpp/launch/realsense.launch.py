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
    rs_use_poll_arg = DeclareLaunchArgument(
        "rs_use_poll",
        default_value="false",
        description="Use non-blocking RealSense poll_for_frames() instead of wait_for_frames()",
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
    show_arg = DeclareLaunchArgument(
        "show", default_value="false", description="OpenCV debug window"
    )
    debug_publish_image_raw_arg = DeclareLaunchArgument(
        "debug_publish_image_raw",
        default_value="false",
        description="Publish detector/image_raw for external viewing instead of an in-process OpenCV window",
    )
    hfov_arg = DeclareLaunchArgument(
        "hfov_degrees", default_value="69.4", description="Horizontal FOV deg"
    )
    vfov_arg = DeclareLaunchArgument(
        "vfov_degrees", default_value="42.5", description="Vertical FOV deg"
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
                "camera_type": "realsense",
                "realsense_width": LaunchConfiguration("rs_width"),
                "realsense_height": LaunchConfiguration("rs_height"),
                "realsense_fps": LaunchConfiguration("rs_fps"),
                "realsense_exposure": LaunchConfiguration("rs_exposure"),
                "realsense_auto_exposure": LaunchConfiguration("rs_auto_exposure"),
                "realsense_use_poll": LaunchConfiguration("rs_use_poll"),
                "model_path": LaunchConfiguration("model_path"),
                "task": LaunchConfiguration("task"),
                "conf": LaunchConfiguration("conf"),
                "show": LaunchConfiguration("show"),
                "debug_publish_image_raw": LaunchConfiguration("debug_publish_image_raw"),
                "hfov_degrees": LaunchConfiguration("hfov_degrees"),
                "vfov_degrees": LaunchConfiguration("vfov_degrees"),
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
            model_path_arg,
            task_arg,
            conf_arg,
            show_arg,
            debug_publish_image_raw_arg,
            hfov_arg,
            vfov_arg,
            track_min_hits_arg,
            track_max_coast_frames_arg,
            track_drop_frames_arg,
            association_max_center_distance_ratio_arg,
            association_min_iou_arg,
            target_switch_min_score_delta_arg,
            target_lock_hysteresis_frames_arg,
            publish_zero_angles_arg,
            detector_node,
            tracker_node,
            solver_node,
        ]
    )
