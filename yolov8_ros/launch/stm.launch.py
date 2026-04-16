from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # --- Camera backend arg ---
    camera_type_arg = DeclareLaunchArgument(
        "camera_type",
        default_value="realsense",
        description="Camera backend: realsense or basler",
    )

    # --- Serial args ---
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

    # --- Basler camera args ---
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

    # --- RealSense camera args ---
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
        default_value="120.0",
        description="RealSense manual exposure (higher than previous default)",
    )
    realsense_auto_exposure_arg = DeclareLaunchArgument(
        "realsense_auto_exposure",
        default_value="true",
        description="Enable RealSense auto exposure",
    )
    realsense_use_max_native_fps_arg = DeclareLaunchArgument(
        "realsense_use_max_native_fps",
        default_value="true",
        description="Use the best native RealSense BGR8 FPS for the requested resolution",
    )

    # --- YOLO args ---
    use_kalman_arg = DeclareLaunchArgument(
        "use_kalman", default_value="true", description="Enable Kalman filter tracking"
    )
    inference_mode_arg = DeclareLaunchArgument(
        "inference_mode",
        default_value="predict",
        description="YOLO modular inference mode: predict or track",
    )
    show_arg = DeclareLaunchArgument(
        "show",
        default_value="false",
        description="Show OpenCV visualization window (off by default for production)",
    )

    # --- Nodes ---
    camera_node = Node(
        package="yolov8_ros",
        executable="camera_node",
        name="camera_node",
        output="screen",
        parameters=[
            {
                "camera_type": LaunchConfiguration("camera_type"),
                "imgsz_width": LaunchConfiguration("basler_width"),
                "imgsz_height": LaunchConfiguration("basler_height"),
                "basler_fps": LaunchConfiguration("basler_fps"),
                "basler_exposure_us": LaunchConfiguration("basler_exposure_us"),
                "basler_gain_db": LaunchConfiguration("basler_gain_db"),
                "basler_use_full_sensor": True,
                "basler_centered_roi": True,
                "realsense_width": LaunchConfiguration("realsense_width"),
                "realsense_height": LaunchConfiguration("realsense_height"),
                "realsense_fps": LaunchConfiguration("realsense_fps"),
                "realsense_exposure": LaunchConfiguration("realsense_exposure"),
                "realsense_auto_exposure": LaunchConfiguration(
                    "realsense_auto_exposure"
                ),
                "realsense_use_max_native_fps": LaunchConfiguration(
                    "realsense_use_max_native_fps"
                ),
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
                "show": LaunchConfiguration("show"),
                "use_kalman": LaunchConfiguration("use_kalman"),
                "inference_mode": LaunchConfiguration("inference_mode"),
            }
        ],
    )

    stm_bridge_node = Node(
        package="yolov8_ros",
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
            serial_port_arg,
            serial_baudrate_arg,
            serial_verbose_arg,
            basler_width_arg,
            basler_height_arg,
            basler_fps_arg,
            basler_exposure_arg,
            basler_gain_arg,
            realsense_width_arg,
            realsense_height_arg,
            realsense_fps_arg,
            realsense_exposure_arg,
            realsense_auto_exposure_arg,
            realsense_use_max_native_fps_arg,
            use_kalman_arg,
            inference_mode_arg,
            show_arg,
            camera_node,
            yolov8_node,
            stm_bridge_node,
        ]
    )
