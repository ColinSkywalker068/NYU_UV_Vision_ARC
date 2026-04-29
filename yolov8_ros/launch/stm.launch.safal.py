from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # --- Serial args ---
    serial_port_arg = DeclareLaunchArgument(
        "serial_port",
        default_value="/dev/ttyACM0",
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

    # --- YOLO args ---
    use_kalman_arg = DeclareLaunchArgument(
        "use_kalman", default_value="false", description="Enable Kalman filter tracking"
    )
    tracker_type_arg = DeclareLaunchArgument(
        "tracker_type", default_value="lkf", description="Tracker type: lkf or ukf"
    )
    show_arg = DeclareLaunchArgument(
        "show", default_value="false", description="Show OpenCV visualization window"
    )
    benchmark_arg = DeclareLaunchArgument(
        "benchmark", default_value="false", description="Enable benchmark CSV logging"
    )

    # --- Nodes ---
    camera_node = Node(
        package="yolov8_ros",
        executable="camera_node",
        name="camera_node",
        output="screen",
        parameters=[
            {
                "camera_type": "basler",
                "imgsz_width": LaunchConfiguration("basler_width"),
                "imgsz_height": LaunchConfiguration("basler_height"),
                "basler_fps": LaunchConfiguration("basler_fps"),
                "basler_exposure_us": LaunchConfiguration("basler_exposure_us"),
                "basler_gain_db": LaunchConfiguration("basler_gain_db"),
                "basler_use_full_sensor": True,
                "basler_centered_roi": True,
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
                "tracker_type": LaunchConfiguration("tracker_type"),
                "benchmark": LaunchConfiguration("benchmark"),
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
            serial_port_arg,
            serial_baudrate_arg,
            serial_verbose_arg,
            basler_width_arg,
            basler_height_arg,
            basler_fps_arg,
            basler_exposure_arg,
            basler_gain_arg,
            use_kalman_arg,
            tracker_type_arg,
            show_arg,
            benchmark_arg,
            camera_node,
            yolov8_node,
            stm_bridge_node,
        ]
    )
