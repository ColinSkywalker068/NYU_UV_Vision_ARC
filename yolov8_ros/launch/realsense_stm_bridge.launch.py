from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
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

    camera_node = Node(
        package="yolov8_ros",
        executable="camera_node",
        name="camera_node",
        output="screen",
        parameters=[
            {
                "camera_type": "realsense",
                "realsense_width": LaunchConfiguration("realsense_width"),
                "realsense_height": LaunchConfiguration("realsense_height"),
                "realsense_fps": LaunchConfiguration("realsense_fps"),
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
            realsense_width_arg,
            realsense_height_arg,
            realsense_fps_arg,
            camera_node,
            stm_bridge_node,
        ]
    )
