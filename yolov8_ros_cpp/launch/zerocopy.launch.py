from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    container = ComposableNodeContainer(
        name="yolov8_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            ComposableNode(
                package="yolov8_ros_cpp",
                plugin="yolov8_ros_cpp::CameraNode",
                name="camera_node",
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
            ComposableNode(
                package="yolov8_ros_cpp",
                plugin="yolov8_ros_cpp::YOLOv8Node",
                name="yolov8_modular_node",
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
        output="screen",
    )

    return LaunchDescription([container])
