"""Launch the legacy combined perception component with intra-process transport."""

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Create a composable container for zero-copy perception experiments."""
    container = ComposableNodeContainer(
        name="yolo_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            ComposableNode(
                package="yolo_cpp",
                plugin="yolo_cpp::PerceptionNode",
                name="perception_node",
                extra_arguments=[{"use_intra_process_comms": True}],
            ),
        ],
        output="screen",
    )

    return LaunchDescription([container])
