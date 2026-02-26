from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """
    Launch file to run the camera and YOLOv8 node in a single container for zero-copy.
    """

    # Create a container for the nodes
    container = ComposableNodeContainer(
        name="yolov8_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="yolov8_ros",
                plugin="yolov8_ros.camera_node.CameraNode",
                name="camera_node",
                parameters=[
                    {
                        # Add any parameters for the camera node here
                        # e.g., 'camera_type': 'realsense'
                    }
                ],
            ),
            ComposableNode(
                package="yolov8_ros",
                plugin="yolov8_ros.yolov8_node.YOLOv8Node",
                name="yolov8_modular_node",
                parameters=[
                    {
                        # Add any parameters for the yolov8 node here
                        # e.g., 'model_path': '/path/to/your/model.engine'
                    }
                ],
            ),
        ],
        output="screen",
    )

    return LaunchDescription([container])
