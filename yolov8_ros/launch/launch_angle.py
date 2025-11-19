from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Get the package share directory
    package_share_dir = get_package_share_directory("yolov8_ros")

    # Calculate path to scripts directory
    # From share/yolov8_ros, go up to workspace root, then to src/yolov8_ros/scripts
    # This works when package is installed
    workspace_root = os.path.dirname(os.path.dirname(package_share_dir))
    installed_script_path = os.path.join(
        workspace_root, "src", "yolov8_ros", "scripts", "yolov8_noRos_angle.py"
    )

    # Alternative: if running from source (launch file in source tree)
    launch_file_dir = os.path.dirname(os.path.abspath(__file__))
    source_script_path = os.path.join(
        os.path.dirname(launch_file_dir),  # Go up from launch/ to yolov8_ros/
        "scripts",
        "yolov8_noRos_angle.py",
    )

    # Use source path if it exists (development), otherwise use installed path
    if os.path.exists(source_script_path):
        script_to_run = source_script_path
    elif os.path.exists(installed_script_path):
        script_to_run = installed_script_path
    else:
        # Fallback: try to find it relative to current working directory
        script_to_run = os.path.join(
            os.getcwd(), "yolov8_ros", "scripts", "yolov8_noRos_angle.py"
        )

    return LaunchDescription(
        [
            # YOLOv8 Angle Node - runs the Python script directly
            # This script uses pyrealsense2 directly (no ROS camera node needed)
            ExecuteProcess(
                cmd=[
                    "python3",
                    script_to_run,
                    "--ros-args",
                    "-p",
                    "model_path:=/home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine",
                    "-p",
                    "task:=detect",
                    "-p",
                    "show:=False",
                    "-p",
                    "conf:=0.05",
                    "-p",
                    "hfov_degrees:=79.2",
                    "-p",
                    "vfov_degrees:=63.78",
                    "-r",
                    "__node:=yolov8_angle_node",  # Unique node name to avoid confusion
                ],
                output="screen",
                name="yolov8_angle_node_launcher",
            ),
        ]
    )
