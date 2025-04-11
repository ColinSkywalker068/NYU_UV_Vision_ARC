from ament_index_python.packages import get_package_share_directory
import os

realsense_package_path = get_package_share_directory("realsense2_camera")
print(f"Realsense2 camera package path: {realsense_package_path}")
