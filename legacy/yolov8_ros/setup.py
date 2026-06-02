import os
from glob import glob
from setuptools import find_packages, setup

package_name = "yolov8_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.[pxy][yma]*")),
        ),
        (
            os.path.join("share", package_name, "models"),
            [f for f in glob(os.path.join("models", "*")) if os.path.isfile(f)],
        ),
    ],
    install_requires=["setuptools", "pyserial"],
    zip_safe=True,
    maintainer="parallels",
    maintainer_email="shubhamgupto@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolov8_modular_node = yolov8_ros.yolov8_node:main",
            "camera_node = yolov8_ros.camera_node:main",
            "stm32_serial_bridge = yolov8_ros.stm32_serial_bridge:main",
            "stm32_yaw_packet_test = yolov8_ros.stm32_yaw_packet_test:main",
            # "frame_grabber = yolov8_ros.frame_grabber:main",
            # "yolov8_noRos = yolov8_ros.yolov8_noRos:main",
            # "yolov8_old = yolov8_ros.yolov8_old:main",
            # "yolov8_ONNX = yolov8_ros.yolov8_ONNX:main",
        ],
    },
)
