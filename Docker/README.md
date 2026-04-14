# How to Run the Object Detection Pipeline

https://limewire.com/d/Ha8tz#M7d4qSlJjH

## Quick Start (ROS 2 Launch)

Instead of multiple windows, the new modular node handles both camera and inference in one.
Open a terminal, run `uv update` to ensure environment is current.
Then run:

```bash
uv docker
uv build yolov8_ros
uv run yolov8_ros launch.py
```

This runs the `yolov8_modular_node` using the configuration in `/yolov8_ros/launch/launch.py`.

## Manual Run (ROS 2 Node)

You can also run the node directly with custom parameters:

```bash
uv docker
uv build yolov8_ros
uv run yolov8_ros yolov8_modular_node --ros-args -p model_path:='/home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine' -p task:='obb' -p show:=True
```

## Python Standalone Run (No ROS)

To test the camera or components without ROS, you can run the scripts in `yolov8_ros/scripts/`:

**Basler Camera Test:**

```bash
uv docker
python3 src/yolov8_ros/scripts/Camera/basler_test.py --show
```

(Note: Adjust the path to the script based on where it is mounted in the container, usually under `src/` or similar).

---

## Legacy / Realsense Info

(If using the old Realsense setup, see below)

You will need two terminal windows:
In either of the windows, run `uv update`.
In the first run the yolov8_node:

```
uv docker
uv build yolov8_ros yolov8_node --ros-args -r video_frames:=/camera/camera/color/image_raw ...
```

In the second window run the realsense_node:

```
uv docker
ros2 launch realsense2_camera rs_launch.py rgb_camera.profile:=640x480x60
```

Camera Options for RealSense (Check online for up to date):

```
/camera/camera:
  .camera.color.image_raw.format
  .camera.color.image_raw.jpeg_quality
  ...

```

## Build and Push RealSense Wrapper + Ultralytics

Use the helper script to build a thin wrapper image (adds `vim` and updates `ultralytics`) and push it:

```bash
cd Docker
./build_realsense_wrapper_ultra.sh
```

Optional arguments:

```bash
./build_realsense_wrapper_ultra.sh <repo> <base-tag> <target-tag>
```

Example:

```bash
./build_realsense_wrapper_ultra.sh nyuultraviolet/cv-dockerfile jp60-pytorch21-humble-tensorrt-realsense-wrapper jp60-pytorch21-humble-tensorrt-realsense-wrapper-ultra
```

Compose now defaults to the ultra image. To override manually:

```bash
CV_BASE_IMAGE=nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-realsense-wrapper-ultra docker compose up --build
```
