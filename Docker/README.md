# How to Run the Object Detection Pipeline

https://limewire.com/d/Ha8tz#M7d4qSlJjH

## Quick Start (ROS 2 Launch)

Instead of multiple windows, the modular C++ stack handles camera, detection, tracking, and aiming in one launch flow.
Open a terminal, run `uv update` to ensure environment is current.
Then run:

```bash
uv docker
uv build yolo_cpp
uv run yolo_cpp basler.launch.py
```

This runs the active Basler field stack using the package launch files. `stm.launch.py` remains as a compatibility alias for older commands.

## Manual Run (ROS 2 Node)

You can also run the node directly with custom parameters:

```bash
uv docker
uv build yolo_cpp
uv run yolo_cpp yolo_modular_node --ros-args -p model_path:='/home/ultraviolet/CV_Detection/yolo_cpp/models/best-roboflow-nitish-obb.onnx_b1_gpu0_fp16.engine' -p task:='obb' -p show:=True
```

## Python Standalone Run (No ROS)

To test the camera or components without ROS, you can still use the legacy Python helper scripts:

**Basler Camera Test:**

```bash
uv docker
python3 <legacy-python-camera-script> --show
```

(Note: Adjust the path to the script based on where it is mounted in the container, usually under `src/` or similar).

---

## Legacy / Realsense Info

(If using the old Realsense setup, see below)

You will need two terminal windows:
In either of the windows, run `uv update`.
In the first run the legacy detector node:

```
uv docker
uv build
uv run yolo_cpp realsense.launch.py
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

Note: the wrapper pins `numpy<2` to avoid ABI breakage with wheels compiled against NumPy 1.x.

Optional arguments:

```bash
./build_realsense_wrapper_ultra.sh <repo> <base-tag> <target-tag>
```

Example:

```bash
./build_realsense_wrapper_ultra.sh nyuultraviolet/cv-dockerfile jp60-pytorch21-humble-tensorrt-realsense-wrapper jp60-pytorch21-humble-tensorrt-realsense-wrapper-ultra
```

Compose now defaults to the C++ image. To override manually:

```bash
CV_BASE_IMAGE=nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus docker compose up --build
```

## Basler Pylon C++ Image

If `uv docker` can load `pylonsrc` but the clean service image cannot, bake the working container's Pylon install into a stable local image:

```bash
cd Docker
UPDATE_STM_ENV=1 ./bake_cplusplus_pylon_image.sh
```

That creates `cv-detection-cplusplus-pylon:local`, verifies `gst-inspect-1.0 pylonsrc`, and updates `/etc/cv-detection/stm-stack.env` when `UPDATE_STM_ENV=1`.

For the fully reproducible vendor-package path, copy the Basler SDK and GStreamer package archives under `Docker/pylon/`, then build with `PYLON_PACKAGE` and `PYLON_GSTREAMER_PACKAGE` set before running `./build_cplusplus.sh`.
