# C++ camera + inference in container

## What works today

- **C++ `camera_node`**: camera capture and `image_raw` publishing.
- **C++ `yolov8_modular_node`**: the branch is moving toward a dedicated C++/DeepStream container path, separate from the original Python image.
- **Recommended hybrid**: use the Python image for the old hybrid flow, and use the new C++ image for the DeepStream branch.

## 1) Start container with GUI + devices

Run on host:

```bash
xhost +local:root
./services/run-cplusplus.sh
```

Optional cleanup on host after container exits:

```bash
xhost -local:root
```

## 2) Build packages inside container

**C++ only:**

```bash
cd /home/ultraviolet/CV_Detection
source /opt/ros/humble/setup.bash
colcon build --packages-select yolov8_ros_cpp --symlink-install
```

**C++ camera + Python YOLO (hybrid):**

```bash
cd /home/ultraviolet/CV_Detection
source /opt/ros/humble/setup.bash
colcon build --packages-select yolov8_ros_cpp yolov8_ros --symlink-install
```

Ensure Python deps (image is usually enough): `pip3 show ultralytics cv_bridge` should be present; install `cv_bridge` via apt if missing:

```bash
sudo apt-get update && sudo apt-get install -y ros-humble-cv-bridge
```

## 3) Source ROS and workspace overlay

Run in every new shell:

```bash
source /opt/ros/humble/setup.bash
source /home/ultraviolet/CV_Detection/install/setup.bash
```

Quick package check:

```bash
ros2 pkg list | rg yolov8_ros_cpp
```

## 4a) Legacy hybrid: C++ camera + Python YOLO

Two terminals, both after sourcing (section 3).

**Terminal A — C++ camera only:**

```bash
ros2 run yolov8_ros_cpp camera_node --ros-args \
  -p camera_type:=realsense \
  -p realsense_width:=640 \
  -p realsense_height:=480 \
  -p realsense_fps:=60 \
  -p realsense_auto_exposure:=true
```

**Terminal B — Python inference (real YOLO / TensorRT engine):**

```bash
ros2 run yolov8_ros yolov8_modular_node --ros-args \
  -p model_path:=/home/ultraviolet/CV_Detection/yolov8_ros/model_test/bestmerge.engine \
  -p task:=obb \
  -p imgsz_width:=640 \
  -p imgsz_height:=480 \
  -p show:=false

Note on YOLO26 and post-processing:
- Ultralytics YOLO26 models can export in two different inference modes.
- `end2end=true` exports already contain final detections, so applying a legacy NMS step after inference is incorrect.
- `end2end=false` exports still produce raw predictions and do require host-side decode + NMS, even for YOLO26. This commonly happens for TensorRT INT8 exports because Ultralytics may disable end-to-end export for compatibility.
- The C++ pipeline now branches on engine metadata instead of assuming one fixed YOLO post-process path.
- Ultralytics INT8 engines are supported in the common export configuration where TensorRT keeps FP32 or FP16 I/O bindings and quantizes internally.
- Engines with true INT8 input or output bindings still need explicit binding-scale handling and are rejected with a clear runtime error.
```

Both nodes use topic `image_raw` (no remap needed if default names and no extra namespace).

## 4b) C++ DeepStream launch

```bash
ros2 launch yolov8_ros_cpp stm.launch.py \
  detector_backend:=deepstream \
  camera_type:=basler
```

This starts the new C++ DeepStream branch runtime from the separate image. Use the Python image only for the old hybrid flow.

## 5) Optional runtime checks (new terminal in same container)

```bash
source /opt/ros/humble/setup.bash
source /home/ultraviolet/CV_Detection/install/setup.bash
ros2 node list
ros2 topic list
ros2 topic hz /image_raw
```

## Notes

- If `Package 'yolov8_ros_cpp' not found`, re-run the source commands in section 3.
- If `Package 'yolov8_ros' not found`, build `yolov8_ros` (section 2) and source `install/setup.bash` again.
- Use the C++ DeepStream image for the branch runtime, and keep the Python image for the old hybrid path.
