# C++ camera + inference in container

## What works today

- **C++ `camera_node`**: RealSense capture and `image_raw` publishing — use this.
- **C++ `yolov8_modular_node`**: Inference is a **stub** (`DeepStreamDetector::infer` returns no detections). Turning on real YOLO in C++ means implementing TensorRT/DeepStream (or similar) in `yolov8_ros_cpp` — that is a full backend project, not a flag flip.
- **Recommended hybrid**: run **C++ camera** + **Python `yolov8_modular_node`** on the same `image_raw` topic for real detections now.

## 1) Start container with GUI + devices

Run on host:

```bash
xhost +local:root
docker run --rm -it \
  --runtime nvidia \
  --network host \
  --privileged \
  --ipc host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -v /dev:/dev \
  -v "$PWD":/home/ultraviolet/CV_Detection \
  cv-realsense:pylon
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

## 4a) Hybrid (recommended): C++ camera + Python YOLO

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
```

Both nodes use topic `image_raw` (no remap needed if default names and no extra namespace).

## 4b) All-C++ launch (camera works; YOLO stub only)

```bash
ros2 launch yolov8_ros_cpp realsense.launch.py \
  model_path:=/home/ultraviolet/CV_Detection/yolov8_ros/model_test/bestmerge.engine \
  task:=obb \
  rs_width:=640 rs_height:=480 rs_fps:=60 \
  rs_auto_exposure:=true \
  show:=true
```

This starts `camera_node` (C++) + `yolov8_modular_node` (C++), but the detector still logs `deepstream_stub` until you implement inference in `deepstream_detector.cpp`.

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
- C++ `yolov8_modular_node` stays on `deepstream_stub` until you add a real TensorRT/DeepStream path in the workspace.

