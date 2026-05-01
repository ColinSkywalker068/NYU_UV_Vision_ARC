# C++ camera + inference in container

## What works today

- **C++ `camera_node`**: camera capture and `image_raw` publishing.
- **C++ detector / tracker / solver**: active runtime path lives in `yolov8_ros_cpp`.

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

## 4) C++ runtime launch

```bash
ros2 launch yolov8_ros_cpp stm.launch.py \
  detector_backend:=deepstream \
  camera_type:=basler
```

This starts the active C++ runtime path.

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
