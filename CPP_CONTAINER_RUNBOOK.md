# C++ camera + inference in container

## What works today

- **C++ `camera_node`**: camera capture and `image_raw` publishing.
- **C++ detector / tracker / solver**: active runtime path lives in `yolo_cpp`.

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
colcon build --packages-select yolo_cpp --symlink-install
```

Important: Basler support is only compiled if the Pylon SDK is visible inside the container at build time,
typically at `/opt/pylon`. `services/run-cplusplus.sh` now auto-mounts a host `/opt/pylon` into the container
when that directory exists on the host. If Pylon is missing during `colcon build`, the build succeeds but the
Basler backend is disabled and `camera_type:=basler` will throw at runtime.

## 3) Source ROS and workspace overlay

Run in every new shell:

```bash
source /opt/ros/humble/setup.bash
source /home/ultraviolet/CV_Detection/install/setup.bash
```

Quick package check:

```bash
ros2 pkg list | rg yolo_cpp
```

## 4) C++ runtime launch

```bash
ros2 launch yolo_cpp stm.launch.py \
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

- If `Package 'yolo_cpp' not found`, re-run the source commands in section 3.
