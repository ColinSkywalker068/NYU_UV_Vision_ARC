# C++ Nodes in Container: Build and Run

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

## 2) Build C++ package inside container

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

## 4) Launch both C++ nodes (camera + yolov8)

```bash
ros2 launch yolov8_ros_cpp realsense.launch.py \
  model_path:=/home/ultraviolet/CV_Detection/yolov8_ros/model_test/bestmerge.engine \
  task:=obb \
  rs_width:=640 rs_height:=480 rs_fps:=60 \
  rs_auto_exposure:=true \
  show:=true
```

This launch starts:

- `camera_node` (C++)
- `yolov8_modular_node` (C++)

## 5) Optional runtime checks (new terminal in same container)

```bash
source /opt/ros/humble/setup.bash
source /home/ultraviolet/CV_Detection/install/setup.bash
ros2 node list
ros2 topic list
ros2 topic hz /camera/image_raw
```

## Notes

- If `Package 'yolov8_ros_cpp' not found`, re-run the source commands in section 3.
- Current C++ YOLO backend may log `deepstream_stub`; in that mode camera runs, but inference output is fallback/scaffold behavior.

