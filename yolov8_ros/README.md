# yolov8_ros

## Quick Start

Build once with symlink install so edits take effect immediately:

```bash
cd /path/to/CV_Detection
source /opt/ros/<distro>/setup.bash
colcon build --packages-select yolov8_ros --symlink-install
source install/setup.bash
```

## Launch Configurations

### Basler + YOLO + STM32 Serial (full pipeline)

The **STM launch** uses a Basler camera and includes the serial bridge to the STM32.

```bash
ros2 launch yolov8_ros stm.launch.py
```

Override defaults:

```bash
ros2 launch yolov8_ros stm.launch.py \
    basler_exposure_us:=6000.0 \
    basler_gain_db:=8.0 \
    basler_auto_exposure:=false \
    basler_enable_grab_thread:=true \
    serial_port:=/dev/ttyACM0 \
    serial_verbose:=true
```

Notes:
- `basler_width` / `basler_height` in `stm.launch.py` feed the published image size for the ROS pipeline.
- The STM launch keeps `basler_use_full_sensor:=True`, so these values are not a hardware ROI crop unless that parameter is changed in a different launch path.

**Nodes launched:** `camera_node` (basler) → `yolov8_modular_node` → `stm32_serial_bridge`

---

### RealSense + YOLO (no serial)

```bash
ros2 launch yolov8_ros realsense.launch.py
```

Override defaults:

```bash
ros2 launch yolov8_ros realsense.launch.py \
    rs_fps:=30 \
    rs_exposure:=200.0 \
    show:=true \
    conf:=0.3
```

**Nodes launched:** `camera_node` (realsense) → `yolov8_modular_node`

---

### Running Individual Nodes

```bash
# Camera only (useful for testing camera init/exit)
ros2 run yolov8_ros camera_node --ros-args -p camera_type:=basler
ros2 run yolov8_ros camera_node --ros-args -p camera_type:=realsense

# YOLO node (subscribes to /image_raw topic)
ros2 run yolov8_ros yolov8_modular_node --ros-args -p show:=true

# Serial bridge
ros2 run yolov8_ros stm32_serial_bridge --ros-args -p serial_port:=/dev/ttyACM0
```

## Testing Tools

### Test Packet Sender (Serial)

Send randomized Seasky vision packets to the STM32:

```bash
python3 scripts/Serial/test_packet_sender.py --port /dev/ttyACM0 --rate_hz 20 --verbose
```

### Camera Test Scripts

Standalone scripts in `scripts/Camera/` (no ROS required):

```bash
python3 scripts/Camera/basler_test.py --show          # Basler + YOLO inference test
python3 scripts/Camera/realsense_test.py --show        # RealSense connectivity test
python3 scripts/Camera/realsense_max_fps_test.py       # RealSense max-native FPS preview
python3 scripts/Camera/realsense_max_fps_test.py --format yuyv --width 640 --height 360 --fps 90
python3 scripts/Camera/camera_zoom_test.py             # Basler sensor diagnostics
```

## Architecture

```
camera.py (unified module)
├── BaslerCamera     — pypylon SDK
└── RealSenseCamera  — pyrealsense2 SDK

Both implement: .initialize() / .get_frame() → BGR np.ndarray / .release()

camera_node.py
  ├─ creates BaslerCamera or RealSenseCamera based on camera_type param
  └─ publishes BGR frames on "image_raw" topic

yolov8_node.py
  ├─ subscribes to "image_raw"
  ├─ letterbox → YOLO inference → detections
  └─ publishes bounding_box + angles

stm32_serial_bridge.py
  ├─ subscribes to bounding_box + angles
  └─ sends Seasky packets to STM32 over serial
```

## Build Notes

If you build **without** `--symlink-install`, `ros2 run` uses copied files under `install/` and can get out of date until you run `colcon build` again. With `--symlink-install`, edits in `yolov8_ros/yolov8_ros/*.py` take effect immediately.
