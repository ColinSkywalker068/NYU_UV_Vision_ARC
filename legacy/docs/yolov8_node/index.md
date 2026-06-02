# YOLOv8 ROS Node Documentation

Technical reference for every module in `yolov8_ros/yolov8_ros/`.

---

## Table of Contents

- [System Overview](#system-overview)
- [Node: camera\_node](#node-camera_node)
- [Node: yolov8\_modular\_node](#node-yolov8_modular_node)
- [Node: stm32\_serial\_bridge\_node](#node-stm32_serial_bridge_node)
- [Module: camera.py](#module-camerapy)
- [Module: detector.py](#module-detectorpy)
- [Module: tracker.py](#module-trackerpy)
- [Module: utils.py](#module-utilspy)
- [Launch Files](#launch-files)
  - [stm.launch.py](#stmlaunchpy-basler--stm32-standard)
  - [realsense.launch.py](#realsenselaunchpy-realsense-no-stm32)
- [Topic Map](#topic-map)

---

## System Overview

```
┌──────────────┐   image_raw   ┌──────────────────────┐   bounding_box_angles   ┌───────────────────────┐
│  camera_node │ ─────────────▶│  yolov8_modular_node │ ───────────────────────▶│ stm32_serial_bridge   │
│  (BaslerCam  │               │  (detect + track)    │   override (Bool)       │ (Seasky serial TX/RX) │
│  or RS cam)  │               └──────────────────────┘ ───────────────────────▶│                       │
└──────────────┘                         ▲                                       └───────────────────────┘
                                         │ robot_id (Int32)                               │
                                         │                                                │ attitude_data
                                         └────────────────────────────────────────────────┘
                                           (optional, from STM32 RX)
```

Three ROS 2 nodes form the full detection pipeline:

| Node | Executable | Source file |
|---|---|---|
| `camera_node` | `camera_node` | `camera_node.py` |
| `yolov8_modular_node` | `yolov8_modular_node` | `yolov8_node.py` |
| `stm32_serial_bridge_node` | `stm32_serial_bridge` | `stm32_serial_bridge.py` |

---

## Node: `camera_node`

**File:** [`camera_node.py`](../../yolov8_ros/yolov8_ros/camera_node.py)

Captures frames from either a **Basler** (pypylon) or **Intel RealSense** camera and publishes them as `sensor_msgs/Image` at the configured FPS.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `camera_type` | string | `"basler"` | `"basler"` or `"realsense"` |
| `imgsz_width` | int | `640` | Published image width |
| `imgsz_height` | int | `640` | Published image height |
| **Basler-specific** | | | |
| `basler_exposure_us` | float | `6000.0` | Exposure in microseconds |
| `basler_gain_db` | float | `8.0` | Gain in dB |
| `basler_fps` | int | `60` | Target capture FPS |
| `basler_use_full_sensor` | bool | `True` | Use full sensor area (ignores ROI width/height) |
| `basler_centered_roi` | bool | `True` | Center ROI when not using full sensor |
| `basler_binning` | int | `1` | Pixel binning factor (≥1) |
| `basler_decimation` | int | `1` | Pixel decimation factor (≥1) |
| `basler_auto_exposure` | bool | `False` | Enable camera auto-exposure |
| `basler_auto_gain` | bool | `False` | Enable camera auto-gain |
| `basler_auto_white_balance` | bool | `False` | Enable camera auto white balance |
| `basler_enable_grab_thread` | bool | `True` | Use a background latest-frame grab thread |
| `basler_enable_transport_tuning` | bool | `True` | Apply Basler USB transport/buffer tuning |
| **RealSense-specific** | | | |
| `realsense_width` | int | `640` | Stream width |
| `realsense_height` | int | `480` | Stream height |
| `realsense_fps` | int | `60` | Stream FPS |
| `realsense_exposure` | float | `120.0` | Manual exposure value |
| `realsense_gain` | float | `64.0` | Manual gain value |
| `realsense_auto_exposure` | bool | `False` | Enable auto-exposure |
| `realsense_auto_white_balance` | bool | `False` | Enable auto white balance |
| `realsense_white_balance` | float | `4600.0` | Manual white balance (Kelvin) |
| `realsense_use_poll` | bool | `False` | Use `poll_for_frames` instead of `wait_for_frames` |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `image_raw` | `sensor_msgs/Image` | BGR8 frame at configured FPS; `header.frame_id` = camera type |

### Subscribed Topics

None.

### Behaviour

- Selects camera backend based on `camera_type`.
- Creates a ROS timer at `1/fps` Hz that fires `timer_callback`.
- On each tick, calls `camera.get_frame()` and publishes via `cv_bridge` with `bgr8` encoding.
- On shutdown, cancels the timer and releases the camera before calling `super().destroy_node()`.

---

## Node: `yolov8_modular_node`

**File:** [`yolov8_node.py`](../../yolov8_ros/yolov8_ros/yolov8_node.py)

The core inference node. Subscribes to `image_raw`, runs YOLOv8 detection/tracking, and publishes bounding boxes and pan/tilt angles.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | string | `<share>/models/best.engine` | Path to model file (`.engine`, `.pt`, `.onnx`) |
| `task` | string | `"detect"` | YOLO task: `detect`, `obb`, `segment` |
| `type` | string | `"yolo"` | Model backend: `yolo`, `rtdetr`, `nas` |
| `half` | bool | `True` | Use FP16 inference |
| `imgsz_width` | int | `640` | Inference image width |
| `imgsz_height` | int | `640` | Inference image height |
| `conf` | float | `0.05` | Detection confidence threshold |
| `show` | bool | `False` | Display OpenCV debug window |
| `enable_tracking` | bool | `True` | Use ByteTrack multi-object tracking |
| `persist_tracking` | bool | `True` | Persist tracker state across frames |
| `use_kalman` | bool | `False` | Apply custom Kalman lookahead prediction |
| `tracker_timeout` | int | `15` | Frames before dropping a lost track |
| `prediction_lookahead_steps` | float | `0.5` | Kalman lookahead multiplier (× dt) |
| `enable_color_verification` | bool | `False` | Verify armor plate color via HSV before publishing |
| `hfov_degrees` | float | `79.2` | Camera horizontal FOV for angle calculation |
| `vfov_degrees` | float | `63.78` | Camera vertical FOV for angle calculation |

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `image_raw` | `sensor_msgs/Image` | Raw frames from `camera_node` |
| `robot_id` | `std_msgs/Int32` | Our robot's ID — determines which team color to target |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `bounding_box` | `std_msgs/Int32MultiArray` | `[x1, y1, x2, y2]` of the selected target |
| `override` | `std_msgs/Bool` | `true` if a valid target is found this frame |
| `bounding_box_angles` | `std_msgs/Float32MultiArray` | `[yaw_rad, pitch_rad]` to target center |

### Image Pipeline (per frame)

1. **Convert** — `cv_bridge` decodes `Image` → BGR numpy array.
2. **Letterbox** — `letterbox_frame()` pads frame to `imgsz_width × imgsz_height` preserving aspect ratio.
3. **Inference** — `YOLODetector.track()` (ByteTrack) or `YOLODetector.predict()` depending on `enable_tracking`.
4. **Filter** — iterate detections; optionally call `check_enemy()` to discard wrong-color armor plates.
5. **Track / Predict** — if `use_kalman=True`, update `ArmorPlateTracker` per track ID and apply lookahead; otherwise pick the first valid box directly.
6. **Publish** — emit bbox + angles, or a centered no-target message when nothing is found.

### Team / Target Logic

`robot_id` is used to set `target_class`:
- IDs `1–11` → **Red team** → target **Blue** armor plates (`BLUE_CLASS = 0`)
- IDs `101–111` → **Blue team** → target **Red** armor plates (`RED_CLASS = 1`)

---

## Node: `stm32_serial_bridge_node`

**File:** [`stm32_serial_bridge.py`](../../yolov8_ros/yolov8_ros/stm32_serial_bridge.py)

Bridges ROS topics to/from the STM32 control board over a **Seasky serial protocol** connection.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `transmit_enable` | bool | `True` | Enable serial TX; set `False` for debug-only mode |
| `serial_port` | string | `"/dev/ttyUSB0"` | Serial device path |
| `serial_baudrate` | int | `115200` | Baud rate |
| `serial_verbose` | bool | `False` | Log every TX packet (not just detections) |
| `fire_mode` | int | `2` | Seasky TX fire mode bits `[0..3]` |
| `target_type` | int | `0` | Seasky TX target type bits `[0..15]` |

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `override` | `std_msgs/Bool` | Sets `target_state` — `true` = target converging |
| `bounding_box_angles` | `std_msgs/Float32MultiArray` | `[yaw_rad, pitch_rad]` forwarded over serial |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `attitude_data` | `std_msgs/Float32MultiArray` | `[yaw, pitch, roll]` received from STM32 |
| `attitude_flags` | `std_msgs/Int32MultiArray` | `[enemy_color, work_mode, bullet_speed, flags_reg, cmd_id]` from STM32 RX |

### Seasky Protocol Summary

Each serial frame:

```
[ SOF(0xA5) | len_lo | len_hi | CRC8 ] [ cmd_id(2B) | flags(2B) | payload ] [ CRC16 ]
```

- **TX command** `0x0001` — sends `[pitch_f32, yaw_f32]` + packed flags to STM32.
- **RX command** `0x0002` — receives `[yaw_f32, pitch_f32, roll_f32]` + attitude flags from STM32.
- CRC8 covers the 3-byte header; CRC16 covers the full frame minus the trailing 2 bytes.
- Frame assembly: `build_seasky_frame()` → `build_vision_flags()` → `serial.write()`.
- Frame parsing: byte-stream buffered in `rx_buffer`; SOF-aligned, header-CRC-validated, then CRC16-validated before dispatch.

### RX Poll

A 10 ms timer (`handle_rx_poll`) drains the serial input buffer each tick, parses complete frames, and publishes `attitude_data` and `attitude_flags`.

---

## Module: `camera.py`

**File:** [`camera.py`](../../yolov8_ros/yolov8_ros/camera.py)

Pure-Python camera backends. No ROS dependency — used by `camera_node.py`.

### `BaslerCamera`

Wraps **pypylon** `InstantCamera` with configurable ROI, binning, decimation, exposure, and gain.

| Method | Description |
|---|---|
| `initialize()` | Opens first available Basler device, configures it, starts grabbing (LatestImageOnly strategy) |
| `get_frame()` | Retrieves latest grab result, converts to BGR8 numpy array, releases grab; returns `None` on timeout |
| `get_applied_settings()` | Returns dict of settings actually applied to the hardware |
| `release()` | Stops grabbing and closes the device |

**Configuration detail:** `_configure_camera()` applies binning/decimation first (which reduces effective max size), then sets ROI width/height/offsets. Pixel format is auto-selected from a preferred order: `BayerRG8 > BayerBG8 > … > Mono8`.

### `RealSenseCamera`

Wraps **pyrealsense2** pipeline for color-only streaming.

| Method | Description |
|---|---|
| `initialize()` | Disables all streams, enables color stream at target resolution/FPS, starts pipeline, configures sensor options |
| `get_frame()` | Calls `wait_for_frames` (or `poll_for_frames` if `use_poll=True`), returns a **deep copy** of the color frame data |
| `release()` | Stops the pipeline |

> **Why `np.array(..., copy=True)`?** `np.asanyarray` gives a zero-copy view into the SDK's internal buffer. When the SDK recycles that buffer as the next frame arrives, the old view becomes a dangling reference — causing a double-free if YOLO still holds it. Copying the data prevents this.

---

## Module: `detector.py`

**File:** [`detector.py`](../../yolov8_ros/yolov8_ros/detector.py)

Thin wrapper around Ultralytics model backends.

### `YOLODetector`

| Constructor arg | Values | Backend loaded |
|---|---|---|
| `model_type="yolo"` | default | `ultralytics.YOLO` |
| `model_type="rtdetr"` | RT-DETR | `ultralytics.RTDETR` |
| `model_type="nas"` | NAS | `ultralytics.NAS` |

| Method | Description |
|---|---|
| `track(image, **kwargs)` | Calls `model.track()` with `verbose=False` unless overridden |
| `predict(image, **kwargs)` | Calls `model(image, **kwargs)` |

---

## Module: `tracker.py`

**File:** [`tracker.py`](../../yolov8_ros/yolov8_ros/tracker.py)

Custom NumPy-based Kalman filter for armor plate motion estimation.

### `ArmorPlateTracker`

6-state linear Kalman filter: **[x, y, vₓ, vᵧ, aₓ, aᵧ]** — position, velocity, and acceleration.

| Method | Description |
|---|---|
| `predict(dt)` | Propagates state forward by `dt` seconds; returns `(x, y)` |
| `update(x, y, width, height)` | Kalman correction step with new measurement; updates stored `width`/`height` |

Noise defaults: `proc_noise=(0.01, 0.05, 0.1)`, `meas_noise=0.15`.

The `.kf.statePost` property shim lets `yolov8_node.py` read velocity/acceleration state for lookahead prediction without needing OpenCV's `KalmanFilter`.

### `TrackerWrapper`

Pairs an `ArmorPlateTracker` with a `lost_frames` counter. When `lost_frames >= tracker_timeout`, the track is removed from `active_trackers`.

---

## Module: `utils.py`

**File:** [`utils.py`](../../yolov8_ros/yolov8_ros/utils.py)

Shared constants and image utilities.

### Constants

| Name | Value | Description |
|---|---|---|
| `RED_TEAM` | `(1–11)` | Robot IDs belonging to the red team |
| `BLUE_TEAM` | `(101–111)` | Robot IDs belonging to the blue team |
| `RED_CLASS` | `1` | YOLO class index for red armor |
| `BLUE_CLASS` | `0` | YOLO class index for blue armor |
| `BLUE_RANGES` | HSV `[100,150,150]→[128,255,255]` | Hue range for blue armor detection |
| `RED_RANGES` | HSV `[0,150,150]→[10,255,255]` + `[170,150,150]→[180,255,255]` | Two hue spans (red wraps around 180°) |

### Functions

#### `letterbox_frame(frame, target_w, target_h)`

Resizes `frame` to fit within `target_w × target_h` while preserving aspect ratio, then pads symmetrically with gray (114, 114, 114). YOLO models expect consistently-sized square inputs — this avoids distortion.

#### `calculate_angles(x1, y1, x2, y2, imgsz_width, imgsz_height, hfov_rad, vfov_rad)`

Converts bounding box pixel coordinates to yaw/pitch angles (radians):

```
pixel_offset_x = (cx - W/2) / (W/2)          # normalise to [-1, 1]
yaw   = atan(pixel_offset_x × tan(hfov/2))
pitch = atan(pixel_offset_y × tan(vfov/2))
```

Returns `(yaw_rad, pitch_rad)`.

#### `check_enemy(roi_image, target_class)`

Converts ROI to HSV, applies `BLUE_RANGES` or `RED_RANGES` masks, counts colored pixels. Returns `(True, pixel_count)` if colored area exceeds 1 % of the ROI area.

---

## Launch Files

### `stm.launch.py` — Basler + STM32 (standard)

**Used with:** `uv run yolov8_ros stm.launch.py`

Launches **3 nodes**: `camera_node` (Basler), `yolov8_modular_node`, `stm32_serial_bridge_node`.

| Argument | Default | Description |
|---|---|---|
| `serial_port` | `/dev/ttyUSB0` | STM32 serial device |
| `serial_baudrate` | `115200` | Serial baud rate |
| `serial_verbose` | `false` | Log all serial TX packets |
| `basler_width` | `640` | Published/inference image width |
| `basler_height` | `640` | Published/inference image height |
| `basler_fps` | `60` | Basler FPS |
| `basler_exposure_us` | `6000.0` | Basler exposure (µs) |
| `basler_gain_db` | `8.0` | Basler gain (dB) |
| `basler_auto_exposure` | `false` | Enable Basler auto exposure |
| `basler_auto_gain` | `false` | Enable Basler auto gain |
| `basler_auto_white_balance` | `false` | Enable Basler auto white balance |
| `basler_enable_grab_thread` | `true` | Use background latest-frame grabbing |
| `basler_enable_transport_tuning` | `true` | Apply Basler transport tuning |
| `use_kalman` | `true` | Enable Kalman lookahead in YOLO node |
| `show` | `false` | Show OpenCV debug window |

**Fixed values (not overridable at launch):** `basler_use_full_sensor=True`, `basler_centered_roi=True`, `transmit_enable=True`.

Because `basler_use_full_sensor=True` is fixed in this launch path, `basler_width` / `basler_height` do not crop the Basler sensor. They define the ROS pipeline image size while the camera stays on the full sensor area.

Example override:
```bash
uv run yolov8_ros stm.launch.py serial_port:=/dev/ttyUSB1 basler_exposure_us:=3000.0
```

---

### `realsense.launch.py` — RealSense (no STM32)

**Used with:** `uv run yolov8_ros realsense.launch.py`

Launches **2 nodes**: `camera_node` (RealSense), `yolov8_modular_node`. No serial bridge.

| Argument | Default | Description |
|---|---|---|
| `rs_width` | `640` | RealSense stream width |
| `rs_height` | `480` | RealSense stream height |
| `rs_fps` | `30` | RealSense FPS |
| `rs_exposure` | `120.0` | Manual exposure |
| `rs_gain` | `64.0` | Manual gain |
| `rs_auto_exposure` | `false` | Enable auto-exposure |
| `rs_auto_white_balance` | `false` | Enable auto white balance |
| `rs_white_balance` | `4600.0` | Manual white balance (K) |
| `rs_use_poll` | `false` | Use poll instead of blocking wait |
| `model_path` | `<share>/models/best.engine` | YOLO model file |
| `task` | `detect` | YOLO task |
| `conf` | `0.25` | Confidence threshold |
| `show` | `true` | OpenCV debug window |
| `use_kalman` | `false` | Kalman lookahead |
| `enable_color_verification` | `false` | HSV color check |
| `hfov_degrees` | `69.4` | Horizontal FOV (D435 default) |
| `vfov_degrees` | `42.5` | Vertical FOV (D435 default) |

> For a **D455**, set `hfov_degrees:=87.0 vfov_degrees:=58.0`.

Example override:
```bash
uv run yolov8_ros realsense.launch.py hfov_degrees:=87.0 vfov_degrees:=58.0 conf:=0.4
```

---

## Topic Map

```
camera_node
  └── pub: image_raw (sensor_msgs/Image)

yolov8_modular_node
  ├── sub: image_raw
  ├── sub: robot_id (std_msgs/Int32)
  ├── pub: bounding_box (std_msgs/Int32MultiArray)   → [x1,y1,x2,y2]
  ├── pub: override (std_msgs/Bool)                  → detection flag
  └── pub: bounding_box_angles (std_msgs/Float32MultiArray) → [yaw,pitch]

stm32_serial_bridge_node
  ├── sub: override
  ├── sub: bounding_box_angles
  ├── pub: attitude_data (std_msgs/Float32MultiArray) → [yaw,pitch,roll]
  └── pub: attitude_flags (std_msgs/Int32MultiArray)  → [enemy_color,work_mode,bullet_speed,flags,cmd_id]
```
