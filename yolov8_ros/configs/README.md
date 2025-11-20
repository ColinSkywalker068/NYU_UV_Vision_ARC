# DeepStream nvinfer Config Files

This directory contains configuration files for the DeepStream nvinfer plugin used in the GStreamer pipeline.

## Quick Run (Copy & Paste)

**1. Start Docker:**

```bash
./services/run-deepstream.sh
```

**2. Inside container, run:**

```bash
# Simplest - just run the wrapper script
./yolov8_ros/scripts/run_gstreamer.sh

# With display
./yolov8_ros/scripts/run_gstreamer.sh --display

# Custom model
./yolov8_ros/scripts/run_gstreamer.sh --model-engine /home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb2.engine
```

That's it! The script uses defaults and auto-generates the config file.

---

## How to Run

### Prerequisites

1. **Docker container must be running** with the RealSense image:

   ```bash
   ./services/run-deepstream.sh
   ```

   Or manually:

   ```bash
   docker run --rm -it --runtime nvidia \
     --device /dev/video0:/dev/video0 \
     --device /dev/nvhost-ctrl \
     --device /dev/nvhost-ctrl-gpu \
     --device /dev/nvhost-prof-gpu \
     --device /dev/nvhost-vic \
     --device /dev/nvmap \
     -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra:ro \
     -v /usr/lib/aarch64-linux-gnu/gstreamer-1.0:/usr/lib/aarch64-linux-gnu/gstreamer-1.0:ro \
     -v "$PWD":/home/ultraviolet/CV_Detection \
     --workdir /home/ultraviolet/CV_Detection \
     --network host --ipc host \
     cv-realsense:latest /bin/bash
   ```

2. **Camera must be connected** and accessible at `/dev/video0` (or specify with `--device`)

3. **TensorRT engine file** must exist (e.g., `yolov8_ros/models/best.engine`)

### Running the Pipeline

**Easiest way - use the wrapper script:**

Inside the Docker container:

```bash
# Basic run (headless)
./yolov8_ros/scripts/run_gstreamer.sh

# With display
DISPLAY_FLAG=true ./yolov8_ros/scripts/run_gstreamer.sh

# Custom model
MODEL_ENGINE=/home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb2.engine ./yolov8_ros/scripts/run_gstreamer.sh
```

**Or run directly with Python:**

## Quick Start

### Option 1: Auto-generate Config (Recommended)

If you don't have a config file, you can now run the pipeline with just your TensorRT engine file:

```bash
python3 yolov8_ros/scripts/yolov8_op.py \
  --model-engine /home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine \
  --num-classes 1 \
  --threshold 0.5 \
  --display
```

The script will automatically generate a config file next to your engine file (e.g., `best_nvinfer_config.txt`).

**Example with display:**

```bash
python3 yolov8_ros/scripts/yolov8_op.py \
  --model-engine /home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine \
  --num-classes 1 \
  --threshold 0.5 \
  --nms-threshold 0.4 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --display
```

**Example headless (no display):**

```bash
python3 yolov8_ros/scripts/yolov8_op.py \
  --model-engine /home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine \
  --num-classes 1 \
  --threshold 0.5
```

### Option 2: Use Template Config

1. Copy the template:

   ```bash
   cp yolov8_ros/configs/nvinfer_config_template.txt yolov8_ros/configs/my_config.txt
   ```

2. Edit `my_config.txt` and update:

   - `model-engine-file`: Path to your TensorRT `.engine` file
   - `num-detected-classes`: Number of classes in your model
   - `threshold`: Detection confidence threshold
   - Other parameters as needed

3. Run with the config:
   ```bash
   python3 yolov8_ros/scripts/yolov8_op.py \
     --config /home/ultraviolet/CV_Detection/yolov8_ros/configs/my_config.txt \
     --display
   ```

### Command-Line Arguments

| Argument          | Required | Default       | Description                                          |
| ----------------- | -------- | ------------- | ---------------------------------------------------- |
| `--config`        | No\*     | None          | Path to DeepStream nvinfer config file               |
| `--model-engine`  | No\*     | None          | Path to TensorRT engine file (auto-generates config) |
| `--num-classes`   | No       | 1             | Number of classes (used when auto-generating config) |
| `--threshold`     | No       | 0.5           | Detection confidence threshold                       |
| `--nms-threshold` | No       | 0.4           | NMS IoU threshold                                    |
| `--device`        | No       | `/dev/video0` | V4L2 camera device path                              |
| `--width`         | No       | 640           | Input video width                                    |
| `--height`        | No       | 480           | Input video height                                   |
| `--fps`           | No       | 30            | Input video FPS                                      |
| `--display`       | No       | False         | Enable display output (nveglglessink)                |
| `--log-level`     | No       | 2             | GStreamer debug level (0-7)                          |

\* Either `--config` or `--model-engine` must be provided.

### Troubleshooting

**Camera not found:**

- Check camera is connected: `ls -l /dev/video*`
- Try different device: `--device /dev/video1`

**Config file not found:**

- Use `--model-engine` to auto-generate config
- Or provide full absolute path to `--config`

**Display not working:**

- Run without `--display` flag for headless mode
- Ensure X11 forwarding if using display: `xhost +local:root`

**Pipeline fails to start:**

- Check GStreamer plugins: `gst-inspect-1.0 nvinfer`
- Verify DeepStream is installed: `ls /opt/nvidia/deepstream/deepstream`
- Check log level: `--log-level 4` for more verbose output

## Config File Parameters

Key parameters you may need to adjust:

- **model-engine-file**: Path to your TensorRT engine file (`.engine`)
- **num-detected-classes**: Number of object classes your model detects
- **threshold**: Minimum confidence score for detections (0.0-1.0)
- **nms-iou-threshold**: IoU threshold for non-maximum suppression
- **output-blob-names**: Output layer name (typically "output0" for YOLOv8)

## Available Models

Check `yolov8_ros/models/` for available engine files:

- `best.engine`
- `bestObb2.engine`
- etc.

---

## GStreamer Pipeline vs Normal Pipeline Comparison

### `yolov8_op.py` (GStreamer/DeepStream Pipeline)

**Architecture:**

- GStreamer pipeline using DeepStream plugins
- Pipeline: `v4l2src` → `capsfilter` (CPU) → `nvvidconv` → `capsfilter` (NVMM) → `nvstreammux` → `nvinfer` → `nvvideoconvert` → `nvdsosd` → `nveglglessink`/`fakesink`
- Uses NVMM (NVIDIA Video Memory Manager) for zero-copy GPU memory transfers
- All processing happens in GStreamer pipeline

**Camera Input:**

- V4L2 device (`/dev/video0`) via `v4l2src` element
- Format: YUY2 → NV12 conversion via `nvvidconv`
- No RealSense SDK dependency

**Inference:**

- DeepStream `nvinfer` plugin
- Requires DeepStream nvinfer config file (`.txt`)
- TensorRT engine loaded via config file
- Inference runs entirely in GPU pipeline

**Output:**

- `nvdsosd` for on-screen display (bounding boxes)
- `nveglglessink` for display or `fakesink` for headless
- No ROS integration
- No custom post-processing

**Dependencies:**

- GStreamer 1.0 with DeepStream plugins
- Python GObject introspection (`gi.repository.Gst`)
- DeepStream 6.4 runtime
- No Ultralytics library
- No ROS2

**Code Structure:**

- 319 lines
- Single-purpose: run inference pipeline
- No tracking, filtering, or color detection
- No frame-by-frame Python processing

---

### `yolov8_noRos.py` (Normal Pipeline)

**Architecture:**

- Python-based processing loop
- Pyrealsense2 → OpenCV → Ultralytics YOLO → Post-processing → ROS2
- CPU/GPU hybrid: frames in CPU, inference on GPU via Ultralytics

**Camera Input:**

- Pyrealsense2 SDK (`rs.pipeline`)
- Direct BGR8 frame access
- RealSense-specific controls (exposure, white balance, etc.)

**Inference:**

- Ultralytics YOLO library (`YOLO()`, `model.track()`)
- TensorRT engine loaded via Ultralytics
- Supports YOLO, RTDETR, NAS model types
- Letterbox preprocessing (maintains aspect ratio)

**Post-Processing:**

- Kalman Filter tracking (`ArmorPlateTracker` class)
- ByteTrack multi-object tracking (`tracker="bytetrack.yaml"`)
- HSV-based enemy color detection (`check_enemy()`)
- Prediction lookahead for lag compensation
- Team-based target selection (RED_TEAM/BLUE_TEAM)

**Output:**

- ROS2 publishers: `bounding_box` (Int32MultiArray), `override` (Bool)
- ROS2 subscriber: `robot_id` (Int32)
- Optional OpenCV visualization (`cv2.imshow`)
- Frame-by-frame logging and FPS monitoring

**Dependencies:**

- ROS2 (rclpy)
- Ultralytics YOLO library
- Pyrealsense2
- OpenCV (cv2)
- NumPy
- No GStreamer/DeepStream

**Code Structure:**

- 1061 lines
- ROS2 node class (`YOLOv8Node`)
- Timer-based frame acquisition
- Complex state management (trackers, frame timing)
- Configurable via ROS2 parameters

---

### Key Differences Summary

| Aspect              | GStreamer (`yolov8_op.py`) | Normal (`yolov8_noRos.py`)  |
| ------------------- | -------------------------- | --------------------------- |
| **Framework**       | GStreamer/DeepStream       | Python/Ultralytics/ROS2     |
| **Camera**          | V4L2 (`v4l2src`)           | Pyrealsense2 SDK            |
| **Memory**          | NVMM (zero-copy GPU)       | CPU → GPU transfers         |
| **Inference**       | DeepStream `nvinfer`       | Ultralytics YOLO            |
| **Config**          | DeepStream config file     | ROS2 parameters             |
| **Tracking**        | None                       | ByteTrack + Kalman Filter   |
| **Color Detection** | None                       | HSV-based enemy detection   |
| **ROS Integration** | None                       | Full ROS2 node              |
| **Output**          | Display/fakesink           | ROS2 topics                 |
| **Complexity**      | Simple pipeline            | Complex processing logic    |
| **Performance**     | Optimized GPU pipeline     | CPU overhead for processing |
