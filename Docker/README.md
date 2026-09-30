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

## Manual Run With Custom Model

You can also launch the active stack with custom parameters:

```bash
uv docker
uv build yolo_cpp
uv run yolo_cpp realsense.launch.py model_path:=/path/to/model.engine show:=gui
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
CV_BASE_IMAGE=nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon docker compose up --build
```

## Basler Pylon C++ Image

If `uv docker` can load `pylonsrc` but the clean service image cannot, bake the working container's Pylon install into a stable local image:

```bash
cd Docker
UPDATE_STM_ENV=1 ./bake_cplusplus_pylon_image.sh
```

That creates `cv-detection-cplusplus-pylon:local`, verifies `gst-inspect-1.0 pylonsrc`, and updates `services/stm-stack.env` when `UPDATE_STM_ENV=1`. After validation, tag and push it as the runtime C++ image, for example `nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon`.

For the fully reproducible vendor-package path, copy the Basler SDK and GStreamer package archives under `Docker/pylon/`, then build with `PYLON_PACKAGE` and `PYLON_GSTREAMER_PACKAGE` set before running `./build_cplusplus.sh`.

## JetPack 6.0 GA compatibility

`Dockerfile.cplusplus` defaults to DeepStream 7.0 for L4T 36.3 / JetPack
6.0 GA (CUDA 12.2, TensorRT 8.6). DeepStream 6.4 targets the earlier
JetPack 6.0 developer preview. Match the host release before building;
do not reuse an engine from a different TensorRT/GPU configuration without
loading it successfully on the deployment Jetson.

The version can be overridden together with `DEEPSTREAM_DEB_VERSION` for
a deliberately supported older host. Basler archives remain local under
`Docker/pylon/`; do not commit vendor archives. Validate `pylonsrc`,
`nvinfer`, `nvstreammux`, and `nvvideoconvert` with the NVIDIA runtime.

If the team C++ image is already cached, the smaller upgrade recipe is:

```bash
cd Docker
DOCKERFILE=Dockerfile.jetpack60-ga ./build_cplusplus.sh \
  nyuultraviolet/cv-dockerfile \
  jp60-pytorch21-humble-tensorrt-cplusplus orin2-ds70-base
```

This keeps the cached ROS/CUDA/TensorRT toolchain and replaces DeepStream 6.4
with 7.0. To include Basler, pass the SDK `.deb` extracted from its official
ARM64 Debian bundle as argument 4 and the matching Ubuntu 22.04 ARM64
GStreamer plugin `.deb` as argument 5; both files go in `Docker/pylon/`.
An image built without those arguments is only a dependency/build image,
not a camera-ready runtime. The service preflight rejects missing `pylonsrc`.

### Reuse the team's already-baked Basler SDK

The team camera image contains large archived installers. To reuse only its
Basler SDK/plugin layer on a Jetson with the GA dependency image built above:

```bash
python3 Docker/bake_team_pylon_layer.py \
  --base-image nyuultraviolet/cv-dockerfile:orin2-ds70-base \
  --target-image cv-detection-orin2:jp60-ds70-pylon
```

The helper pins and verifies the published layer checksum, restricts extracted
paths to the Basler SDK and its plugin libraries, builds a local image, and
checks `pylonsrc` using the NVIDIA runtime. It does not mount camera or serial
devices. An existing downloaded layer can be supplied with `--archive` for
offline reuse. This reuses the team's packaged SDK; a new vendor SDK version
should use the official Basler installer path and a matching plugin instead.

## Ramon: JetPack 7.2 / DeepStream 9.1

Ramon currently runs Ubuntu 24.04, L4T 39.2.1, CUDA 13.2 and TensorRT 10.16.
Use the separate ROS Jazzy recipe; the JetPack 6 / Humble image remains the
default for Orin2. Build on the Jetson:

```bash
docker build --network host -f Docker/Dockerfile.jetpack72 \
  -t cv-detection-ramon:jp72-ds91-jazzy-base .
python3 Docker/bake_team_pylon_layer.py \
  --base-image cv-detection-ramon:jp72-ds91-jazzy-base \
  --target-image cv-detection-ramon:jp72-ds91-jazzy-pylon
```

The Basler bake verifies the pinned team SDK checksum and checks `pylonsrc`.
Set `CV_STM_IMAGE=cv-detection-ramon:jp72-ds91-jazzy-pylon` and
`CV_STM_ROS_DISTRO=jazzy` in `services/stm-stack.env`. Set
`CV_STM_REPO_ROOT` to the actual checkout, rebuild the workspace in this image,
and keep both TX and fire recommendation flags false during validation. Build
with `--cmake-args -DYOLO_CPP_ENABLE_NATIVE_TENSORRT=OFF` on this runtime: the
optional native backend uses TensorRT 8 APIs; Ramon uses DeepStream 9.1 for
inference with native fallback disabled. CUDA camera preprocessing remains
enabled.

Rebuild a TensorRT engine from the tracked ONNX **inside the selected runtime image**.
The official DeepStream image has TensorRT 10.16.1 even when the host has 10.16.2;
serialized engines must match the container runtime. Use its bundled `trtexec`. Engines from
TensorRT 8 / JetPack 6 cannot be carried over to TensorRT 10. Keep the ONNX class
metadata embedded in the new engine so the detector uses the correct color mapping.
After `trtexec` builds a raw plan, wrap its ONNX metadata with:

```bash
python3 yolo_cpp/tools/wrap_trtexec_engine.py \
  --onnx yolo_cpp/models/latest_best.onnx \
  --plan /home/ultraviolet/cv-models/ramon-jp72-plan.engine \
  --output /home/ultraviolet/cv-models/ramon-jp72.engine
```

Do not overwrite a machine's tuned env file when updating source code.
