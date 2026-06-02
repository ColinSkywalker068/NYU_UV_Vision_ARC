# Operations Runbook: CV Detection

<!-- SCOPE: Operational procedures for local development, Docker startup, deployment, restart, logs, testing, CI checks, and troubleshooting only. -->
<!-- DOC_KIND: how-to -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when you need setup, build, launch, deploy, restart, log inspection, recovery, or troubleshooting commands. -->
<!-- SKIP_WHEN: Skip when you only need static infrastructure inventory, ROS 2 node architecture, or detector implementation details. -->
<!-- PRIMARY_SOURCES: Docker/docker-compose.yml, Docker/Dockerfile.run-cv, Docker/Dockerfile.cplusplus, Docker/build_cplusplus.sh, services/uv.sh, services/start-container.sh, services/stm-stack.service, services/stm-stack.sh, services/stm-stack-entrypoint.sh, services/setup-stm-stack.sh, CPP_CONTAINER_RUNBOOK.md -->

> **Status:** Current operational guide
> **Last Updated:** 2026-05-30

## Quick Navigation

- [Infrastructure](infrastructure.md)
- [Setup Guide](../Setup/Setup.md)
- [Software Install Guide](../Install_Software/Install_Software.md)
- [Jetson Connection Guide](../Connecting_Jetson/Connecting_Jetson.md)
- [Pre-commit Guide](../Pre-commit.md)

## Agent Entry

| Signal | Value |
|--------|-------|
| Purpose | Canonical command guide for running, deploying, restarting, and diagnosing the CV Detection robot vision stack. |
| Read When | You need exact Docker, ROS 2, systemd, test, or troubleshooting commands. |
| Skip When | You need static host inventory, CI/CD inventory, or high-level architecture. |
| Canonical | Yes |
| Next Docs | [Infrastructure](infrastructure.md), [Setup Guide](../Setup/Setup.md) |
| Primary Sources | `Docker/docker-compose.yml`, `Docker/Dockerfile.run-cv`, `Docker/Dockerfile.cplusplus`, `Docker/build_cplusplus.sh`, `services/uv.sh`, `services/start-container.sh`, `services/stm-stack.service`, `services/stm-stack.sh`, `services/stm-stack-entrypoint.sh`, `services/setup-stm-stack.sh`, `CPP_CONTAINER_RUNBOOK.md` |

## 1. Overview

CV Detection is the NYU Robomaster ROS 2 Humble vision runtime. The active C++ stack builds with `colcon` and runs the `yolo_cpp` package for camera capture, YOLO/TensorRT/DeepStream armor detection, tracking, 3D processing, ballistic aim solving, and STM32 serial output.

The repository has three primary runtime paths:

| Path | Use When | Command Surface |
|------|----------|-----------------|
| `uv` wrapper with `services/start-container.sh` | Day-to-day robot and development use, especially when the host does not have ROS 2 sourced locally | `uv docker`, `uv build`, `uv run yolo_cpp basler.launch.py` |
| Competition STM service | Headless robot startup, match restarts, SSH diagnostics, and service logs | `./services/stm-stack.sh preflight`, `./services/stm-stack.sh restart`, `./services/stm-stack.sh health` |
| Docker Compose service | One-command RealSense stack startup from the `Docker/` directory | `docker compose -f Docker/docker-compose.yml up --build` |

## 2. Contacts

| Name | Email |
|------|-------|
| Kushal Mamillapalli | `km6238@nyu.edu` |
| Shubham Gupta | `shubhamgupto@gmail.com` |
| 20af02 | `ammaarfirozi@gmail.com` |

## 3. Prerequisites

| Requirement | Verified Source | Notes |
|-------------|-----------------|-------|
| Repository access | `git remote -v`, `services/setup-orin-nano.sh` | The current remote is `git@gitlab.com:nyu-robomaster/CV_Detection.git`. |
| Docker Engine and Compose plugin | `services/setup-orin-nano.sh`, `Docker/docker-compose.yml` | Required for containerized runtime paths. |
| NVIDIA runtime on robot host | `Docker/docker-compose.yml`, `services/start-container.sh` | Required for Jetson GPU production paths. |
| ROS 2 Humble environment | `services/uv.sh`, `Docker/Dockerfile.run-cv` | Usually provided by the selected Docker image. |
| Host camera and USB devices | `Docker/docker-compose.yml`, `services/start-container.sh` | RealSense and Basler access require host device mounts. |
| STM32 serial device | `yolo_cpp/launch/basler.launch.py` | Basler field stack defaults to `/dev/ttyUSB0` at `115200` baud. |
| Optional Basler Pylon SDK | `Docker/Dockerfile.cplusplus`, `CPP_CONTAINER_RUNBOOK.md` | Basler backend is enabled only when Pylon is visible at build time. |
| Optional X11 display access | `Docker/docker-compose.yml`, `services/start-container.sh` | Needed when enabling OpenCV/GUI display output. |

## 4. Environment Variables

No `.env.example` file exists in the current repository. The operational variables below are declared in Docker, service, or helper script sources.

| Variable | Required For | Default Or Example | Source |
|----------|--------------|--------------------|--------|
| `CV_BASE_IMAGE` | Docker Compose build base image override | `nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus` | `Docker/docker-compose.yml` |
| `CV_STM_IMAGE` | Competition STM service Docker image | `nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus`; use `cv-detection-cplusplus-pylon:local` after baking Pylon locally | `services/stm-stack.service`, `services/stm-stack.sh`, `Docker/bake_cplusplus_pylon_image.sh` |
| `CV_STM_REPO_ROOT` | Repository path inside the service container | `/home/ultraviolet/CV_Detection` | `services/stm-stack.service`, `services/stm-stack-entrypoint.sh` |
| `CV_STM_LAUNCH_PROFILE` | Service launch profile | `basler`; set `realsense` for RealSense + native TensorRT bring-up | `services/stm-stack-entrypoint.sh` |
| `CV_STM_TARGET_COLOR` | Default enemy color before STM or robot-id feedback arrives | `blue`; set `red` when testing red armor without STM feedback | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_USE_MODEL_COLOR_CLASSES` | Whether model class ids decide enemy color | `true`; set `false` with `CV_STM_ENABLE_COLOR_VERIFICATION=false` to accept both colors during bring-up | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_ENABLE_COLOR_VERIFICATION` | Whether HSV color checks run when model color classes are disabled | `false`; keep disabled while trusting model color classes | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_SAVE_TARGET_FRAMES` | Optional target-frame JPEG capture for visual checks and heatmaps | `false`; set `true` to save up to `CV_STM_TARGET_FRAME_SAVE_MAX` images under `pictures/` next to `vision_logs/` | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_TARGET_FRAME_SAVE_DIR` | Optional override for target-frame output directory | Empty by default; unset saves under `pictures/` next to `vision_logs/` | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_TARGET_FRAME_SAVE_MAX` | Maximum target-frame images saved per run | `100` | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_TARGET_FRAME_JPEG_QUALITY` | JPEG quality for optional target-frame capture | `85` | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_CAMERA_TYPE` | Lower-level camera backend override for the Basler-profile launch only | Usually unset; prefer `CV_STM_LAUNCH_PROFILE` | `services/stm-stack-entrypoint.sh` |
| `CV_STM_MODEL_PATH` | Competition model or engine path override | Empty by default; `yolo_cpp/config/default_model_path.txt` supplies the robot-local fallback | `services/stm-stack-entrypoint.sh` |
| `CV_STM_SERIAL_PORT` | STM32 serial device for competition startup | `/dev/ttyUSB0` | `services/stm-stack-entrypoint.sh` |
| `CV_STM_REQUIRE_SERIAL` | Whether preflight must fail when the serial path is absent | Defaults to `true` for `basler`, `false` for `realsense` | `services/stm-stack-entrypoint.sh` |
| `CV_STM_EXTRA_LAUNCH_ARGS` | Optional extra selected-profile launch arguments | Empty by default | `services/stm-stack-entrypoint.sh` |
| `CV_STM_BASLER_PIXEL_FORMAT` | Basler source pixel format for the DeepStream/Pylon path | `bayer_rggb8`; alternatives include `yuy2`, `bgr8`, `mono8` | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_BASLER_FPS` | Basler requested caps/frame-rate fallback | `160` for the field Basler launch; lower to `90` for a conservative no-drop cap | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_BASLER_SOURCE_WIDTH` | Basler source width for full-sensor capture | `1920` for `a2A1920-160ucBAS` | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_BASLER_SOURCE_HEIGHT` | Basler source height for full-sensor capture | `1200` for `a2A1920-160ucBAS` | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_DEEPSTREAM_PRE_MUX_RESIZE` | Resize full-FOV Basler frames before `nvstreammux` padding | `true`; set `false` only for A/B testing the older full-resolution path | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_DEEPSTREAM_PRE_MUX_WIDTH` / `CV_STM_DEEPSTREAM_PRE_MUX_HEIGHT` | Optional pre-mux resize bounding box | `0`; unset/zero uses the model dimensions as the resize box | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_DEEPSTREAM_PROFILE_STAGES` | Log per-stage DeepStream pad timing for conversion diagnosis | `true` for current Basler profiling; set `false` to quiet service logs | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `CV_STM_DEEPSTREAM_PROFILE_STAGE_INTERVAL` | Frames between stage-profile summaries | `300` | `services/stm-stack-entrypoint.sh`, `yolo_cpp/launch/basler.launch.py` |
| `DISPLAY` | X11 GUI forwarding | Host `DISPLAY` value | `Docker/docker-compose.yml`, `services/start-container.sh` |
| `QT_X11_NO_MITSHM` | Qt/X11 compatibility | `1` | `Docker/docker-compose.yml` |
| `V4L2_DEVICES` | Camera device forwarding | Auto-assembled `/dev/video0` through `/dev/video9` flags on Jetson | `Docker/docker-compose.yml`, `services/start-container.sh` |
| `NUMBA_THREADING_LAYER` | Python legacy runtime compatibility | `omp` | `Docker/docker-compose.yml` |
| `XAUTHORITY` | X11 auth inside container | `/tmp/.docker.xauth` | `Docker/docker-compose.yml`, `services/start-container.sh` |
| `IMAGE` | `services/run-cplusplus.sh` image override | `nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus` | `services/run-cplusplus.sh` |
| `UV_CONTAINER_NAME` | Stable or custom Docker container name | Competition service uses `cv-stm-stack`; otherwise unset uses `uv-` plus sanitized image name | `services/start-container.sh`, `services/stm-stack.service` |
| `HUGGINGFACE_TOKEN` | Optional model/token pass-through | Host value when present | `services/start-container.sh` |
| `RMW_IMPLEMENTATION` | ROS 2 middleware in Jetson container path | `rmw_fastrtps_cpp` | `services/start-container.sh` |
| `PYLON_ROOT` | Basler Pylon install root | `/opt/pylon` | `Docker/Dockerfile.cplusplus`, `CPP_CONTAINER_RUNBOOK.md` |
| `PYLON_PACKAGE` | Build-time Basler SDK package under `Docker/pylon/` | Unset by default | `Docker/Dockerfile.cplusplus`, `Docker/build_cplusplus.sh` |
| `PYLON_GSTREAMER_PACKAGE` | Build-time Basler GStreamer package under `Docker/pylon/` | Unset by default | `Docker/Dockerfile.cplusplus`, `Docker/build_cplusplus.sh` |
| `GST_PLUGIN_PATH` | DeepStream, GStreamer, and Pylon plugin discovery | DeepStream/Pylon plugin paths | `Docker/Dockerfile.cplusplus`, `services/start-container.sh` |
| `GST_PLUGIN_PATH_1_0` | GStreamer 1.0 plugin discovery | Same path set as `GST_PLUGIN_PATH` | `Docker/Dockerfile.cplusplus`, `services/start-container.sh` |
| `LD_LIBRARY_PATH` | DeepStream, CUDA, NVIDIA, and Pylon library discovery | DeepStream, Tegra, CUDA, Pylon library paths | `Docker/Dockerfile.cplusplus`, `services/start-container.sh` |
| `GST_REGISTRY` | Runtime GStreamer registry cache | `/tmp/gst-registry-deepstream.bin` | `services/start-container.sh` |

## 5. Local Development Setup

### 5.1 Clone And Enter The Repository

```bash
git clone git@gitlab.com:nyu-robomaster/CV_Detection.git
cd CV_Detection
```

### 5.2 Start A Development Container

Use the `uv` wrapper when it is installed as a shell command:

```bash
uv docker
```

Use the repository script directly when `uv` is not installed on the host:

```bash
./services/run-cplusplus.sh
```

For X11 GUI/debug windows on Linux, allow the container root user before starting the container:

```bash
xhost +si:localuser:root
```

### 5.3 Build The Active C++ Stack

Inside a ROS 2 environment or inside the container:

```bash
uv build yolo_cpp
```

The equivalent direct `colcon` build for the active stack is:

```bash
colcon build --packages-select cv_auto_aim_interfaces yolo_cpp --symlink-install --cmake-args -DBUILD_TESTING=OFF
```

### 5.4 Launch Runtime Stacks

Basler camera-to-STM32 field stack:

```bash
uv run yolo_cpp basler.launch.py show:=false
```

RealSense-only detector, tracker, processor, and solver stack:

```bash
uv run yolo_cpp realsense.launch.py show:=false
```

## 6. Docker Compose Runtime

The Compose service is `cv-container` in `Docker/docker-compose.yml`. It builds `Docker/Dockerfile.run-cv`, uses host networking, uses the NVIDIA runtime, mounts robot devices, and launches the RealSense stack through the Dockerfile entrypoint.

Start or rebuild the Compose runtime:

```bash
docker compose -f Docker/docker-compose.yml up --build
```

Watch logs:

```bash
docker compose -f Docker/docker-compose.yml logs -f cv-container
```

Stop the Compose runtime:

```bash
docker compose -f Docker/docker-compose.yml down
```

## 7. Robot Systemd Deployment

The declared boot service is `services/stm-stack.service`. It runs from `/home/ultraviolet/CV_Detection`, requires `docker.service`, removes the previous `cv-stm-stack` container, and launches the selected `CV_STM_LAUNCH_PROFILE` through `services/stm-stack-entrypoint.sh`. The default profile is the full Basler + DeepStream + STM32 `basler.launch.py` stack; `CV_STM_LAUNCH_PROFILE=realsense` launches the RealSense + native TensorRT bring-up stack without requiring serial by default.

The service uses the C++ image `nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus` and reads optional overrides from `/etc/cv-detection/stm-stack.env`. Startup is intentionally prebuilt: it fails when `install/setup.bash` is missing instead of running `uv build` during match bring-up.

Install or refresh the service on the robot host without starting it:

```bash
./services/stm-stack.sh install
```

Run the competition preflight before startup:

```bash
./services/stm-stack.sh preflight
```

Check status:

```bash
./services/stm-stack.sh status
```

Restart the service:

```bash
./services/stm-stack.sh restart
```

Follow live logs:

```bash
./services/stm-stack.sh logs
```

Check the live ROS graph after startup:

```bash
./services/stm-stack.sh health
```

Enter the running service container:

```bash
./services/stm-stack.sh shell
```

Stop the service:

```bash
./services/stm-stack.sh stop
```

## 8. Basler And DeepStream Builds

Basler support is compiled only when Pylon is visible during `colcon build`. The C++ Dockerfile supports build-time packages copied under `Docker/pylon/`.

If an existing `uv docker` C++ container can load `pylonsrc` but the clean service image cannot, bake the working Pylon install into a stable local service image:

```bash
cd Docker
UPDATE_STM_ENV=1 ./bake_cplusplus_pylon_image.sh
cd ..
CV_STM_REQUIRE_SERIAL=false ./services/stm-stack.sh preflight
```

This creates `cv-detection-cplusplus-pylon:local`, verifies `gst-inspect-1.0 pylonsrc`, and writes `CV_STM_IMAGE=cv-detection-cplusplus-pylon:local` to `/etc/cv-detection/stm-stack.env`.

Build the C++ DeepStream image with the default tags:

```bash
cd Docker
./build_cplusplus.sh
```

To build with official Basler package archives, copy the SDK and GStreamer package archives under `Docker/pylon/`, then set `PYLON_PACKAGE` and `PYLON_GSTREAMER_PACKAGE` before invoking `Docker/build_cplusplus.sh`. The repository declares the build arguments, but does not include Basler package files.

After adding or mounting Pylon, force a clean ROS workspace rebuild so CMake re-detects the SDK:

```bash
uv clean
uv build yolo_cpp
```

For Basler high-FPS bring-up on `a2A1920-160ucBAS`, keep `basler_use_full_sensor:=true` and use the default raw Bayer source caps: `basler_pixel_format:=bayer_rggb8 basler_source_width:=1920 basler_source_height:=1200 basler_fps:=160 basler_uncapped_fps:=true`. Use the default `pre_mux_resize:=true` or `CV_STM_DEEPSTREAM_PRE_MUX_RESIZE=true` when you need full FOV with fewer pixels entering `nvstreammux`; the default 640x640 model box resizes 1920x1200 to 640x400 before streammux padding. The default `bayer_conversion_backend:=cuda` / `CV_STM_DEEPSTREAM_BAYER_CONVERSION_BACKEND=cuda` uses the in-process CUDA Bayer resize path when the build supports it, and logs a fallback to CPU `bayer2rgb` otherwise. `show:=true` uses a downscaled preview frame and is throttled by `preview_max_fps` / `CV_STM_PREVIEW_MAX_FPS`; raw image publishing, HSV verification, and target-frame saving still need full-resolution BGR. The startup logs should show the requested and negotiated GStreamer caps plus the camera-reported resulting frame rate. If `basler_pixel_format:=bgr8` is used at full resolution, USB3 bandwidth can reasonably limit capture near 60 FPS; that is not evidence of a Jetson hard cap.

On the observed `a2A1920-160ucBAS` / serial `41923000` setup, direct Pylon probing reported `Binning*`, `Decimation*`, and `Scaling*` GenICam nodes as absent. Treat `Width` and `Height` as ROI/crop controls there, not full-FOV downsampling controls.

To separate Jetson model throughput from camera ingest, run the FPS isolation harness on the Jetson:

```bash
./services/fps-benchmark.sh --mode all --duration 10 --warmup 2
```

Read the three modes as different ceilings: `inference` is synthetic-frame TensorRT throughput using the STM engine, `camera` is capture-only throughput through the active camera backend, and `deepstream` is the Basler + DeepStream result cadence.

To restart the headless STM service after a rebuild, run:

```bash
uv build
./services/stm-stack.sh restart
```

Stage profiling is on by default for the current Basler bring-up. To quiet the service after profiling, set:

```bash
CV_STM_DEEPSTREAM_PROFILE_STAGES=false
```

## 9. Testing And Formatting

Run C++ package tests:

```bash
colcon test --packages-select yolo_cpp
```

Run the repository formatter/lint hook set:

```bash
pre-commit run --all-files
```

The GitLab CI configuration also runs `pre-commit run --all-files` and commits formatting changes back to the branch when changes are produced.

## 10. Operational Checks

Check running containers:

```bash
docker ps
```

Check ROS graph after sourcing the workspace in a container shell:

```bash
source /opt/ros/humble/setup.bash
source /home/ultraviolet/CV_Detection/install/setup.bash
ros2 node list
ros2 topic list
```

Check image stream rate when the camera stack is running:

```bash
ros2 topic hz /image_raw
```

Check STM32 serial device availability on the host or inside the privileged container:

```bash
ls -l /dev/ttyUSB0
```

Check DeepStream/GStreamer plugin availability inside the C++ container:

```bash
gst-inspect-1.0 nvstreammux
gst-inspect-1.0 nvinfer
gst-inspect-1.0 pylonsrc
```

Check the competition service ROS graph from the host:

```bash
./services/stm-stack.sh health
```

## 11. Troubleshooting

### Container Does Not Start With GPU Runtime

Diagnosis:

```bash
docker info
docker ps
```

Resolution:

1. Confirm the host is a Jetson-class Linux target or has NVIDIA container runtime configured.
2. Confirm Docker is running with `sudo systemctl status docker`.
3. Re-run the runtime through `uv docker` or `docker compose -f Docker/docker-compose.yml up --build`.

### Camera Device Is Missing

Diagnosis:

```bash
ls /dev/video*
lsusb
docker ps
```

Resolution:

1. Reconnect the camera and confirm it appears on the host.
2. Use the `uv` runtime on Jetson so `services/start-container.sh` discovers `/dev/video0` through `/dev/video9`.
3. For Compose, confirm the expected camera is available as `/dev/video0`, because `Docker/docker-compose.yml` mounts that device explicitly.

### Basler Backend Is Disabled

Diagnosis:

```bash
ls /opt/pylon
gst-inspect-1.0 pylonsrc
```

Resolution:

1. For the DeepStream Basler launch, `pylonsrc` is the runtime requirement. `/opt/pylon` is the usual SDK path and is mounted when present, but preflight can continue without it if `gst-inspect-1.0 pylonsrc` succeeds.
2. If `pylonsrc` is missing, install or mount Pylon at `/opt/pylon`, or build the C++ image with `PYLON_PACKAGE` and `PYLON_GSTREAMER_PACKAGE` under `Docker/pylon/`.
3. Run `uv clean`.
4. Run `uv build yolo_cpp` so CMake rechecks Pylon.

### DeepStream Or GStreamer Plugin Is Missing

Diagnosis:

```bash
echo "$GST_PLUGIN_PATH"
echo "$LD_LIBRARY_PATH"
gst-inspect-1.0 nvinfer
```

Resolution:

1. Start the C++ image through `services/start-container.sh` or `services/run-cplusplus.sh`; those scripts populate the DeepStream and Pylon plugin paths for C++ images.
2. Rebuild `Docker/Dockerfile.cplusplus` with `Docker/build_cplusplus.sh` if the image lacks DeepStream 6.4.

### STM32 Serial Output Is Not Connected

Diagnosis:

```bash
ls -l /dev/ttyUSB0
ros2 node list
```

Resolution:

1. Confirm the STM32 is attached as `/dev/ttyUSB0`, or launch `basler.launch.py` with a different `serial_port:=` value.
2. Confirm the full stack, not the RealSense-only stack, is running.
3. Check `./services/stm-stack.sh logs` when running through systemd.

The serial bridge is implemented to stay alive when the serial port cannot be opened, so a running ROS graph does not prove TX is connected.

### ROS Package Is Not Found

Diagnosis:

```bash
source /opt/ros/humble/setup.bash
source /home/ultraviolet/CV_Detection/install/setup.bash
ros2 pkg list | grep yolo_cpp
```

Resolution:

1. Run `uv build yolo_cpp`.
2. Source `/home/ultraviolet/CV_Detection/install/setup.bash` in the shell that launches ROS commands.

### Existing Container Is Reused Unexpectedly

Diagnosis:

```bash
docker ps -a
```

Resolution:

1. Set `UV_CONTAINER_NAME` to create a separate container for the session.
2. Stop the old container with `docker stop` when it is no longer needed.
3. Remove the old container with `docker rm` only after confirming it is not the active robot runtime.

## 12. Recovery Procedures

| Situation | Procedure |
|-----------|-----------|
| Systemd stack is unhealthy | Run `./services/stm-stack.sh restart`, then follow `./services/stm-stack.sh logs` and `./services/stm-stack.sh health`. |
| Compose runtime is unhealthy | Run `docker compose -f Docker/docker-compose.yml down`, then `docker compose -f Docker/docker-compose.yml up --build`. |
| ROS workspace overlay is stale | Run `uv clean`, `uv build yolo_cpp`, then source `install/setup.bash`. |
| Camera was replugged | Restart the active container or systemd service so device discovery and mounts are refreshed. |
| Basler SDK was added after build | Run `uv clean` and rebuild `yolo_cpp` so CMake enables Basler support. |

## 13. Service Dependencies

| Runtime | Depends On | Declared Source |
|---------|------------|-----------------|
| `cv-container` Compose service | Docker Engine, NVIDIA runtime, host network, `/dev/video0`, `/dev/bus/usb`, `/tmp/argus_socket`, `/dev`, `/home/ultraviolet` | `Docker/docker-compose.yml` |
| `stm-stack.service` | `docker.service`, `/home/ultraviolet/CV_Detection`, `services/start-container.sh`, `services/stm-stack-entrypoint.sh`, C++ DeepStream image, STM32 serial device for Basler TX | `services/stm-stack.service` |
| `yolo_cpp` Basler field stack | `cv_auto_aim_interfaces`, model path from `CV_STM_MODEL_PATH` or `yolo_cpp/config/default_model_path.txt`, Basler camera device, optional `/dev/ttyUSB0` | `yolo_cpp/launch/basler.launch.py` |
| C++ DeepStream/Basler image | DeepStream 6.4, GStreamer plugins, optional Pylon packages under `Docker/pylon/` | `Docker/Dockerfile.cplusplus`, `Docker/build_cplusplus.sh` |

## 14. Port Mapping

| Runtime | Declared Port Mappings | Network Notes |
|---------|------------------------|---------------|
| `cv-container` | None | `Docker/docker-compose.yml` uses `network_mode: host`. |
| `services/start-container.sh` containers | None | The script uses `docker run --network host` on Jetson, Linux, macOS, and fallback paths. |

## Maintenance

**Update Triggers:**
- New Docker images, Dockerfiles, Compose services, device mounts, or host network behavior
- Changes to `services/uv.sh`, `services/start-container.sh`, `services/stm-stack.service`, or setup scripts
- Changes to ROS launch files, default model paths, serial defaults, or camera defaults
- New CI behavior, formatting commands, or release automation
- New troubleshooting findings from robot operation

**Verification:**
```bash
docker compose -f Docker/docker-compose.yml config
colcon build --packages-select cv_auto_aim_interfaces yolo_cpp --symlink-install --cmake-args -DBUILD_TESTING=OFF
colcon test --packages-select yolo_cpp
```

---
