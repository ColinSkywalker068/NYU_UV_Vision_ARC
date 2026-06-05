[![pipeline status](https://gitlab.com/nyu-robomaster/CV_Detection/badges/main/pipeline.svg)](https://gitlab.com/nyu-robomaster/CV_Detection/-/commits/main)

# CV Detection

NYU Robomaster ROS 2 C++ auto-aim vision pipeline for camera capture, YOLO/TensorRT/DeepStream armor detection, temporal tracking, 3D target processing, ballistic aim solving, and STM32 serial output.

## Active Runtime

The 2026 runtime is centered on:

- `yolo_cpp/`: C++ ROS 2 nodes, launch files, DeepStream config, tests, package docs, and active model assets.
- `cv_auto_aim_interfaces/`: custom ROS 2 messages shared by detector, tracker, processor, solver, and STM bridge.
- `services/`: Jetson/container helper scripts, STM stack service, preflight, health, and benchmark tools.
- `Docker/`: active container images and Docker bring-up notes.
- `docs/` and `yolo_cpp/docs/`: current project, architecture, runbook, and package documentation.

## Quick Start

On the robot, use the `cv` operator command for the headless Basler + DeepStream + STM32 service.
Install or refresh the shortcut once:

```bash
sudo ln -sf /home/ultraviolet/CV_Detection/services/stm-stack.sh /usr/local/bin/cv
```

Run the competition stack:

```bash
cv install
cv preflight
cv restart
cv logs
```

Common operator commands:

```bash
cv status
cv health
cv preview-off
cv basler-defaults
cv model-orin
cv model-agx
```

If the `cv` symlink is not installed yet, use the helper directly from the repo root:

```bash
./services/stm-stack.sh install
./services/stm-stack.sh preflight
./services/stm-stack.sh restart
./services/stm-stack.sh logs
```

For development shells and manual ROS launches, use the `uv` wrapper from the repo root:

```bash
uv docker
uv build yolo_cpp
uv run yolo_cpp basler.launch.py
uv run yolo_cpp realsense.launch.py
```

The robot-local fallback model path is configured at:

```text
yolo_cpp/config/default_model_path.txt
```

Override it with `CV_STM_MODEL_PATH` or a launch `model_path:=...` argument. The `model-nano` shortcut remains available as a backward-compatible alias for `model-orin`.

## Documentation

- [Documentation map](docs/README.md)
- [C++ runtime overview](yolo_cpp/docs/overview.md)
- [Pipeline stages](yolo_cpp/docs/pipeline.md)
- [ROS nodes](yolo_cpp/docs/nodes.md)
- [Launch and config](yolo_cpp/docs/launch-and-config.md)
- [Container and bring-up runbook](CPP_CONTAINER_RUNBOOK.md)
- [Project runbook](docs/project/runbook.md)

## Legacy Archive

Historical Python tooling, including the former `yolov8_ros/` package, Mac/CoreML prototypes, old setup notes, stale model variants, and retired Docker files are archived under `legacy/`. Treat them as reference material unless a task explicitly targets legacy behavior.
