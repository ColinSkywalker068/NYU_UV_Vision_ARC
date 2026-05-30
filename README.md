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

Use the `uv` wrapper from the repo root.

```bash
uv docker
uv build yolo_cpp
```

Launch the full Basler + DeepStream + STM32 field stack:

```bash
uv run yolo_cpp basler.launch.py
```

Launch the RealSense bring-up stack:

```bash
uv run yolo_cpp realsense.launch.py
```

For the headless STM service:

```bash
./services/stm-stack.sh install
./services/stm-stack.sh preflight
./services/stm-stack.sh restart
./services/stm-stack.sh logs
```

The default runtime engine lives at:

```text
yolo_cpp/models/best-roboflow-nitish-obb.onnx_b1_gpu0_fp16.engine
```

## Documentation

- [Documentation map](docs/README.md)
- [C++ runtime overview](yolo_cpp/docs/overview.md)
- [Pipeline stages](yolo_cpp/docs/pipeline.md)
- [ROS nodes](yolo_cpp/docs/nodes.md)
- [Launch and config](yolo_cpp/docs/launch-and-config.md)
- [Container and bring-up runbook](CPP_CONTAINER_RUNBOOK.md)
- [Project runbook](docs/project/runbook.md)

## Legacy Archive

Historical Python tooling, Mac/CoreML prototypes, old setup notes, stale model variants, and retired Docker files are archived under `legacy/`. Treat them as reference material unless a task explicitly targets legacy behavior.
