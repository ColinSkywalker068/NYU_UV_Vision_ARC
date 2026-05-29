# Tech Stack: CV Detection

<!-- SCOPE: Technology inventory, versions where declared, dependency roles, and stack rationale for the current repository. -->
<!-- DOC_KIND: reference -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when adding dependencies, changing build/runtime images, or validating environment assumptions. -->
<!-- SKIP_WHEN: Skip when you only need architecture flow, requirements, or operator procedures. -->
<!-- PRIMARY_SOURCES: yolo_cpp/package.xml, yolo_cpp/CMakeLists.txt, cv_auto_aim_interfaces/package.xml, Docker/Dockerfile.cplusplus, Docker/docker-compose.yml, services/uv.sh, services/stm-stack.service, services/stm-stack.sh, .pre-commit-config.yaml -->

## Quick Navigation

| Need | Read |
|------|------|
| Architecture | [architecture.md](architecture.md) |
| Requirements | [requirements.md](requirements.md) |
| Runtime operation | [runbook.md](runbook.md) |
| Infrastructure | [infrastructure.md](infrastructure.md) |

## Agent Entry

| Signal | Value |
|--------|-------|
| Purpose | Lists the active languages, frameworks, runtime dependencies, build tools, images, and CI tools. |
| Read When | You need to add or change technology choices. |
| Skip When | You only need node-level runtime behavior. |
| Canonical | Yes |
| Primary Sources | `package.xml`, `CMakeLists.txt`, Docker files, service scripts, CI config |

## Stack Summary

| Layer | Technology | Declared Version or Source | Role |
|-------|------------|----------------------------|------|
| Robotics runtime | ROS 2 Humble | Docker image tags and `/opt/ros/humble` setup in [uv.sh](../../services/uv.sh) | Node runtime, launch, topics, messages |
| Active language | C++ | C++17 in [yolo_cpp/CMakeLists.txt](../../yolo_cpp/CMakeLists.txt) | Low-latency runtime path |
| Support language | Python | Python 3.9+ pre-commit target; legacy scripts under `legacy/` and `mac/` | Legacy, training, setup, launch files |
| Build system | CMake and `ament_cmake` | CMake 3.16 for `yolo_cpp`; CMake 3.8 for `cv_auto_aim_interfaces` | ROS package build |
| Workspace tool | `colcon` | Used by [uv.sh](../../services/uv.sh) | Package build and test orchestration |
| Interface generation | `rosidl_default_generators` | [cv_auto_aim_interfaces/package.xml](../../cv_auto_aim_interfaces/package.xml) | Custom ROS message generation |
| Containers | Docker and Docker Compose | [Docker/docker-compose.yml](../../Docker/docker-compose.yml) | Robot/container runtime |
| GPU runtime | NVIDIA Docker runtime | `runtime: nvidia` in compose | Jetson GPU access |

## ROS Packages

| Package | Type | Purpose | Source |
|---------|------|---------|--------|
| `yolo_cpp` | `ament_cmake` C++ package | Camera, detector, tracker, processor, solver, STM bridge, and legacy combined nodes. | [yolo_cpp/package.xml](../../yolo_cpp/package.xml) |
| `cv_auto_aim_interfaces` | ROS interface package | Custom messages for detections, tracks, targets, spin observer state, and aim solutions. | [cv_auto_aim_interfaces/package.xml](../../cv_auto_aim_interfaces/package.xml) |

## ROS Dependencies

| Dependency | Used For | Declared In |
|------------|----------|-------------|
| `rclcpp` | C++ ROS nodes | [yolo_cpp/package.xml](../../yolo_cpp/package.xml) |
| `rclcpp_components` | Component registration and composable nodes | [yolo_cpp/package.xml](../../yolo_cpp/package.xml) |
| `sensor_msgs` | Image and camera info messages | [yolo_cpp/package.xml](../../yolo_cpp/package.xml) |
| `std_msgs` | Legacy controller topics and simple feedback | [yolo_cpp/package.xml](../../yolo_cpp/package.xml) |
| `geometry_msgs` | Pose, point, vector, and armor corner fields | [cv_auto_aim_interfaces/package.xml](../../cv_auto_aim_interfaces/package.xml) |
| `tf2`, `tf2_ros`, `tf2_geometry_msgs` | Target-frame transform support | [yolo_cpp/package.xml](../../yolo_cpp/package.xml) |
| `visualization_msgs` | Processor marker output | [yolo_cpp/package.xml](../../yolo_cpp/package.xml) |
| `ament_cmake_gtest` | C++ unit tests | [yolo_cpp/CMakeLists.txt](../../yolo_cpp/CMakeLists.txt) |

## Vision and Hardware Dependencies

| Technology | Detection Mechanism | Role |
|------------|---------------------|------|
| OpenCV | Required `find_package(OpenCV REQUIRED)` | Image processing, math, and tests |
| GStreamer | `pkg_check_modules` for GStreamer packages | Camera/media pipeline support |
| Intel RealSense | Header and library lookup for `librealsense2` | RealSense camera backend |
| TensorRT | `NvInfer.h`, CUDA runtime headers, `nvinfer`, and `cudart` lookup | Native TensorRT detector backend |
| NVIDIA DeepStream | DeepStream include and metadata libraries | DeepStream detector backend |
| Basler Pylon | Pylon headers and libraries under `/opt/pylon` or `PYLON_ROOT` | Basler camera backend |
| STM32 serial | `stm32_serial_bridge` executable and SeaSky protocol helpers | Controller boundary |

## Build and Test Tooling

| Tool | Role | Command or Config |
|------|------|-------------------|
| `uv` shell wrapper | Starts containers, builds packages, runs launch files and nodes. | [services/uv.sh](../../services/uv.sh) |
| `colcon build` | Builds ROS workspace packages. | `uv build yolo_cpp` wraps package selection for active stack. |
| `colcon test` | Runs ROS package tests when available. | `colcon test --packages-select yolo_cpp` |
| GoogleTest via `ament_cmake_gtest` | Tests detection postprocess and tracking core. | [yolo_cpp/test](../../yolo_cpp/test/test_tracking_core.cpp) |
| pre-commit | Python formatting and general checks. | [.pre-commit-config.yaml](../../.pre-commit-config.yaml) |
| Commitizen | Conventional commit helper. | [package.json](../../package.json) |

## Container Stack

| File | Purpose |
|------|---------|
| [Docker/docker-compose.yml](../../Docker/docker-compose.yml) | Defines the `cv-container` service with host networking, NVIDIA runtime, camera/USB/device mounts, and robot host volumes. |
| [Docker/Dockerfile.run-cv](../../Docker/Dockerfile.run-cv) | Runtime entrypoint image that runs `uv clean`, `uv build`, and `uv run yolo_cpp realsense.launch.py show:=false`. |
| [Docker/Dockerfile.cplusplus](../../Docker/Dockerfile.cplusplus) | C++ DeepStream image layer with ROS, GStreamer, DeepStream 6.4, optional Pylon, CUDA/TensorRT paths, and runtime library setup. |
| [services/start-container.sh](../../services/start-container.sh) | Cross-host Docker launcher for Jetson, Linux, macOS, and fallback Windows paths. |
| [services/stm-stack.service](../../services/stm-stack.service) | Systemd unit for headless full-stack launch. |
| [services/stm-stack.sh](../../services/stm-stack.sh) | Operator CLI for installing, restarting, preflighting, logging, health-checking, and entering the competition service container. |
| [services/stm-stack-entrypoint.sh](../../services/stm-stack-entrypoint.sh) | Container-side competition startup entrypoint that validates prebuilt ROS, model, DeepStream/Pylon, and serial prerequisites before launching. |

## CI and Quality Gates

| System | Config | Current Behavior |
|--------|--------|------------------|
| GitHub Actions | [.github/workflows/main.yml](../../.github/workflows/main.yml) | On push to `main`, bumps a tag and creates a release. |
| GitLab CI | [.gitlab-ci.yml](../../.gitlab-ci.yml) | Runs pre-commit in a CI image and auto-commits formatting changes when present. |
| pre-commit | [.pre-commit-config.yaml](../../.pre-commit-config.yaml) | Black, pyupgrade, commitlint, YAML/JSON checks, debug statement checks, merge conflict checks, and a local mock assertion grep. |

## Rationale

| Choice | Rationale |
|--------|-----------|
| ROS 2 Humble | Matches the robot runtime style and provides launch, node, topic, and interface tooling. |
| C++17 hot path | Keeps camera, inference, tracking, solving, and serial work in a low-latency compiled runtime. |
| Custom message package | Keeps stage contracts explicit and reusable across nodes. |
| Optional dependency detection | Lets the same source tree build on machines without every camera or GPU SDK installed. |
| Docker/NVIDIA runtime | Encapsulates Jetson GPU and media dependencies that are difficult to reproduce on a host directly. |
| `uv` wrapper | Gives operators one command surface for container entry, build, launch, topics, services, and image updates. |

## Maintenance

**Update Triggers:**
- New ROS package, message dependency, executable, or test target
- Docker base image or compose service change
- Camera, inference, GPU, or serial dependency change
- CI or pre-commit configuration change

**Verification:**
- [ ] Dependency names match package manifests and CMake
- [ ] Docker image tags match compose and service scripts
- [ ] Commands match `services/uv.sh`
- [ ] Optional dependency behavior matches `HAVE_*` checks in CMake

**Last Updated:** 2026-05-15
