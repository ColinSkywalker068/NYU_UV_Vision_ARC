# Requirements: CV Detection

<!-- SCOPE: Functional and quality requirements for the active ROS 2 auto-aim vision stack. -->
<!-- DOC_KIND: reference -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when changing runtime behavior, launch defaults, message contracts, or operator-facing workflows. -->
<!-- SKIP_WHEN: Skip when you only need architecture rationale, build commands, or infrastructure inventory. -->
<!-- PRIMARY_SOURCES: README.md, yolo_cpp/docs/overview.md, yolo_cpp/docs/pipeline.md, yolo_cpp/docs/interfaces.md, yolo_cpp/launch/basler.launch.py, yolo_cpp/launch/realsense.launch.py -->

## Quick Navigation

| Need | Read |
|------|------|
| System architecture | [architecture.md](architecture.md) |
| Technology inventory | [tech_stack.md](tech_stack.md) |
| Runtime operation | [runbook.md](runbook.md) |
| Package pipeline details | [yolo_cpp overview](../../yolo_cpp/docs/overview.md) |

## Agent Entry

| Signal | Value |
|--------|-------|
| Purpose | Defines the expected behavior and quality targets for the active auto-aim stack. |
| Read When | You are changing detector, tracker, processor, solver, launch, or STM bridge behavior. |
| Skip When | You only need setup commands or source-level implementation details. |
| Canonical | Yes |
| Primary Sources | `README.md`, `yolo_cpp/docs/`, `yolo_cpp/launch/`, `cv_auto_aim_interfaces/msg/` |

## Functional Requirements

| ID | Priority | Requirement | Current Evidence |
|----|----------|-------------|------------------|
| FR-CAM-001 | Must | The runtime shall acquire frames from RealSense or Basler camera backends through the shared C++ camera backend. | [camera_backend.hpp](../../yolo_cpp/include/yolo_cpp/camera_backend.hpp), [basler.launch.py](../../yolo_cpp/launch/basler.launch.py), [realsense.launch.py](../../yolo_cpp/launch/realsense.launch.py) |
| FR-CAM-002 | Should | Camera launch parameters shall expose frame size, frame rate, exposure, and backend-specific options. | [basler.launch.py](../../yolo_cpp/launch/basler.launch.py), [realsense.launch.py](../../yolo_cpp/launch/realsense.launch.py) |
| FR-DET-001 | Must | The detector shall publish armor candidates as `cv_auto_aim_interfaces/msg/Armors` on `detector/armors`. | [pipeline.md](../../yolo_cpp/docs/pipeline.md), [Armors.msg](../../cv_auto_aim_interfaces/msg/Armors.msg) |
| FR-DET-002 | Must | Armor candidates shall include image-space boxes, confidence, class, enemy-color status, optional OBB angle, optional corners, and optional camera-frame pose. | [Armor.msg](../../cv_auto_aim_interfaces/msg/Armor.msg) |
| FR-DET-003 | Should | The detector shall support native TensorRT and DeepStream-backed inference paths where the runtime dependencies are available. | [CMakeLists.txt](../../yolo_cpp/CMakeLists.txt), [detector_backend.hpp](../../yolo_cpp/include/yolo_cpp/detector_backend.hpp), [deepstream_runtime.hpp](../../yolo_cpp/include/yolo_cpp/deepstream_runtime.hpp) |
| FR-TRK-001 | Must | The tracker shall convert frame-level detections into stable tracks and publish them on `tracker/tracked_armors`. | [pipeline.md](../../yolo_cpp/docs/pipeline.md), [TrackedArmors.msg](../../cv_auto_aim_interfaces/msg/TrackedArmors.msg) |
| FR-TRK-002 | Must | The tracker shall publish a selected target on `tracker/target` for downstream control. | [Target.msg](../../cv_auto_aim_interfaces/msg/Target.msg), [nodes.md](../../yolo_cpp/docs/nodes.md) |
| FR-TRK-003 | Should | Tracking shall tolerate short detection gaps and avoid unstable target switching through configurable thresholds. | [tracker.hpp](../../yolo_cpp/include/yolo_cpp/tracker.hpp), [realsense.launch.py](../../yolo_cpp/launch/realsense.launch.py) |
| FR-PROC-001 | Must | The processor shall publish a filtered target on `processor/target` and keep a 2D-compatible fallback when 3D pose is missing. | [pipeline.md](../../yolo_cpp/docs/pipeline.md), [armor_processor_node.hpp](../../yolo_cpp/include/yolo_cpp/armor_processor_node.hpp) |
| FR-PROC-002 | Should | The processor shall publish spin observer and visualization outputs for tuning. | [nodes.md](../../yolo_cpp/docs/nodes.md), [SpinInfo.msg](../../cv_auto_aim_interfaces/msg/SpinInfo.msg) |
| FR-SOL-001 | Must | The solver shall convert the selected target into yaw, pitch, fire recommendation, legacy controller topics, and `AimSolution`. | [AimSolution.msg](../../cv_auto_aim_interfaces/msg/AimSolution.msg), [aim_solver_node.hpp](../../yolo_cpp/include/yolo_cpp/aim_solver_node.hpp) |
| FR-SOL-002 | Should | The solver shall support 3D pose and velocity solving with 2D FOV fallback. | [pipeline.md](../../yolo_cpp/docs/pipeline.md), [realsense.launch.py](../../yolo_cpp/launch/realsense.launch.py) |
| FR-STM-001 | Must | The full Basler field stack shall bridge controller-facing output to STM32 serial transport and ingest attitude, flags, color, or speed feedback. | [stm32_serial_bridge_node.hpp](../../yolo_cpp/include/yolo_cpp/stm32_serial_bridge_node.hpp), [basler.launch.py](../../yolo_cpp/launch/basler.launch.py) |
| FR-OPS-001 | Must | Operators shall be able to enter the container, build the active stack, and launch the full runtime through `uv` commands. | [README.md](../../README.md), [uv.sh](../../services/uv.sh), [runbook.md](runbook.md) |
| FR-DOC-001 | Should | Runtime behavior and interfaces shall remain documented in package-level docs under `yolo_cpp/docs/`. | [yolo_cpp/docs](../../yolo_cpp/docs/overview.md) |

## Non-Functional Requirements

| ID | Priority | Requirement | Acceptance Signal |
|----|----------|-------------|-------------------|
| NFR-LAT-001 | Must | The pipeline shall prioritize low-latency camera-to-aim operation on Jetson-class GPU hardware. | Launch files expose inference, tracker, and solver parameters; Docker runtime uses NVIDIA GPU access. |
| NFR-REL-001 | Must | Runtime stages shall communicate through explicit ROS 2 message contracts rather than shared implementation state. | Message definitions live in [cv_auto_aim_interfaces/msg](../../cv_auto_aim_interfaces/msg/). |
| NFR-MOD-001 | Must | The active runtime shall preserve separable detector, tracker, processor, solver, and STM bridge boundaries. | `yolo_cpp` registers component nodes and standalone executables in [CMakeLists.txt](../../yolo_cpp/CMakeLists.txt). |
| NFR-HW-001 | Must | Hardware-specific dependencies shall degrade predictably at build time when optional SDKs are absent. | CMake gates RealSense, TensorRT, DeepStream, GStreamer, and Pylon features through `HAVE_*` definitions. |
| NFR-MAINT-001 | Should | Legacy Python and combined-node paths shall remain clearly separated from the active C++ runtime. | Active package docs call out legacy/reference paths in [nodes.md](../../yolo_cpp/docs/nodes.md). |
| NFR-OPS-001 | Should | Docker and service scripts shall document host device, GPU, and path assumptions. | [infrastructure.md](infrastructure.md) and [runbook.md](runbook.md). |

## Out of Scope

| Area | Reason | Canonical Location |
|------|--------|--------------------|
| Web frontend behavior | This repository does not contain a detected React, Vue, Angular, or Svelte frontend. | Not applicable |
| HTTP API behavior | The active system is a ROS 2 runtime, not an HTTP backend under the coordinator detection rules. | ROS interfaces are covered in [interfaces.md](../../yolo_cpp/docs/interfaces.md) |
| Database persistence | No database dependency or migration tree is part of the active runtime. | Not applicable |
| Training pipeline guarantees | Training assets and scripts are legacy/supporting material rather than the active runtime contract. | [legacy/training/README.md](../../legacy/training/README.md) |

## Acceptance Checklist

| Change Area | Required Checks |
|-------------|-----------------|
| Message definitions | Update package docs, rebuild `cv_auto_aim_interfaces`, and confirm dependent nodes compile. |
| Launch parameters | Update [runbook.md](runbook.md) and package launch docs when operator commands change. |
| Detector or tracker behavior | Verify `detector/armors`, `tracker/tracked_armors`, and `tracker/target` contracts remain compatible. |
| Solver or STM bridge behavior | Verify controller-facing topic units and serial assumptions remain documented. |
| Docker/runtime setup | Update [infrastructure.md](infrastructure.md) and [runbook.md](runbook.md). |

## Maintenance

**Update Triggers:**
- New ROS node, executable, topic, or custom message
- Launch default changes
- Camera, detector, tracker, solver, or serial behavior changes
- Docker image or hardware requirement changes

**Verification:**
- [ ] Links resolve
- [ ] Requirement evidence paths exist
- [ ] Requirements still match the current ROS 2 launch and message contracts
- [ ] No web, database, or HTTP API requirements are implied without source evidence

**Last Updated:** 2026-05-15
