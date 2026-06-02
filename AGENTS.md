# CV Detection

NYU Robomaster ROS 2 C++ auto-aim vision pipeline for camera capture, YOLO/TensorRT/DeepStream armor detection, temporal tracking, 3D target processing, ballistic aim solving, and STM32 serial output. Legacy Python training, prototype tools, stale model variants, old setup notes, and retired Docker files remain under `legacy/`.

<!-- SCOPE: Canonical machine-facing entry point with repo map, critical rules, command overview, and links to detailed documentation only. -->
<!-- DOC_KIND: index -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Start here when you need the project map, local rules, or the next canonical document. -->
<!-- SKIP_WHEN: Skip when you already know the exact target document or code area. -->
<!-- PRIMARY_SOURCES: README.md, yolo_cpp/docs/overview.md, yolo_cpp/docs/pipeline.md, yolo_cpp/docs/nodes.md, yolo_cpp/docs/interfaces.md, CPP_CONTAINER_RUNBOOK.md, services/uv.sh -->

## Quick Navigation

| Need | Read |
|------|------|
| Documentation map | [docs/README.md](docs/README.md) |
| Project requirements | [docs/project/requirements.md](docs/project/requirements.md) |
| Project architecture | [docs/project/architecture.md](docs/project/architecture.md) |
| Tech stack | [docs/project/tech_stack.md](docs/project/tech_stack.md) |
| Operations runbook | [docs/project/runbook.md](docs/project/runbook.md) |
| Documentation standards | [docs/documentation_standards.md](docs/documentation_standards.md) |
| Development principles | [docs/principles.md](docs/principles.md) |
| Active C++ runtime overview | [yolo_cpp/docs/overview.md](yolo_cpp/docs/overview.md) |
| Pipeline stages | [yolo_cpp/docs/pipeline.md](yolo_cpp/docs/pipeline.md) |
| ROS nodes | [yolo_cpp/docs/nodes.md](yolo_cpp/docs/nodes.md) |
| Custom interfaces | [yolo_cpp/docs/interfaces.md](yolo_cpp/docs/interfaces.md) |
| Container and bring-up runbook | [CPP_CONTAINER_RUNBOOK.md](CPP_CONTAINER_RUNBOOK.md) |

## Agent Entry

| Field | Guidance |
|-------|----------|
| Purpose | Canonical repo map and routing layer for agents working in CV Detection. |
| Read when | You need local rules, active package boundaries, commands, or the next canonical doc. |
| Skip when | You already know the exact source file or document to inspect. |
| Canonical | Yes. Keep shared agent guidance here; keep harness-specific deltas in thin stubs. |
| Read next | `docs/README.md`, then the relevant `yolo_cpp/docs/` page or source area. |
| Primary sources | `README.md`, `yolo_cpp/docs/`, `CPP_CONTAINER_RUNBOOK.md`, `services/uv.sh`, package CMake files. |

## Critical Rules

| Category | Rule | When to Apply |
|----------|------|---------------|
| Scope | Treat `yolo_cpp/` and `cv_auto_aim_interfaces/` as the active ROS 2 runtime. | Runtime, build, and launch changes. |
| Legacy | Treat `legacy/` as archived or prototype material unless a task explicitly targets it. | Training, migration, or historical comparisons. |
| Interfaces | Update `cv_auto_aim_interfaces/msg/` and downstream publishers/subscribers together. | Message field, topic, or control-contract changes. |
| Pipeline | Preserve the detector -> tracker -> processor -> solver -> STM bridge boundary unless the task proves a cross-stage change is needed. | Behavior changes in the active stack. |
| Hardware | Verify optional camera and inference backends at configure and launch time: RealSense, Basler Pylon, TensorRT, CUDA, DeepStream, and GStreamer are environment-dependent. | Container, Jetson, camera, or detector backend work. |
| Documentation | Link to canonical docs instead of duplicating long implementation detail. | Documentation and agent-instruction edits. |
| Testing | Run the narrowest meaningful build or test for the touched area; use `colcon test --packages-select yolo_cpp` for C++ tests when available. | Before calling code changes complete. |
| Formatting | Use existing project tooling and pre-commit hooks for mechanical formatting. | Before committing broad text or script changes. |
| Safety | Do not transmit controller-facing assumptions to STM32 without checking unit conventions; ROS aim angles are radians, STM TX converts to degrees by default. | Solver, serial bridge, and protocol work. |

### Key Principles

| Principle | Meaning Here |
|-----------|--------------|
| Active stack first | Prefer the modular C++ ROS 2 path for runtime work. |
| Stage ownership | Keep capture, detection, tracking, target processing, solving, and serial transport responsibilities separated. |
| Verified hardware claims | Treat camera and inference backend availability as a build or launch condition to check, not an assumption. |
| Canonical docs | Update the owning doc and link outward rather than copying facts between files. |

## Navigation Rules

| Topic | Canonical Source |
|-------|------------------|
| Project quick start | [README.md](README.md) |
| Project requirements | [docs/project/requirements.md](docs/project/requirements.md) |
| Project architecture | [docs/project/architecture.md](docs/project/architecture.md) |
| Tech stack and dependencies | [docs/project/tech_stack.md](docs/project/tech_stack.md) |
| Infrastructure and operations | [docs/project/infrastructure.md](docs/project/infrastructure.md) and [docs/project/runbook.md](docs/project/runbook.md) |
| Active runtime shape | [yolo_cpp/docs/overview.md](yolo_cpp/docs/overview.md) |
| Stage-by-stage data flow | [yolo_cpp/docs/pipeline.md](yolo_cpp/docs/pipeline.md) |
| Node responsibilities | [yolo_cpp/docs/nodes.md](yolo_cpp/docs/nodes.md) |
| ROS message contracts | [yolo_cpp/docs/interfaces.md](yolo_cpp/docs/interfaces.md) and [cv_auto_aim_interfaces/msg](cv_auto_aim_interfaces/msg) |
| Launch parameters | [yolo_cpp/docs/launch-and-config.md](yolo_cpp/docs/launch-and-config.md) |
| Runtime bring-up | [CPP_CONTAINER_RUNBOOK.md](CPP_CONTAINER_RUNBOOK.md) and [Docker/README.md](Docker/README.md) |
| Pre-commit setup | [docs/Pre-commit.md](docs/Pre-commit.md) |

Use each document's `SCOPE`, `READ_WHEN`, and `SKIP_WHEN` metadata before reading deep content. Keep `AGENTS.md` as the single source for shared agent instructions; `CLAUDE.md` imports it with `@AGENTS.md`.

## Development Commands

| Task | Bash Command | Notes |
|------|--------------|-------|
| Enter container | `uv docker` | Selects or starts the C++ container image through `services/uv.sh`. |
| Build all ROS packages | `uv build` | Runs `colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF` in a ROS workspace. |
| Build active stack | `uv build yolo_cpp` | Builds `cv_auto_aim_interfaces` and `yolo_cpp` together. |
| Launch Basler field stack | `uv run yolo_cpp basler.launch.py` | Basler, DeepStream, detector, tracker, processor, solver, and STM bridge. |
| Launch RealSense stack | `uv run yolo_cpp realsense.launch.py` | Detector, tracker, processor, and solver without serial bridge. |
| Launch compatibility alias | `uv run yolo_cpp stm.launch.py` | Deprecated alias that forwards to `basler.launch.py`. |
| Launch zero-copy experiment | `uv run yolo_cpp zerocopy.launch.py` | Component-container path for the combined perception node. |
| Install competition service | `./services/stm-stack.sh install` | Installs and enables the headless STM systemd service without starting it. |
| Competition preflight | `./services/stm-stack.sh preflight` | Checks image, package, model, DeepStream/Pylon plugins, and serial path before startup. |
| Restart competition stack | `./services/stm-stack.sh restart` | Restarts the `cv-stm-stack` service container through systemd. |
| Competition logs and health | `./services/stm-stack.sh logs` / `./services/stm-stack.sh health` | Follows service logs or checks expected ROS nodes and topics. |
| Jetson FPS isolation benchmark | `./services/fps-benchmark.sh --mode all` | Measures synthetic TensorRT inference, camera capture-only, and Basler+DeepStream result FPS using STM defaults. |
| Native build | `colcon build --packages-select cv_auto_aim_interfaces yolo_cpp --symlink-install --cmake-args -DBUILD_TESTING=OFF` | Use in an environment with ROS 2 Humble and dependencies installed. |
| C++ tests | `colcon test --packages-select yolo_cpp` | Uses the package test targets. |
| Pre-commit | `pre-commit run --all-files` | Runs configured repository hooks. |

## Repo Map

| Path | Owns |
|------|------|
| `yolo_cpp/include/yolo_cpp/` | C++ public headers for nodes, backends, tracking, solving, and protocol helpers. |
| `yolo_cpp/src/` | Active ROS 2 C++ runtime implementations. |
| `yolo_cpp/launch/` | `basler.launch.py`, compatibility alias `stm.launch.py`, `realsense.launch.py`, and `zerocopy.launch.py`. |
| `yolo_cpp/config/` | DeepStream and detector configuration assets. |
| `yolo_cpp/config/default_model_path.txt` | Robot-local fallback model path used when launch arguments and `CV_STM_MODEL_PATH` are empty. |
| `yolo_cpp/test/` | Focused C++ tests for tracking and detection postprocess behavior. |
| `cv_auto_aim_interfaces/msg/` | Custom ROS 2 messages for detections, tracks, targets, spin state, and aim solutions. |
| `Docker/` and `services/` | Container images, compose config, and helper scripts. |
| `legacy/models/` | Stale or experimental model variants and archived export utilities. |
| `legacy/yolov8_ros/` | Archived Python ROS 2 YOLOv8 package from the former main branch. |
| `legacy/cpp/` | Archived C++ reference nodes removed from the active package build. |
| `legacy/prototypes/mac/` | Archived macOS/CoreML prototype tooling. |
| `legacy/docker/` | Retired Dockerfiles, compose files, and VIAM-era container helpers. |
| `legacy/docs/setup/`, `legacy/docs/yolov8_node/`, `legacy/envs/`, `legacy/nav/`, and `legacy/scripts/` | Historical setup notes, archived YOLOv8 Python docs, conda environments, navigation notes, and root helper scripts. |

## Maintenance

**Update Triggers:**
- Active package, launch, topic, or command structure changes.
- New canonical docs are added, removed, or renamed.
- Runtime ownership moves between active C++ packages and legacy/prototype areas.

**Verification:**
- [ ] All internal links resolve.
- [ ] Commands match `services/uv.sh`, launch files, and package build files.
- [ ] The file remains under 200 lines.
- [ ] Shared guidance appears here rather than being duplicated in `CLAUDE.md`.

**Last Updated:** 2026-05-30
