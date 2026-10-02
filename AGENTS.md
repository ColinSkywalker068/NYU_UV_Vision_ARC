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
| Saved env and startup | [docs/project/stm-stack-config.md](docs/project/stm-stack-config.md) |
| Tracking deployment and diagnostics | [docs/validation/autoaim-manual-checks.md](docs/validation/autoaim-manual-checks.md) |
| Local inference and replay | [tools/model-eval/README.md](tools/model-eval/README.md) |
| Operations runbook | [docs/project/runbook.md](docs/project/runbook.md) |
| Jetson SSH access | [docs/project/ssh-access.md](docs/project/ssh-access.md) |
| Jetson Orin Nano reflash | [docs/project/jetson-reflash.md](docs/project/jetson-reflash.md) |
| Jetson VNC access | [docs/project/jetson-vnc.md](docs/project/jetson-vnc.md) |
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
| Testing | Run the narrowest meaningful check. ROS tests require a build with `BUILD_TESTING=ON`, a sourced install, `colcon test --packages-select yolo_cpp --return-code-on-test-failure`, and `colcon test-result --verbose`. | Before calling code changes complete. |
| Formatting | Use existing project tooling and pre-commit hooks for mechanical formatting. | Before committing broad text or script changes. |
| Safety | Do not transmit controller-facing assumptions to STM32 without checking unit conventions; current controls firmware expects CV->STM pitch/yaw as radians, so `transmit_angles_in_degrees` defaults false. | Solver, serial bridge, and protocol work. |

### Current runtime contracts

- The live env is Git-ignored `services/stm-stack.env` in the Jetson checkout; `.env.example` is a template. `cv install` preserves or migrates existing settings and records the service paths. Do not overwrite a tuned env or keep editing the old `/etc/cv-detection` copy. See the [config guide](docs/project/stm-stack-config.md).
- Env-only changes need save + `cv restart`. Source, message, and launch changes need a compatible rebuild first. JetPack 6/Humble and JetPack 7.2/Jazzy use different runtimes; do not assume TensorRT engines are portable between them.
- `transmit_enable:=false` keeps normal bridge command TX off while preview and RX can run. Absolute/offset display mode and provisional calibration are not TX switches. Keep fire recommendations disabled unless the operator explicitly enables them; MCU acceptance gates are separate.
- `cv start` / `cv restart` check ROS readiness and fresh decoded attitude when RX/TX is enabled. A failed check leaves CV running for inspection. These commands do not calibrate or flash the Type C. `cv health` repeats the read-only check.
- Preserve image capture time, attitude alignment, frame conventions, measurement age, and bounded coasting across detector/tracker/processor/solver changes. Host receive timestamps do not prove MCU source-clock synchronization. See the [telemetry plan](docs/project/tracker-telemetry-plan.md).
- Older models in `yolo_cpp/models/` and `legacy/models/` remain available. Preserve them when adding a backend; switching requires matching model path, parser/task, class mapping, and engine runtime, not just a filename change.
- Jetson owns perception and solver configuration; Type C owns motor control, slew limits, and acceptance. The test fixture is a recorded controller reference, not proof of currently flashed firmware. Simulation and fresh telemetry do not establish physical tracking stability.
- Diagnose with saved `vision_logs/` alignment records and `tools/analyze-tracking-alignment.py`; serial write success is not proof of MCU acceptance or motor response. See [manual validation](docs/validation/autoaim-manual-checks.md).

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
| Jetson SSH access | [docs/project/ssh-access.md](docs/project/ssh-access.md) |
| Jetson Orin Nano reflash | [docs/project/jetson-reflash.md](docs/project/jetson-reflash.md) |
| Jetson remote desktop | [docs/project/jetson-vnc.md](docs/project/jetson-vnc.md) |
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
| Restart competition stack | `./services/stm-stack.sh restart` | Restarts through systemd, then checks ROS and fresh Type C telemetry. |
| Toggle service preview | `./services/stm-stack.sh preview-web` / `preview-gui` / `preview-off` | Persists `CV_STM_SHOW` and restarts; does not change TX configuration. |
| Competition logs and health | `./services/stm-stack.sh logs` / `./services/stm-stack.sh health` | Follows service logs or checks ROS readiness and fresh decoded attitude. |
| Jetson FPS isolation benchmark | `./services/fps-benchmark.sh --mode all` | Measures synthetic TensorRT inference, camera capture-only, and Basler+DeepStream result FPS using STM defaults. |
| Native build | `colcon build --packages-select cv_auto_aim_interfaces yolo_cpp --symlink-install --cmake-args -DBUILD_TESTING=OFF` | Source the matching Humble or Jazzy runtime; this production build does not build tests. |
| Local ROS build/tests without hardware | `bash tools/run-local-tracking-tests.sh` | Docker Humble build, live ROS health scenarios, and package tests using synthetic inputs/PTYs. |
| Health-check integration test | `python3 services/test_typec_link_ros.py` | Run in a sourced ROS test workspace; synthetic publishers, no hardware. |
| Pre-commit | `pre-commit run --all-files` | Runs configured repository hooks. |

## Repo Map

| Path | Owns |
|------|------|
| `yolo_cpp/include/yolo_cpp/` | C++ public headers for nodes, backends, tracking, solving, and protocol helpers. |
| `yolo_cpp/src/` | Active ROS 2 C++ runtime implementations. |
| `yolo_cpp/launch/` | Field Basler/RealSense launches, compatibility aliases, `tracking_observe.launch.py` (no solver/serial), and zero-copy experiment. |
| `yolo_cpp/config/` | DeepStream and detector configuration assets. |
| `yolo_cpp/config/default_model_path.txt` | Robot-local fallback model path used when launch arguments and `CV_STM_MODEL_PATH` are empty. |
| `yolo_cpp/test/` | Geometry, tracking, closed-loop ROS, launch, and serial reconnect tests; recorded Type C reference fixture. |
| `cv_auto_aim_interfaces/msg/` | Custom ROS 2 messages for detections, tracks, targets, spin state, and aim solutions. |
| `tools/model-eval/` | Active Mac inference/replay evaluation and model candidate tooling; not legacy. |
| `.gitlab-ci.yml` | Python tracking checks and ROS Humble build/health/package tests. |
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

**Last Updated:** 2026-10-01
