# Architecture Patterns Catalog: CV Detection

<!-- SCOPE: Catalog of architectural patterns detected in the current codebase and where to audit them. -->
<!-- DOC_KIND: reference -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when evaluating recurring design patterns, refactoring runtime boundaries, or adding ADRs. -->
<!-- SKIP_WHEN: Skip when you only need operational commands or exact message fields. -->
<!-- PRIMARY_SOURCES: docs/project/architecture.md, yolo_cpp/docs/pipeline.md, yolo_cpp/src, yolo_cpp/include/yolo_cpp, cv_auto_aim_interfaces/msg -->

## Quick Navigation

| Need | Read |
|------|------|
| Architecture overview | [architecture.md](../project/architecture.md) |
| Requirements | [requirements.md](../project/requirements.md) |
| Tech stack | [tech_stack.md](../project/tech_stack.md) |
| Package pipeline | [pipeline.md](../../yolo_cpp/docs/pipeline.md) |

## Agent Entry

| Signal | Value |
|--------|-------|
| Purpose | Tracks recurring architecture patterns and their audit points. |
| Read When | You are changing boundaries, messaging, hardware adapters, or fallback behavior. |
| Skip When | You only need launch commands or source-level implementation details. |
| Canonical | Yes |
| Primary Sources | `docs/project/architecture.md`, `yolo_cpp/docs/`, `yolo_cpp/src`, `cv_auto_aim_interfaces/msg` |

## Pattern Summary

| Pattern | Status | Evidence | Audit Focus |
|---------|--------|----------|-------------|
| Pipeline Stages | Detected | Detector, tracker, processor, solver, and bridge nodes communicate through ROS topics. | Confirm each stage owns a single clear responsibility and topic contract. |
| Message Contract Boundary | Detected | `cv_auto_aim_interfaces/msg` contains the shared runtime contracts. | Check compatibility before changing fields or units. |
| Hardware Adapter | Detected | Camera backend supports RealSense and Basler; detector backend supports native TensorRT and DeepStream paths. | Keep adapter-specific behavior behind backend boundaries. |
| Optional Capability Gating | Detected | CMake defines `HAVE_*` flags when optional SDKs are found. | Verify launch defaults do not assume a feature compiled out on the target host. |
| Fallback Path | Detected | 3D pose solving can fall back to 2D target projection when pose is unavailable. | Ensure fallback behavior remains explicit and observable. |
| Legacy Reference Path | Detected | `PerceptionNode`, `YoloNode`, `legacy/`, and `mac/` remain outside the primary split runtime. | Avoid routing new active behavior through legacy paths without a documented decision. |

## Pattern Details

### Pipeline Stages

| Attribute | Value |
|-----------|-------|
| Intent | Separate camera/detection, tracking, target processing, solving, and serial transport. |
| Benefit | Easier tuning, debugging, replacement, and topic inspection. |
| Cost | Message compatibility and launch wiring must stay synchronized. |
| Primary Sources | [pipeline.md](../../yolo_cpp/docs/pipeline.md), [nodes.md](../../yolo_cpp/docs/nodes.md) |

### Message Contract Boundary

| Attribute | Value |
|-----------|-------|
| Intent | Use generated ROS messages as the stable contract between runtime stages. |
| Benefit | Nodes can evolve internally without sharing implementation state. |
| Cost | Message field changes are breaking changes for every subscriber. |
| Primary Sources | [interfaces.md](../../yolo_cpp/docs/interfaces.md), [cv_auto_aim_interfaces/msg](../../cv_auto_aim_interfaces/msg/Armor.msg) |

### Hardware Adapter

| Attribute | Value |
|-----------|-------|
| Intent | Isolate camera and detector hardware differences behind backend abstractions. |
| Benefit | RealSense, Basler, native TensorRT, and DeepStream support can coexist. |
| Cost | Runtime failures can occur when launch parameters request a backend that was not built. |
| Primary Sources | [camera_backend.hpp](../../yolo_cpp/include/yolo_cpp/camera_backend.hpp), [detector_backend.hpp](../../yolo_cpp/include/yolo_cpp/detector_backend.hpp) |

### Optional Capability Gating

| Attribute | Value |
|-----------|-------|
| Intent | Detect SDK availability during CMake configure and compile feature flags accordingly. |
| Benefit | Development machines can build partial capability sets. |
| Cost | Operators need clear build-time verification before using optional runtime features. |
| Primary Sources | [yolo_cpp/CMakeLists.txt](../../yolo_cpp/CMakeLists.txt), [CPP_CONTAINER_RUNBOOK.md](../../CPP_CONTAINER_RUNBOOK.md) |

### Fallback Path

| Attribute | Value |
|-----------|-------|
| Intent | Preserve aiming output when calibrated 3D pose is temporarily unavailable. |
| Benefit | Runtime can continue operating from image-space target projection. |
| Cost | Solver consumers must understand whether the output used 3D state or fallback projection. |
| Primary Sources | [AimSolution.msg](../../cv_auto_aim_interfaces/msg/AimSolution.msg), [pipeline.md](../../yolo_cpp/docs/pipeline.md) |

## Four-Score Review Model

Use this table when auditing a pattern before a refactor.

| Score | Question | Good Signal |
|-------|----------|-------------|
| Fit | Does the pattern match the runtime problem? | Stage or adapter boundary maps to an actual hardware or ROS contract. |
| Clarity | Is ownership obvious? | The owning node, message, and launch parameter are easy to locate. |
| Change Cost | Can the pattern evolve safely? | Callers depend on documented messages or parameters, not hidden shared state. |
| Testability | Can behavior be checked? | There is a launch, topic, unit test, or documented command that exercises the boundary. |

## Trend Tracking

| Trend | Current State | Watch For |
|-------|---------------|-----------|
| Split runtime maturity | Active docs and launch files favor detector -> tracker -> processor -> solver. | New behavior landing in combined legacy nodes first. |
| 3D target processing | Custom messages and processor/solver support pose, velocity, and range fields. | Unit ambiguity or inconsistent fallback behavior. |
| Hardware dependency growth | Docker and CMake now manage several GPU and camera SDK paths. | Build scripts becoming the only source of deployment truth. |
| Documentation maturity | Package docs and project docs now share canonical links. | Duplicate claims drifting between `docs/project` and `yolo_cpp/docs`. |

## Maintenance

**Update Triggers:**
- New architectural pattern is introduced
- A legacy path becomes active again
- A backend abstraction changes
- Message fields or topic ownership changes

**Verification:**
- [ ] Pattern evidence links resolve
- [ ] Status reflects current code, not intended future state
- [ ] Architecture doc and pattern catalog agree on runtime boundaries
- [ ] Any formal ADR path is linked once introduced

**Last Updated:** 2026-05-15
