# Development Principles

<!-- SCOPE: Project development principles and tradeoffs only. Contains reusable principles, decision order, anti-patterns, and verification guidance. -->
<!-- DOC_KIND: explanation -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when making implementation or documentation decisions and you need the governing principles. -->
<!-- SKIP_WHEN: Skip when you only need routing or exact factual lookup. -->
<!-- PRIMARY_SOURCES: docs/documentation_standards.md, AGENTS.md, yolo_cpp/docs/development-notes.md -->

## Quick Navigation

| Need | Read |
|------|------|
| Documentation rules | [documentation_standards.md](documentation_standards.md) |
| Documentation map | [README.md](README.md) |
| Active package development notes | [../yolo_cpp/docs/development-notes.md](../yolo_cpp/docs/development-notes.md) |
| Runtime pipeline | [../yolo_cpp/docs/pipeline.md](../yolo_cpp/docs/pipeline.md) |

## Agent Entry

| Field | Guidance |
|-------|----------|
| Purpose | Explain the governing principles and decision hierarchy for CV Detection changes. |
| Read when | You need to choose between valid approaches or justify a tradeoff. |
| Skip when | You only need a direct factual lookup in source or package docs. |
| Canonical | Yes, for project principles. |
| Read next | The package doc or source area that owns the concrete behavior. |
| Primary sources | `docs/documentation_standards.md`, `AGENTS.md`, and `yolo_cpp/docs/development-notes.md`. |

## Core Principles

| # | Principle | Application |
|---|-----------|-------------|
| 1 | Standards First | Use ROS 2, ament, colcon, CMake, and message-generation conventions before inventing local patterns. |
| 2 | YAGNI | Add only the runtime, interface, or documentation behavior needed for the current verified requirement. |
| 3 | KISS | Prefer a small stage-local change in detector, tracker, processor, solver, or bridge code over a broad rewrite. |
| 4 | DRY | Keep one owner for each fact: messages in `cv_auto_aim_interfaces/msg/`, runtime docs in `yolo_cpp/docs/`, root rules in `AGENTS.md`. |
| 5 | Consumer-First Design | Design ROS messages, topics, launch parameters, and docs from the downstream caller's needs. |
| 6 | No Legacy Code | Do not extend historical Python paths when the active C++ runtime is the target; isolate or retire obsolete paths when a task allows it. |
| 7 | Documentation-as-Code | Update docs with behavior, command, topic, or ownership changes in the same workstream. |
| 8 | Security by Design | Treat serial output, host devices, privileged containers, and controller-facing units as safety-sensitive surfaces. |
| 9 | Auto-Generated Migrations Only | When schema or generated-interface workflows exist, derive generated artifacts from source definitions; this repo currently uses ROS message generation rather than a database migration stack. |

## Decision-Making Framework

| Priority | Question |
|----------|----------|
| 1. Security and safety | Could this change harm hardware, transmit the wrong units, expose host devices unnecessarily, or weaken runtime isolation? |
| 2. Standards compliance | Does the approach fit ROS 2, ament, CMake, Docker, and project documentation conventions? |
| 3. Correctness | Does it preserve detector, tracker, processor, solver, and STM bridge contracts? |
| 4. Simplicity | Is there a smaller local change that solves the same problem? |
| 5. Necessity | Is the change required now, or is it speculative? |
| 6. Maintainability | Will the next teammate understand where the behavior lives and how to verify it? |
| 7. Performance | Does it preserve runtime latency and throughput without premature optimization? |

### Trade-Offs

| Conflict | Lower Priority | Higher Priority | Resolution |
|----------|----------------|-----------------|------------|
| Faster field workaround vs controller safety | Speed | Security and safety | Keep unit conversions and serial protocol behavior explicit and verified. |
| Broad architecture cleanup vs targeted runtime fix | Maintainability | Necessity | Make the smallest correct fix, then document larger cleanup separately. |
| Duplicate docs for convenience vs one canonical source | Convenience | DRY | Link to the canonical source and update navigation. |
| Hand-rolled inference path vs proven backend integration | Local control | Standards compliance | Use the existing TensorRT, DeepStream, or detector backend boundaries unless requirements prove otherwise. |
| Extra launch option vs simpler operator workflow | Flexibility | Simplicity | Add parameters only when they map to a real bring-up or tuning need. |

## Anti-Patterns to Avoid

| Anti-Pattern | Why It Hurts This Project |
|--------------|---------------------------|
| Cross-stage edits for a stage-local bug | Makes detector, tracker, processor, solver, and bridge behavior harder to verify. |
| Treating optional hardware as always present | Breaks builds or launches on machines without RealSense, Pylon, TensorRT, CUDA, DeepStream, or GStreamer. |
| Duplicating message facts in prose | Creates drift from `cv_auto_aim_interfaces/msg/` and generated ROS interfaces. |
| Extending legacy Python when active C++ is the target | Splits runtime behavior across two stacks. |
| Hiding unit conversions | Increases risk at the solver and STM serial boundary. |
| Adding docs that link to missing paths | Wastes review time and weakens agent navigation. |
| Optimizing before measuring runtime impact | Adds complexity without proving frame-rate, latency, or stability gains. |

## Verification Checklist

- [ ] Standards First: the change follows ROS 2, CMake, Docker, and docs conventions already present.
- [ ] YAGNI: speculative features and unused abstractions are absent.
- [ ] KISS: the solution is localized to the smallest responsible package or document.
- [ ] DRY: duplicated facts are removed or replaced with links to canonical sources.
- [ ] Consumer-First Design: downstream node, controller, or reader needs are explicit.
- [ ] No Legacy Code: active C++ runtime work does not accidentally expand legacy Python paths.
- [ ] Documentation-as-Code: relevant docs and navigation are updated with the behavior.
- [ ] Security by Design: privileged containers, devices, serial output, and unit conversions are checked.
- [ ] Auto-Generated Migrations Only: generated or schema-derived artifacts come from source definitions where applicable.

## Maintenance

**Update Triggers:**
- Project principles or tradeoff priorities change.
- A recurring review issue deserves a documented principle.
- Runtime ownership changes between active C++ packages and legacy/prototype areas.
- New generated-interface or schema workflows are introduced.

**Verification:**
- [ ] All nine core principles still apply to this repository.
- [ ] Decision order matches current team expectations.
- [ ] Anti-patterns are concrete and repository-specific.
- [ ] Links resolve and point to canonical sources.

**Last Updated:** 2026-05-15
