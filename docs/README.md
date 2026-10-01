# Documentation System

<!-- SCOPE: Documentation hub and navigation layer only. Routes readers to verified repository docs and explains ownership boundaries. -->
<!-- DOC_KIND: index -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when you need the documentation map or want to locate the canonical doc for a topic. -->
<!-- SKIP_WHEN: Skip when you already know the exact canonical document. -->
<!-- PRIMARY_SOURCES: README.md, AGENTS.md, docs/documentation_standards.md, docs/principles.md, yolo_cpp/docs/overview.md -->

## Quick Navigation

| Area | Key Documents | Use For |
|------|---------------|---------|
| Root map | [../AGENTS.md](../AGENTS.md), [../CLAUDE.md](../CLAUDE.md) | Agent entry points; `AGENTS.md` is canonical and `CLAUDE.md` imports it. |
| Project overview | [../README.md](../README.md), [../yolo_cpp/docs/overview.md](../yolo_cpp/docs/overview.md) | High-level project purpose, quick start, and active C++ package overview. |
| Canonical project docs | [project/requirements.md](project/requirements.md), [project/architecture.md](project/architecture.md), [project/tech_stack.md](project/tech_stack.md), [project/infrastructure.md](project/infrastructure.md), [project/runbook.md](project/runbook.md) | Requirements, architecture, stack inventory, infrastructure, and operations. |
| Jetson SSH access | [project/ssh-access.md](project/ssh-access.md) | Same-Wi-Fi `.local` SSH for `rm-standard.local` and `rm-sentry.local`, plus wired SSH through `192.168.55.1`. |
| Jetson reflash | [project/jetson-reflash.md](project/jetson-reflash.md) | Optimized Orin Nano reflash and first-boot recovery for the current DeepStream 6.4 container stack. |
| Jetson operator access | [project/jetson-vnc.md](project/jetson-vnc.md) | Free non-RealVNC remote desktop setup for configuring one or more Jetsons on the same Wi-Fi or lab LAN. |
| Pattern catalog | [architecture/patterns_catalog.md](architecture/patterns_catalog.md) | Recurring runtime patterns and audit points. |
| Runtime pipeline | [../yolo_cpp/docs/pipeline.md](../yolo_cpp/docs/pipeline.md), [../yolo_cpp/docs/nodes.md](../yolo_cpp/docs/nodes.md), [../yolo_cpp/docs/interfaces.md](../yolo_cpp/docs/interfaces.md) | Detector, tracker, processor, solver, STM bridge, and custom messages. |
| Launch and config | [../yolo_cpp/docs/launch-and-config.md](../yolo_cpp/docs/launch-and-config.md), [../CPP_CONTAINER_RUNBOOK.md](../CPP_CONTAINER_RUNBOOK.md), [../Docker/README.md](../Docker/README.md) | Launch choice, container bring-up, camera backends, and inference runtime. |
| Standards | [documentation_standards.md](documentation_standards.md), [principles.md](principles.md), [Pre-commit.md](Pre-commit.md) | Documentation rules, development principles, and pre-commit setup. |
| Local tracking evaluation | [validation/autoaim-local-upgrade.md](validation/autoaim-local-upgrade.md), [validation/autoaim-manual-checks.md](validation/autoaim-manual-checks.md), [../tools/model-eval/README.md](../tools/model-eval/README.md) | ARUW model evidence, production ROS tests, offline video overlays, and manual Jetson evaluation. |
| Tracker telemetry plan | [project/tracker-telemetry-plan.md](project/tracker-telemetry-plan.md) | Planned MCU/camera time alignment, quaternion history, diagnostics, paired commits, and validation gates. |
| Robot tracking plan | [project/robot-tracking-sim2real-plan.md](project/robot-tracking-sim2real-plan.md) | Canonical spin/non-spin tracking, safe observation, simulation, and sim-to-real plan. |
| Earlier roadmaps | [RM_AUTO_AIM_PARITY_NEXT_STEPS.md](RM_AUTO_AIM_PARITY_NEXT_STEPS.md), [SPIN_AND_ARMOR_OFFSET_NEXT_STEPS.md](SPIN_AND_ARMOR_OFFSET_NEXT_STEPS.md) | Historical planning notes that informed the consolidated robot-tracking plan. |
| Legacy archive | [../legacy/yolov8_ros](../legacy/yolov8_ros), [../legacy/docs/yolov8_node](../legacy/docs/yolov8_node), [../legacy/docs/Pipeline/Pipeline.md](../legacy/docs/Pipeline/Pipeline.md), [../legacy/docs/setup](../legacy/docs/setup), [../legacy/prototypes/mac](../legacy/prototypes/mac), [../legacy/docker](../legacy/docker), [../legacy/cpp](../legacy/cpp) | Archived Python YOLOv8 ROS package, historical pipeline docs, old setup notes, prototypes, retired containers, archived C++ references, and stale model variants. |

## Agent Entry

| Field | Guidance |
|-------|----------|
| Purpose | Route readers to the right source without duplicating project knowledge. |
| Read when | You need to choose the next document for build, runtime, pipeline, interface, or standards work. |
| Skip when | You already know the specific file or source area to inspect. |
| Canonical | Yes, for documentation navigation. |
| Read next | The canonical document listed for the task area above. |
| Primary sources | `README.md`, `AGENTS.md`, `yolo_cpp/docs/`, `CPP_CONTAINER_RUNBOOK.md`, and this docs directory. |

## Overview

The repository documentation is split by audience and stability. Root docs identify the active project, agent rules, and documentation standards. `yolo_cpp/docs/` owns the active ROS 2 C++ runtime explanation: package overview, pipeline flow, node responsibilities, launch choices, interfaces, and development notes. The top-level `docs/` directory holds current project docs, planning notes, and repository-wide standards. `Docker/` and `services/` describe or implement the active container workflow. `legacy/` preserves older pipeline material, setup notes, prototype scripts, archived C++ references, retired Docker files, environment files, navigation notes, and stale model variants.

## Documentation Map

| Area | What It Owns | What It Does Not Own |
|------|---------------|----------------------|
| Root docs | Agent entry, root navigation, documentation standards, development principles. | Detailed implementation behavior. |
| `docs/project/` | Canonical project requirements, architecture, tech stack, infrastructure, and runbook. | Package-specific source walkthroughs already owned by `yolo_cpp/docs/`. |
| `docs/architecture/` | Cross-project architecture pattern catalog and audit routing. | Runtime command procedures. |
| `yolo_cpp/docs/` | Active C++ runtime architecture, message flow, nodes, launches, and package development notes. | Legacy Python training details. |
| `docs/` | Current project docs, planning notes, standards, principles, and documentation governance. | Source-level ownership of ROS nodes; historical setup archives. |
| `Docker/` | Active container images, compose defaults, and Docker-specific run notes. | ROS message or topic contracts; retired Docker files. |
| `legacy/` | Historical Python tools, archived Python YOLOv8 ROS package, archived C++ references, prototype scripts, training utilities, stale model variants, retired Docker files, archived setup notes, and old navigation docs. | Current active runtime guidance unless a task explicitly targets legacy behavior. |

## General Documentation Standards

| Standard | Requirement |
|----------|-------------|
| SCOPE tags | Every generated markdown doc starts with `SCOPE`, `DOC_KIND`, `DOC_ROLE`, `READ_WHEN`, `SKIP_WHEN`, and `PRIMARY_SOURCES` comments near the top. |
| Maintenance | Generated docs end with update triggers, verification checks, and a `Last Updated` date. |
| Numbering | Use stable headings and tables so readers can scan without relying on fragile prose ordering. |
| Placeholders | Published docs must not contain unresolved template markers or drafting markers. |
| Source links | Prefer links to canonical docs and source paths over copied implementation detail. |
| Current paths | Every linked repository path must exist at the time the doc is written. |

## Writing Guidelines

Use progressive disclosure: start with a navigation table, then provide the minimum context needed to choose the next source. Prefer tables for ownership, commands, and decision matrices. Keep generated docs in English, concise, and factual. Avoid implementation examples in root standards documents; link to package docs or source when readers need exact behavior.

## Standards Compliance

| Standard | How This Repo Applies It |
|----------|--------------------------|
| ISO/IEC/IEEE 29148:2018 | Requirements and constraints are stated as verifiable claims. |
| Diataxis | Navigation, explanation, how-to, and reference content are separated where practical. |
| arc42 | Architecture docs describe context, building blocks, runtime behavior, and tradeoffs. |
| C4 Model | System and container boundaries are described before component details. |
| ADR format | Architecture decisions should be recorded as context, decision, consequences, and status when introduced. |
| Documentation-as-Code | Docs live with the repo, use relative links, and are reviewed with code changes. |

## Contributing to Documentation

1. Identify the canonical owner before editing; update links instead of creating duplicate explanations.
2. Add the required metadata comments near the top of generated markdown.
3. Keep the first screen useful: `Quick Navigation`, `Agent Entry`, and the core map should appear before deep detail.
4. Verify every relative link and referenced path against the repository.
5. Update the `Maintenance` section when ownership, commands, or verification changes.
6. Run the narrowest useful validation for the changed docs, such as link checks by inspection or pre-commit when configured.

## Maintenance

**Update Triggers:**
- Canonical docs are added, removed, or renamed.
- Runtime ownership changes between `yolo_cpp/`, `cv_auto_aim_interfaces/`, `Docker/`, or legacy areas.
- Documentation standards or principles move.

**Verification:**
- [ ] Every internal link resolves.
- [ ] Ownership boundaries match current repository structure.
- [ ] Navigation points to active docs before historical material.
- [ ] Root agent files remain canonical-stub compliant.

**Last Updated:** 2026-10-01

- [Local pose/tracking upgrade and validation](validation/autoaim-local-upgrade.md) — development contracts, candidate models, reproducible tests, and remaining deployment checks.
