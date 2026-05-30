# Documentation Standards

<!-- SCOPE: Reference rules for generated project documentation only. Defines structure, writing, linking, verification, and maintenance requirements. -->
<!-- DOC_KIND: reference -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when creating, updating, auditing, or validating project documentation. -->
<!-- SKIP_WHEN: Skip when you only need a domain-specific project fact. -->
<!-- PRIMARY_SOURCES: docs/README.md, docs/principles.md, AGENTS.md -->

## Quick Navigation

| Need | Read |
|------|------|
| Documentation map | [README.md](README.md) |
| Development principles | [principles.md](principles.md) |
| Agent entry model | [../AGENTS.md](../AGENTS.md) |
| Active runtime docs | [../yolo_cpp/docs/overview.md](../yolo_cpp/docs/overview.md) |

## Agent Entry

| Field | Guidance |
|-------|----------|
| Purpose | Canonical standards for repository documentation and generated markdown. |
| Read when | You are creating, editing, or auditing project documentation. |
| Skip when | You only need a concrete runtime fact from source or package docs. |
| Canonical | Yes, for documentation structure and validation rules. |
| Read next | `principles.md` for rationale, then the specific doc being edited. |
| Primary sources | `docs/README.md`, `docs/principles.md`, `AGENTS.md`, and the current repo tree. |

## Quick Reference

| # | Category | Requirement |
|---|----------|-------------|
| 1 | Header | Include `SCOPE` near the top of every generated markdown doc. |
| 2 | Header | Include `DOC_KIND` with one of `index`, `reference`, `how-to`, `explanation`, or `record`. |
| 3 | Header | Include `DOC_ROLE` with one of `canonical`, `navigation`, `working`, or `derived`. |
| 4 | Header | Include `READ_WHEN` to tell agents when to load the doc. |
| 5 | Header | Include `SKIP_WHEN` to prevent unnecessary reads. |
| 6 | Header | Include `PRIMARY_SOURCES` with verified source paths. |
| 7 | Header | Keep metadata factual and short. |
| 8 | Header | Put `SCOPE` within the first 10 lines. |
| 9 | Navigation | Include `Quick Navigation` in generated docs. |
| 10 | Navigation | Include `Agent Entry` in generated docs. |
| 11 | Navigation | Include `Maintenance` in generated docs. |
| 12 | Navigation | Use tables for maps, ownership, and command summaries. |
| 13 | Navigation | Link to canonical sources instead of repeating long content. |
| 14 | Navigation | Put current active docs before legacy docs. |
| 15 | Navigation | Keep root docs short enough for agent context. |
| 16 | Links | Verify every relative link before publishing. |
| 17 | Links | Link to existing repository paths only. |
| 18 | Links | Prefer relative links for repository files. |
| 19 | Links | Avoid links to generated build artifacts unless the artifact is intentionally tracked. |
| 20 | Links | Use source paths for implementation detail instead of copying code. |
| 21 | Links | Do not link to private or local-only absolute paths. |
| 22 | Links | Update inbound navigation when a canonical doc moves. |
| 23 | Content | State only facts supported by current repo sources. |
| 24 | Content | Separate active runtime facts from legacy or prototype notes. |
| 25 | Content | Keep hardware-dependent claims conditional. |
| 26 | Content | Treat optional backends as optional unless the build files make them required. |
| 27 | Content | Keep command examples as single-line inline commands. |
| 28 | Content | Avoid implementation snippets in root governance docs. |
| 29 | Content | Keep docs in English. |
| 30 | Content | Avoid stale dates except the maintained `Last Updated` field. |
| 31 | Project | Identify `yolo_cpp/` as the active C++ ROS 2 runtime package. |
| 32 | Project | Identify `cv_auto_aim_interfaces/` as the custom message package. |
| 33 | Project | Identify `legacy/` as historical or prototype material unless a task targets it. |
| 34 | Project | Describe Docker and services as operational support, not application logic. |
| 35 | Project | Describe `yolo_cpp/models/` as active runtime model assets and `legacy/models/` as archived model variants. |
| 36 | Commands | Prefer `uv docker` for container entry in repo-facing docs. |
| 37 | Commands | Prefer `uv build yolo_cpp` for active stack builds. |
| 38 | Commands | Prefer `uv run yolo_cpp basler.launch.py` for Basler field stack launch guidance. |
| 39 | Commands | Prefer `uv run yolo_cpp realsense.launch.py` for RealSense-focused launch guidance. |
| 40 | Commands | Include native `colcon` commands only when ROS 2 dependencies are assumed. |
| 41 | Verification | Match commands against `services/uv.sh` and package build files. |
| 42 | Verification | Match pipeline claims against `yolo_cpp/docs/pipeline.md`. |
| 43 | Verification | Match node claims against `yolo_cpp/docs/nodes.md`. |
| 44 | Verification | Match message claims against `cv_auto_aim_interfaces/msg/` and `yolo_cpp/docs/interfaces.md`. |
| 45 | Verification | Match container claims against `Docker/` and `services/`. |
| 46 | Quality | Publishable docs contain no unresolved placeholders. |
| 47 | Quality | Publishable docs contain no drafting markers. |
| 48 | Quality | Files end with a single POSIX newline. |
| 49 | Quality | Headings use stable title case where practical. |
| 50 | Quality | Avoid duplicate canonical facts. |
| 51 | Quality | Keep root files focused on routing and rules. |
| 52 | Standards | Use ISO/IEC/IEEE 29148:2018 style for verifiable requirements. |
| 53 | Standards | Use Diataxis separation for tutorials, how-to guides, reference, and explanation. |
| 54 | Standards | Use arc42 concepts for architecture context, building blocks, runtime views, and tradeoffs. |
| 55 | Standards | Use C4-style boundary thinking before component detail. |
| 56 | Standards | Use ADR-style records for durable architecture choices. |
| 57 | Maintenance | Include update triggers. |
| 58 | Maintenance | Include verification checks. |
| 59 | Maintenance | Include `Last Updated` with an exact date. |
| 60 | Maintenance | Revise navigation when paths, commands, or ownership change. |
| 61 | Agent Files | Keep `AGENTS.md` canonical. |
| 62 | Agent Files | Keep `CLAUDE.md` as a thin `@AGENTS.md` import stub. |
| 63 | Agent Files | Do not duplicate AGENTS content inside provider-specific stubs. |
| 64 | Agent Files | Put path-specific Claude guidance in `.claude/rules/` only if that directory exists or is being created for the task. |

## 1. Header Contract

| Requirement | Guidance |
|-------------|----------|
| Metadata | Generated markdown documents use the six required HTML comments near the top. |
| Scope | `SCOPE` describes what belongs in the document and what does not. |
| Kind | `DOC_KIND` identifies the information type so readers know how to use it. |
| Role | `DOC_ROLE` identifies whether the doc owns facts, routes readers, tracks work, or derives from another file. |
| Sources | `PRIMARY_SOURCES` lists verified paths that support the claims. |
| Read protocol | `READ_WHEN` and `SKIP_WHEN` must be concrete enough for an agent to decide quickly. |
| First-screen utility | The reader should know whether the doc is relevant before scrolling past the opening sections. |
| Comment discipline | Required metadata comments are allowed; large hidden prose blocks are not useful in published docs. |
| Path format | Source paths in metadata use repository-root-relative paths. |
| Stability | Header values should not include temporary branch names or local machine details. |

## 2. Canonical Ownership

| Requirement | Guidance |
|-------------|----------|
| Single owner | Keep one canonical owner per topic. |
| Root model | `AGENTS.md` owns shared agent guidance; `CLAUDE.md` imports it. |
| Runtime model | `yolo_cpp/docs/` owns active runtime explanations. |
| Interface model | `cv_auto_aim_interfaces/msg/` owns ROS message structure; docs explain how to read it. |
| Legacy model | Legacy content remains discoverable but should not appear as the first source for active C++ runtime work. |
| Command model | `services/uv.sh` owns wrapper behavior; docs summarize its stable entry points. |
| Container model | `Docker/` and `services/` own operational launch environment details. |
| Planning notes | Planning docs under `docs/` may describe direction but should not override source or launch files. |
| Source precedence | When prose and code disagree, verify code and update the prose owner. |
| Derived docs | Derived stubs should identify their source and avoid carrying copied sections. |

## 3. Navigation Structure

| Requirement | Guidance |
|-------------|----------|
| Quick Navigation | The first major section routes common needs to verified files. |
| Agent Entry | The second major section tells agents why, when, and how to read the document. |
| Reading order | Current sources appear before historical sources. |
| Tables | Use tables for path maps and command maps so the document is scannable. |
| No dead ends | Every navigation doc points to the next useful source. |
| Locality | Link to the nearest canonical source for the reader's task. |
| Legacy labeling | Historical references should be labeled so readers do not mistake them for active runtime guidance. |
| Package grouping | Keep `yolo_cpp` docs grouped together when describing the active stack. |
| Operations grouping | Keep Docker, service, and runbook links together when describing bring-up. |
| Reader intent | Navigation labels should name tasks, not just file names. |

## 4. Link Hygiene

| Requirement | Guidance |
|-------------|----------|
| Relative links | Use relative markdown links for repository files. |
| Existing paths | Check the path exists before linking. |
| Source paths | Use inline source path references for directories and files that are not linked. |
| Renames | Update root and docs navigation when files move. |
| External references | Use external links only when they add authority that the repository cannot provide. |
| Directory links | Link to directories only when the renderer and review tool handle them predictably. |
| Anchor links | Use anchors sparingly because heading changes break them quietly. |
| Absolute paths | Avoid local absolute paths in committed docs. |
| Image links | Verify image assets exist before linking or embedding them. |
| Generated outputs | Do not make generated cache output part of canonical navigation unless it is intentionally reviewed. |

## 5. Project-Fact Accuracy

| Requirement | Guidance |
|-------------|----------|
| Active packages | Name `yolo_cpp` and `cv_auto_aim_interfaces` as the active ROS 2 packages. |
| Runtime language | Describe active runtime code as C++17. |
| ROS version | Describe the target runtime as ROS 2 Humble when discussing container or launch workflows. |
| Optional backends | Mark TensorRT, CUDA, DeepStream, GStreamer, RealSense, and Basler Pylon as environment-dependent where applicable. |
| Message contracts | Check package messages before documenting topics or fields. |
| Topic flow | Match detector, tracker, processor, solver, and bridge claims to package docs or source. |
| Unit claims | Check angle and unit claims at solver and serial bridge boundaries. |
| Launch defaults | Confirm launch behavior in `yolo_cpp/launch/` before documenting defaults. |
| Build options | Confirm optional compile definitions in CMake before documenting backend support. |
| Model assets | Treat `yolo_cpp/models/` as active runtime assets, not as the training source of truth. |

## 6. Command Documentation

| Requirement | Guidance |
|-------------|----------|
| Wrapper commands | Use `uv` commands when explaining the team workflow. |
| Native commands | Include native `colcon` commands for ROS-ready environments. |
| Launch commands | Tie launch guidance to `basler.launch.py`, compatibility alias `stm.launch.py`, `realsense.launch.py`, or `zerocopy.launch.py`. |
| Test commands | Prefer package-scoped tests for narrow verification. |
| Formatting commands | Point to pre-commit for repository-wide mechanical checks. |
| Host context | Say whether a command runs on the host, in the container, or in a ROS-ready shell when that matters. |
| Hardware context | Identify when commands require devices, display forwarding, or NVIDIA runtime support. |
| Cleanup context | Describe cleanup commands only when the source script supports them. |
| Service context | Treat systemd service commands as operational tasks with host-side effects. |
| Windows context | Do not imply active ROS runtime support on native Windows unless a verified workflow exists. |

## 7. Writing Style

| Requirement | Guidance |
|-------------|----------|
| Direct prose | State facts directly and avoid promotional language. |
| Short sections | Use compact paragraphs and tables rather than long narrative blocks. |
| English | Keep root and generated docs in English. |
| No filler | Remove generic statements that do not guide a reader to action. |
| Terminology | Use project terms consistently: detector, tracker, processor, solver, STM bridge. |
| Specific nouns | Prefer concrete package, node, topic, and path names over vague labels. |
| Neutral tone | Describe constraints and risks directly. |
| Short cells | Keep table cells compact enough to scan in code review. |
| No speculation | Mark uncertain behavior as something to verify in source rather than presenting it as fact. |
| Exact dates | Use exact dates when dates are relevant. |

## 8. Standards Alignment

| Standard | Application |
|----------|-------------|
| ISO/IEC/IEEE 29148:2018 | Requirements are clear, verifiable, and bounded. |
| Diataxis | Explanations, how-to material, reference tables, and tutorials are separated. |
| arc42 | Architecture docs cover context, building blocks, runtime behavior, and tradeoffs. |
| C4 Model | System and container boundaries appear before components. |
| ADR | Durable decisions record context, decision, consequences, and status. |
| ROS 2 package conventions | Package docs reflect `package.xml`, `CMakeLists.txt`, launch files, and message definitions. |
| CMake conventions | Build claims follow configured targets and dependency discovery. |
| Docker conventions | Container docs distinguish image build, compose orchestration, and runtime execution. |
| Pre-commit conventions | Mechanical formatting and hook setup belong in tooling docs, not repeated root instructions. |
| Review conventions | Docs should make reviewers able to verify claims with nearby repository sources. |

## 9. Generated Documentation Quality

| Requirement | Guidance |
|-------------|----------|
| Publishable output | Generated docs are complete enough to commit as-is. |
| No raw markers | Remove unresolved template tokens and drafting markers. |
| Stable date | Use exact dates in `Last Updated`. |
| POSIX newline | End every file with one final newline. |
| No root code examples | Root governance docs should link to source and package docs instead of embedding code. |
| No template leakage | Remove scaffolding notes and generator-only instructions from published docs. |
| No unresolved claims | Omit unsupported facts rather than filling them with drafting labels. |
| No noisy diffs | Avoid reflowing unrelated sections during targeted documentation edits. |
| No stale ownership | Update owners when a file or package becomes historical. |
| No hidden dependency | State required source files in `PRIMARY_SOURCES` when they support the document. |

## 10. Maintenance Workflow

| Requirement | Guidance |
|-------------|----------|
| Update triggers | List when a document must change. |
| Verification | List checks the maintainer can perform. |
| Scope discipline | Do not combine unrelated doc maintenance with feature edits unless needed for correctness. |
| Reviewability | Keep changes localized so diffs are easy to inspect. |
| Drift control | Update routing docs when canonical documents move. |
| Trigger discipline | Update docs when commands, package boundaries, topics, messages, or hardware assumptions change. |
| Verification discipline | Record checks that a maintainer can actually perform. |
| Date discipline | Change `Last Updated` only for meaningful content updates. |
| Ownership discipline | Route semantic changes to the document that owns the topic. |
| Audit discipline | Treat broken links, stale commands, and false path claims as documentation defects. |

## 11. Verification Matrix

| Claim Type | Verify Against |
|------------|----------------|
| Runtime package ownership | `yolo_cpp/package.xml`, `yolo_cpp/CMakeLists.txt`, `yolo_cpp/docs/overview.md` |
| Interface ownership | `cv_auto_aim_interfaces/package.xml`, `cv_auto_aim_interfaces/CMakeLists.txt`, `cv_auto_aim_interfaces/msg/` |
| Pipeline flow | `yolo_cpp/docs/pipeline.md` |
| Node behavior | `yolo_cpp/docs/nodes.md` |
| Launch behavior | `yolo_cpp/docs/launch-and-config.md`, `yolo_cpp/launch/` |
| Container behavior | `Docker/docker-compose.yml`, `Docker/README.md`, `services/uv.sh`, `services/start-container.sh` |
| Documentation map | `docs/README.md`, repository file tree |
| Build dependencies | `yolo_cpp/CMakeLists.txt`, `cv_auto_aim_interfaces/CMakeLists.txt`, package manifests |
| Test coverage | `yolo_cpp/test/` and package test declarations |
| Pre-commit guidance | `docs/Pre-commit.md` and repository hook configuration |
| Roadmap claims | Planning docs under `docs/` and active implementation state |
| Legacy claims | `legacy/` and archived docs under `legacy/docs/` |
| Model asset claims | `yolo_cpp/models/`, `legacy/models/`, and the launch or config files that reference those assets |
| Camera backend claims | `yolo_cpp/src/camera_backend.cpp`, `yolo_cpp/include/yolo_cpp/camera_backend.hpp`, and launch parameters |
| Detector backend claims | `yolo_cpp/src/detector_backend.cpp`, `yolo_cpp/src/deepstream_detector.cpp`, and CMake feature detection |
| Solver claims | `yolo_cpp/src/aim_solver_node.cpp` and solver-facing package docs |
| Serial protocol claims | `yolo_cpp/src/stm32_serial_bridge_node.cpp`, `yolo_cpp/src/seasky_protocol.cpp`, and related headers |
| Service claims | `services/` scripts and service unit files |

## 12. Root Agent File Rules

| File | Rule |
|------|------|
| `AGENTS.md` | Keep shared agent instructions, project map, critical rules, and command overview here. |
| `CLAUDE.md` | Keep only `@AGENTS.md` plus Claude Code-specific command, memory, and rule-loading notes. |
| Other provider stubs | Derive from `AGENTS.md` if introduced. |
| Path-specific rules | Use supported harness mechanisms and verify their directories before linking them. |
| Size budget | Keep `AGENTS.md` below 200 lines and `CLAUDE.md` below 50 lines. |
| Shared rules | Put shared rules in `AGENTS.md`, not in provider-specific stubs. |
| Harness deltas | Limit harness deltas to commands, memory behavior, and provider-specific loading mechanics. |
| Import integrity | `CLAUDE.md` contains exactly one `@AGENTS.md` import line. |
| Maintenance checks | Validate line counts, import count, links, markers, and final newlines after edits. |

## Maintenance

**Update Triggers:**
- The shared docs-quality contract changes.
- Repository documentation ownership changes.
- Active runtime commands, packages, launches, or interface boundaries change.
- A new documentation standard is adopted by the project.

**Verification:**
- [ ] Header contract is present in generated docs.
- [ ] Root files remain canonical-stub compliant.
- [ ] Internal links and referenced paths resolve.
- [ ] Quick Reference still contains at least 60 concrete requirements.
- [ ] Standards references remain relevant to this repository.

**Last Updated:** 2026-05-30
