# Tracker Telemetry and Time Alignment Plan

<!-- SCOPE: Planned Jetson timing, attitude, diagnostics, replay, and paired Type C integration work. -->
<!-- DOC_KIND: explanation -->
<!-- DOC_ROLE: working -->
<!-- READ_WHEN: Implementing or reviewing source-timestamped tracking telemetry. -->
<!-- SKIP_WHEN: Looking for commands supported by currently deployed firmware. -->
<!-- PRIMARY_SOURCES: cv_auto_aim_interfaces/msg/GimbalAttitude.msg, cv_auto_aim_interfaces/msg/Target.msg, cv_auto_aim_interfaces/msg/AimSolution.msg, yolo_cpp/src/stm32_serial_bridge_node.cpp, yolo_cpp/src/aim_geometry.cpp, yolo_cpp/src/camera_backend.cpp -->

## Quick Navigation

| Need | Reference |
| --- | --- |
| Current implementation and evidence | [Local validation report](../validation/autoaim-local-upgrade.md) |
| Current deployment procedure | [Manual checks](../validation/autoaim-manual-checks.md) |
| Current message semantics | [Interfaces](../../yolo_cpp/docs/interfaces.md) |
| Firmware counterpart | [Controls Type C repository](https://gitlab.com/nyu-robomaster/controls_embedded/controls_typec), `docs/vision-telemetry-plan.md` on `codex/autoaim-slew-reference-fix` |

## Agent Entry

| Field | Guidance |
| --- | --- |
| Status | Implementation plan only; proposed messages, settings, protocol extensions, and tools below are not implemented by this document. |
| Branch | Continue `codex/autoaim-pose-tracking`; inspected runtime baseline `302bed69`. |
| Firmware baseline | `9553b7f` on `codex/autoaim-slew-reference-fix`; not a statement about installed hardware. |
| Paired firmware plan | Type C documentation commit `0908ef8`; implementation stages T1-T3 remain pending. |
| Ownership | Jetson owns clock fitting, calibration, transforms, tracking, recording, and deployment policy. MCU owns sampling, transport, actuator control, and operator interlocks. |
| Contract owner | The Type C counterpart owns proposed wire requirements. Freeze one versioned schema and cross-language byte fixtures there before implementing either encoder or decoder. |

## Decision and Current Gaps

Fix measurement timing and frame semantics before tuning filters or adding encoder/IMU fusion. Preserve the movement packet and control laws. Deliver independently reviewable commits on the two existing branches.

The current bridge publishes receive time, estimated measurement time, validity, and `source_timestamped=false`. It discards older SP samples in a USB read batch because their actual times are unknown. Tracker and solver use bounded attitude history, but currently interpolate Euler angles; the full SP quaternion is not retained in the ROS message. Native Basler capture uses host receipt time. DeepStream PTS mapping does not by itself establish exposure time.

Target/AimSolution already expose measurement age, robot identity, uncertainty, and solver status. Extend these through a separate diagnostic record rather than copying every estimator field into the control message. MCU free-yaw mode can substitute encoder-relative yaw into the legacy attitude path, which lacks an explicit reference-frame identifier.

Type C must audit its timebase and sample ownership first. Its DWT microsecond helper assembles a result using 32-bit arithmetic before returning 64 bits, and pose/gyro values reach the packet through different paths. A timestamp added only during packet assembly would not establish simultaneous sampling.

## Delivery Sequence and Commit Boundaries

Each row is planned work. Include focused tests in its implementation commit; use J6 for integration evidence.

| Step | Repository and proposed commit subject | Deliverable and exit gate |
| --- | --- | --- |
| J1 | CV: `feat(telemetry): record timing provenance and tracking decisions` | Bounded recorder and diagnostics for existing messages; explicit receipt-based timing, frame IDs, effective settings, and rejection reasons. |
| T1 | Type C: `fix(telemetry): provide monotonic coherent sensor snapshots` | Audited integer timebase and sample snapshot API; wrap/preemption and frame-direction fixtures pass. No control-law changes. |
| T2 / J2 | Type C: `feat(telemetry): add versioned read-only telemetry exchange`; CV: `feat(bridge): decode timestamped telemetry and synchronize clocks` | Shared schema/fixtures, session exchange, clock mapping, and parser routing; mixed-protocol, compatibility, reset, drift, and no-movement tests pass. |
| J3 | CV: `feat(camera): preserve acquisition time and timing uncertainty` | Capability report and hardware time mapping where supported; honest native/DeepStream fallback; exposure semantics preserved. |
| J4 | CV: `feat(tracking): align quaternion history with camera measurements` | Full quaternion history, calibrated transforms, bounded SLERP, per-frame timing gates; encoders remain separate joint measurements. |
| T3 / J5 | Type C: `feat(telemetry): report controller acceptance and applied reference`; CV: `feat(diagnostics): correlate observations with controller response` | Received/accepted/limited/measured signals and reasons appear together without claiming exact acknowledgment of an unnumbered command. |
| J6 | CV: `test(tracking): validate telemetry faults and record deployment gates` | Production-node fault tests, baseline comparisons, replay tools, and commissioning checklist with explicit unverified items. |

J1 can precede firmware work. J3 can proceed against recorded camera metadata. J4 requires validated frame conventions; synchronized operation requires T2/J2 and a qualified camera time source. T3/J5 add diagnostics without retuning PID or slew parameters.

## Clock and Camera Contract

- Keep raw MCU microseconds, camera ticks, host monotonic receipt time, and ROS publication time separately. Never subtract unrelated clocks directly.
- Fit `host_monotonic = scale * device_time + offset` per device/session. MCU synchronization uses host-send, MCU-receive, MCU-send, and host-receive times. Reject congested/outlier exchanges and constrain drift. Report conservative uncertainty including path asymmetry and timestamp placement; round-trip time does not prove symmetric USB delay.
- Reset fitting and attitude/tracker history after reboot/session change, backward device time, host suspend, or ROS clock discontinuity. Reject previous-session samples and re-establish quality before synchronized solutions resume.
- Treat `source_timestamped` and `clock_synchronized` as separate facts. Add timing source, sync state, uncertainty, session, and sequence metadata; receipt-only data must never be labeled source-timestamped.
- With valid source times, ingest ordered samples from each USB batch into bounded history instead of keeping only the last packet. Reject duplicates, previous-session and backward samples; receipt of a batch must not refresh historical sample age.
- Query the actual Basler model for timestamp event, tick frequency, shutter type, exposure duration, and reset behavior. Use exposure midpoint for global shutter only when its start/end relationship is known. Rolling shutter requires row-time uncertainty.
- Use supported clock-latch/trigger facilities where available. Arrival-only fitting cannot prove absolute exposure alignment. If the budget cannot be met, retain degraded diagnostics and evaluate a camera exposure/trigger edge captured by the MCU as a later hardware option.
- Preserve frame sequence and timing provenance through detector, tracker, processor, and solver. Preserve the last genuine 3D measurement time during 2D-only updates and coasting.

## Attitude and Tracker Contract

Retain a normalized quaternion with explicit direction and frame IDs. ROS uses xyzw ordering; SP uses wxyz. Test ordering, active/passive rotation, handedness, camera mount, zero offsets, and signs with nonzero yaw, pitch, and roll together. Current control Euler conventions do not make the wire quaternion a standard ROS camera pose.

Use sign-continuous quaternions (`q` and `-q` represent the same rotation) and SLERP between valid timestamped samples. Never interpolate across IMU resets, frame changes, oversized gaps, or unhealthy samples. Any short gyro extrapolation needs an explicit age/error bound; endpoint holding must retain its true age. Tracker and solver use the same tested history implementation.

Keep encoders separate with individual acquisition age and validity. They cannot establish chassis attitude or remove global yaw drift alone. Defer fusion until synchronized recordings demonstrate an observable error and trustworthy chassis attitude is available. Do not treat `ST.real_vx/real_vy/real_wz` as measured odometry: the inspected sentry producer derives them from commands. Chassis translation compensation needs a separately validated source.

## Diagnostics and Recording

| Group | Required information |
| --- | --- |
| Timing | Frame/sample sequence, device/host times, source, offset/drift, uncertainty, sample age, bracket span, queue age, drops, reconnect/reset count. |
| Perception/association | Image and calibration identity, model hash, corners/pose quality, robot/plate IDs, selected slot, measurement/coast state, association score and switch reason. |
| Estimation | Robot center/velocity, phase/rate, uncertainty, accepted/rejected observations and innovation/gating diagnostics. Distinguish model uncertainty from calibrated statistical guarantees. |
| Controller | Requested solution, received and accepted target, limited reference, actual attitude/rates, encoders, mode, acceptance reason and sensor health. |
| Run manifest | Both software revisions, model/calibration hashes, effective non-secret settings, camera metadata and parser version. Exclude credentials and private keys. |

Use a bounded asynchronous recorder with disk quota and dropped-record counters. Preserve original images for reproducible detector replay where storage permits; label lossy video as visual evidence rather than identical detector input. Join by session/sequence/time. Prefer rosbag, a compact manifest, and offline plots; no new web application is needed.

## Compatibility and Preview Behavior

| Pairing | Required behavior |
| --- | --- |
| Existing firmware + updated Jetson | Explicitly degraded legacy reception; no speculative extension probes to an unknown parser. |
| Updated firmware + existing Jetson/bridge | Extension defaults off; existing SP/ST/SX lengths and behavior remain unchanged. |
| Both updated | Operator selects the verified extension-capable profile; the sole serial owner negotiates a session before using source times. |
| Extension/sync failure | Invalidate synchronized solutions and report why. No silent receipt-based fallback during an active synchronized session. |

Proposed Jetson settings distinguish telemetry protocol, read-only synchronization exchange, timing-quality requirements, and recording controls. They are design proposals, not current env options. Document migration before changing defaults.

Preserve `transmit_enable:=false` and `enable_fire_recommendation:=false` for commissioning. If sync is separately enabled, preview may write only recognized telemetry/session requests: never SP movement, SX takeover, AT tuning, or firing commands. A passive profile still produces zero serial writes. Tests must prove both contracts. Sync requests cannot refresh MCU movement watchdogs or change control mode.

## Validation and Promotion Gates

1. **Protocol:** C-generated bytes decode identically in C++ and the Python bridge. Test every split boundary, concatenation, CRC/length/version errors, embedded legacy headers, duplicates, unknown messages, and session changes in direct USB and multiplexed paths.
2. **Timing:** Inject known offset/drift, asymmetric delay, batching, drops, reordering, MCU/camera reset, and host clock changes. Compare with simulator truth and ensure uncertainty covers the injected delay model. Old measurements with fresh receipt times must remain old.
3. **Geometry:** Test stationary-plate camera motion, yaw wrap, q/-q, combined roll/pitch/yaw, encoder-only changes, and frame changes. Both consumers must agree on capture orientation and reject insufficient history.
4. **Tracking:** Re-run lone stationary plate, non-spinning multi-plate, spinning, distractor, occlusion, and far/noisy rejection cases through production nodes and the pinned Type C slew fixture. Preserve good-input baseline bounds; report command availability so rejection cannot masquerade as accuracy.
5. **Authority:** TX disabled through reconnect/restart/sync recovery produces zero movement/fire packets. Passive mode produces zero writes. Invalid sync never reaches a control handler; RC/timeout gates remain effective.
6. **Performance:** Record p50/p95/p99/max latency/sample age, queue depths/drops, recorder overhead, MCU execution time and USB drops. Start at the existing 100 Hz attitude cadence and 10-20 Hz diagnostics; increase only with measured benefit and adequate control-task margin.
7. **Hardware:** Later, with explicit operator authorization, record a stationary target during supervised gimbal motion and compare legacy/synchronized processing on the same data. A camera-less Jetson validates transport/builds, not optical accuracy.

An initial timing-only budget is 0.2 degrees at 100 degrees/second: total camera-to-attitude alignment uncertainty should be at most 2 ms for a qualified sample. Evaluate against measured angular rate while retaining a hard freshness bound. Report p95/p99/max and qualification rate. This is a proposed target, not achieved accuracy or a total tracking-error guarantee.

Use `tools/run-local-tracking-tests.sh` for Mac-hosted ROS tests and rebuild both active packages after message changes. Firmware validation uses `just verify-consolidation`, `just build sentry`, and the build matrix after shared changes. Regenerate the CV reference fixture against the published Type C source revision when its relevant control code changes.

Deploy the updated Jetson decoder/bridge in legacy preview first. Obtain separate authorization to flash/restart the paired firmware, qualify telemetry in preview, and then perform separately authorized motion tests. Save the previous firmware artifact and Jetson revision. Rollback disables the extension and restores the compatible pair without enabling TX. Env-only changes use `cv restart`; source/message changes require `uv build yolo_cpp`; firmware changes require a build and authorized reflash.

## Source Rationale

- [Sumedh's CV learnings](https://forums.arc-robotics.org/t/sumedh-s-various-cv-learnings/247): recording, concurrency, and stationary-target camera-motion checks motivate diagnostics and validation.
- [Tongji gimbal implementation](https://github.com/TongjiSuperPower/sp_vision_25/blob/main/io/gimbal/gimbal.cpp): quaternion history and SLERP are useful references; its serial reader stamps host reception, which does not solve MCU synchronization.
- [Basler data chunks](https://docs.baslerweb.com/data-chunks) and [device timestamps](https://docs.baslerweb.com/timestamp): event semantics and frequency depend on camera model; hardware ticks require clock mapping.
- Session recovery, clock fitting, acceptance gates, and rollout here are project design decisions, not claims that upstream teams use this exact protocol.

## Maintenance

Update after each stage with its commit, tests, schema version, and outstanding hardware evidence. Keep supported operator commands in the manual guide. Verify linked sources and paired contract versions before promotion.

**Last Updated:** 2026-10-01
