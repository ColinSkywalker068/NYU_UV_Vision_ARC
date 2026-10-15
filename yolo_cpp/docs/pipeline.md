# Pipeline

Related pages: [Overview](./overview.md) | [Nodes](./nodes.md) | [Interfaces](./interfaces.md) | [Legacy Pipeline Notes](./legacy-pipeline-notes.md) | [Launch and Config](./launch-and-config.md) | [Development Notes](./development-notes.md)

The live `yolo_cpp` pipeline is split into small ROS 2 stages so each part can be tuned or swapped without rewriting the whole runtime.

## Camera to Detection

The detector stage pulls frames through the shared camera backend. In practice that means RealSense or Basler frames are normalized into a common `FramePacket`, preprocessed, and sent through either the native TensorRT path or the DeepStream-backed path.

The detector publishes `cv_auto_aim_interfaces/msg/Armors` on `detector/armors`. Each message carries the frame metadata plus the candidate armor boxes, confidence, image size, optional OBB angle data, armor corners, and optional camera-frame PnP pose. PnP requires valid camera intrinsics matching the source image dimensions and usable armor corners. Explicit provisional intrinsics can support commissioning, but are not a completed camera calibration.

## Detection to Tracking

The tracker subscribes to `detector/armors` and turns frame-by-frame detections into stable tracks. It keeps enough state to smooth over short gaps, avoid target flicker, and choose a single target when multiple candidates are visible. When PnP pose is available, association uses a 3D distance gate; otherwise it falls back to the previous 2D IoU and center-distance path.

It publishes two outputs:

- `tracker/tracked_armors` for the full tracking state
- `tracker/target` for the current selected target

The first topic retains all plate tracks, including raw camera-frame poses for diagnostics. The second topic identifies the selected target. Converting a camera-frame pose into the inertial tracking frame requires capture-aligned gimbal attitude; a detected plate can remain selected while its inertial pose is unavailable.

## Tracking to Target State

`ArmorProcessorNode` subscribes to `tracker/tracked_armors`, uses the tracker's filtered pose and velocity in the configured processor frame, and publishes `processor/target`. It also publishes `debug/spin_info` and `processor/marker`. A single stationary plate uses its direct pose without requiring a confirmed multi-plate robot model. Confirmed robot state can support spinning or multiple-plate prediction. The processor preserves a selected 2D target for diagnostics when the inertial pose is unavailable; this does not make an absolute command valid.

## Target State to Aim Output

In the full 3D launches, the solver subscribes to `processor/target`. With valid pose, capture-aligned and current attitude, and fresh measurements, it predicts latency plus projectile flight time, applies target velocity and optional gravity compensation, and publishes an atomic angle/rate solution. With `use_3d_solution:=true`, missing pose does not automatically fall back to bounding-box aiming. Missing/stale attitude and capture mismatch are reported before missing 3D pose because absent telemetry can itself prevent the world-frame pose from being produced. Every rejected solution remains invalid and cannot recommend fire.

The solver publishes:

- `bounding_box`
- `bounding_box_angles`
- `override`
- `obb_angle`
- `suggest_fire`
- `aim_solution`

These topics are the handoff between perception and downstream control.

## STM Bridge

In the full Basler field stack launched by `basler.launch.py`, the STM bridge sits at the edge of the pipeline. It handles serial transport to the controller and also listens for feedback such as enemy color, bullet speed, and gimbal attitude. `stm.launch.py` is only a compatibility alias for that launch file.

The bridge publishes decoded controller attitude on `gimbal/attitude`, in radians, with validity and receipt-time information. Absolute aiming uses capture-aligned camera-to-inertial geometry rather than simply adding the latest yaw/pitch to old pixel offsets. Host receipt time does not establish MCU source-clock synchronization. The bridge packs radians by default with `transmit_angles_in_degrees:=false`, matching the controller contract.

`CV_STM_SERIAL_PORT` in the saved `services/stm-stack.env` must identify the connected board's USB CDC data port. A stable `/dev/serial/by-id/` path belongs to one board; changing boards can leave the previous ID absent. ST-LINK provides SWD debugging and does not replace that data link. After correcting the saved port, restart CV and require fresh decoded attitude. See the [config guide](../../docs/project/stm-stack-config.md). No firmware flash is required for an existing compatible USB telemetry implementation.

## Combined Path

The package also still contains a combined `PerceptionNode`. That node folds camera, detection, and aiming behavior into one component. It is useful as a reference path and for zero-copy experiments, but the split detector -> tracker -> processor -> solver path is the main runtime to understand first.
