# Pipeline

Related pages: [Overview](./overview.md) | [Nodes](./nodes.md) | [Interfaces](./interfaces.md) | [Legacy Pipeline Notes](./legacy-pipeline-notes.md) | [Launch and Config](./launch-and-config.md) | [Development Notes](./development-notes.md)

The live `yolo_cpp` pipeline is split into small ROS 2 stages so each part can be tuned or swapped without rewriting the whole runtime.

## Camera to Detection

The detector stage pulls frames through the shared camera backend. In practice that means RealSense or Basler frames are normalized into a common `FramePacket`, preprocessed, and sent through either the native TensorRT path or the DeepStream-backed path.

The detector publishes `cv_auto_aim_interfaces/msg/Armors` on `detector/armors`. Each message carries the frame metadata plus the candidate armor boxes, confidence, image size, optional OBB angle data, armor corners, and optional camera-frame PnP pose. If calibrated `CameraInfo` is not available yet, the detector can fall back to configured FOV intrinsics so the 3D path can still be brought up.

## Detection to Tracking

The tracker subscribes to `detector/armors` and turns frame-by-frame detections into stable tracks. It keeps enough state to smooth over short gaps, avoid target flicker, and choose a single target when multiple candidates are visible. When PnP pose is available, association uses a 3D distance gate; otherwise it falls back to the previous 2D IoU and center-distance path.

It publishes two outputs:

- `tracker/tracked_armors` for the full tracking state
- `tracker/target` for the current selected target

That split is useful in practice. The first topic is better for debugging and visualization, while the second is the compact 2D-compatible control signal.

## Tracking to Target State

`ArmorProcessorNode` subscribes to `tracker/tracked_armors`, transforms the selected target into the configured processor frame, smooths pose and velocity, and publishes `processor/target`. It also publishes `debug/spin_info` and `processor/marker` for small-gyro behavior and visualization. The processor preserves the selected 2D target even when 3D pose is temporarily unavailable, so the solver can fall back cleanly.

## Target State to Aim Output

In the full 3D launches, the solver subscribes to `processor/target`. When pose is present, it predicts latency plus projectile flight time, applies target velocity and optional gravity compensation, and publishes yaw/pitch. When pose is missing, it falls back to the original bounding-box FOV projection. When no target is available, it can publish a safe zero-angle fallback instead.

The solver publishes:

- `bounding_box`
- `bounding_box_angles`
- `override`
- `obb_angle`
- `suggest_fire`
- `aim_solution`

These topics are the handoff between perception and downstream control.

## STM Bridge

In the full stack launched by `stm.launch.py`, the STM bridge sits at the edge of the pipeline. It handles serial transport to the controller and also listens for feedback such as enemy color, bullet speed, and gimbal attitude.

STM attitude feedback on `attitude_data` is treated as radians inside ROS. With `solver_output_mode:=absolute`, `AimSolverNode` adds the STM yaw/pitch feedback to the solved offsets before publishing `bounding_box_angles`. The serial bridge then converts those radians to degrees by default with `transmit_angles_in_degrees:=true` before packing the STM TX frame.

## Combined Path

The package also still contains a combined `PerceptionNode`. That node folds camera, detection, and aiming behavior into one component. It is useful as a reference path and for zero-copy experiments, but the split detector -> tracker -> processor -> solver path is the main runtime to understand first.
