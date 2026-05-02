# Nodes

Related pages: [Overview](./overview.md) | [Pipeline](./pipeline.md) | [Interfaces](./interfaces.md) | [Legacy Pipeline Notes](./legacy-pipeline-notes.md) | [Launch and Config](./launch-and-config.md) | [Development Notes](./development-notes.md)

## CameraNode

`CameraNode` is the camera-facing publisher. It wraps the shared camera backend and emits `image_raw` so other nodes can consume frames without owning device access themselves.

## ArmorDetectorNode

`ArmorDetectorNode` is the main entry into the split pipeline. It handles camera capture, detector backend setup, optional DeepStream fallback behavior, candidate extraction, optional armor-corner PnP, and publication of `detector/armors`.

This is the node to read first if you want to understand how model outputs become ROS messages.

## ArmorTrackerNode

`ArmorTrackerNode` subscribes to `detector/armors`, maintains track state, and chooses the best target. It publishes both the full tracked set and the selected target so debugging and control logic can stay separate.

The core tracking behavior lives in the tracker library code, while the node mostly handles ROS I/O and parameter wiring.

## ArmorProcessorNode

`ArmorProcessorNode` subscribes to `tracker/tracked_armors`, selects the active track, transforms its pose into the configured target frame, smooths pose and velocity, runs the small-gyro fire-gating observer, and publishes `processor/target`.

It also publishes `debug/spin_info` and `processor/marker` for tuning. If the selected target lacks pose for a frame, the node keeps the 2D target alive and lets the solver use its fallback path.

## AimSolverNode

`AimSolverNode` converts the selected target into control-friendly outputs. It can solve from 3D pose and velocity with latency, bullet-speed, and optional gravity compensation, or fall back to the older image-center FOV projection.

STM yaw/pitch feedback from `attitude_data` is interpreted as radians. Set `output_mode:=absolute` to add that feedback to the solved yaw/pitch before publishing `bounding_box_angles`; keep `output_mode:=offset` for pure offsets.

## STM32SerialBridgeNode

`STM32SerialBridgeNode` is the serial boundary to the controller. It publishes and consumes compact protocol data rather than full perception messages, which keeps the handoff predictable for the embedded side. It publishes STM RX attitude in radians on `attitude_data`, publishes flags on `attitude_flags`, and converts solver radians to degrees by default before STM TX.

## PerceptionNode

`PerceptionNode` is the older combined runtime path. It still matters because it preserves an end-to-end implementation in one place and is used by the zero-copy launch setup, but it is not the clearest place to start if you are trying to learn the active modular stack.

## YoloNode

`YoloNode` is another older combined-style path that works directly from `image_raw`. It remains in the package as a useful reference, but the detector, tracker, and solver nodes are the cleaner representation of the current architecture.
