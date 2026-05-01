# Pipeline

Related pages: [Overview](./overview.md) | [Nodes](./nodes.md) | [Interfaces](./interfaces.md) | [Legacy Pipeline Notes](./legacy-pipeline-notes.md) | [Launch and Config](./launch-and-config.md) | [Development Notes](./development-notes.md)

The live `yolo_cpp` pipeline is split into small ROS 2 stages so each part can be tuned or swapped without rewriting the whole runtime.

## Camera to Detection

The detector stage pulls frames through the shared camera backend. In practice that means RealSense or Basler frames are normalized into a common `FramePacket`, preprocessed, and sent through either the native TensorRT path or the DeepStream-backed path.

The detector publishes `cv_auto_aim_interfaces/msg/Armors` on `detector/armors`. Each message carries the frame metadata plus the candidate armor boxes, confidence, image size, and optional OBB angle data that later stages depend on.

## Detection to Tracking

The tracker subscribes to `detector/armors` and turns frame-by-frame detections into stable tracks. It keeps enough state to smooth over short gaps, avoid target flicker, and choose a single target when multiple candidates are visible.

It publishes two outputs:

- `tracker/tracked_armors` for the full tracking state
- `tracker/target` for the current selected target

That split is useful in practice. The first topic is better for debugging and visualization, while the second is the compact control signal that the solver uses.

## Tracking to Aim Output

The solver subscribes to `tracker/target`. When a target is available, it converts the bounding box center into yaw and pitch using the configured camera FOV values. When no target is available, it can publish a safe zero-angle fallback instead.

The solver publishes:

- `bounding_box`
- `bounding_box_angles`
- `override`
- `obb_angle`

These topics are the handoff between perception and downstream control.

## STM Bridge

In the full stack launched by `stm.launch.py`, the STM bridge sits at the edge of the pipeline. It handles serial transport to the controller and also listens for feedback such as enemy color or attitude flags that can influence target selection upstream.

## Combined Path

The package also still contains a combined `PerceptionNode`. That node folds camera, detection, and aiming behavior into one component. It is useful as a reference path and for zero-copy experiments, but the split detector -> tracker -> solver path is the main runtime to understand first.
