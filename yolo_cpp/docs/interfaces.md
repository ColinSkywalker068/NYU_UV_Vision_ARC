# cv_auto_aim_interfaces

Related pages: [Overview](./overview.md) | [Pipeline](./pipeline.md) | [Nodes](./nodes.md) | [Legacy Pipeline Notes](./legacy-pipeline-notes.md) | [Launch and Config](./launch-and-config.md) | [Development Notes](./development-notes.md)

`cv_auto_aim_interfaces` is the message package that lets the detector, tracker, and solver talk to each other without sharing implementation details. The important thing to remember is that these messages describe the perception state at three different levels of confidence and stability.

## Armor

`Armor.msg` is the smallest unit. It describes one detection candidate:

- `bbox_xyxy`: bounding box corners in image pixels
- `center_x`, `center_y`: box center
- `confidence`: detector score
- `class_id`: model class
- `is_enemy`: whether the candidate matches the current target color logic
- `has_obb_angle`, `obb_angle`: optional oriented-box heading
- `image_width`, `image_height`: frame size used to interpret the box

This is the message shape the detector uses before any temporal tracking is applied.

## Armors

`Armors.msg` wraps a whole frame of detections:

- `header`: timestamp and frame metadata
- `source_id`: camera or source label
- `armors`: array of `Armor`

The detector publishes this on `detector/armors`. If you want to inspect raw per-frame perception output, this is the best topic to watch first.

## TrackedArmor

`TrackedArmor.msg` is one `Armor` plus tracker state:

- `armor`: the latest underlying detection
- `track_id`: stable identifier for the target over time
- `state`: `DETECTING`, `TRACKING`, or `COASTING`
- `age`, `hit_streak`, `lost_frames`: lifecycle counters
- `predicted_center_x`, `predicted_center_y`: tracker prediction

This is the bridge from frame-by-frame detections to stable target identity.

## TrackedArmors

`TrackedArmors.msg` is the tracker-level view of the full frame:

- `header`
- `source_id`
- `tracks`: array of `TrackedArmor`
- `selected_track_id`: tracker id chosen as the current target, or `-1`

The tracker publishes this on `tracker/tracked_armors`. It is the right topic for understanding why target selection changed or why a track started coasting.

## Target

`Target.msg` is the compact control-facing output. It keeps only the currently selected target and the fields downstream nodes need:

- target availability via `has_target`
- tracking quality via `state` and `is_coasting`
- geometry via `bbox_xyxy`, `center_*`, and `predicted_center_*`
- classification via `class_id`, `is_enemy`, and optional `obb_angle`
- frame dimensions so the solver can compute angles consistently

The tracker publishes this on `tracker/target`, and `AimSolverNode` consumes it directly.

## How the Interfaces Fit Together

The message flow is:

`Armor` -> `Armors` -> `TrackedArmor` -> `TrackedArmors` -> `Target`

That progression mirrors the runtime:

1. The detector finds boxes.
2. The detector publishes the frame-wide detection set.
3. The tracker adds temporal identity and prediction.
4. The tracker publishes both the full tracked set and the single chosen target.
5. The solver turns the selected target into yaw, pitch, and controller-facing topics.
