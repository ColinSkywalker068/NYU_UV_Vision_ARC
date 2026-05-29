# Spin and Armor Offset Next Steps

Scope: plan for solving chassis spin, also called beyblading, and non-uniform armor height offsets on top of the current 3D auto-aim path.

Related code:
- [`yolo_cpp/src/armor_processor_node.cpp`](/Users/Kushal/NYU/CV_Detection/yolo_cpp/src/armor_processor_node.cpp)
- [`yolo_cpp/src/aim_solver_node.cpp`](/Users/Kushal/NYU/CV_Detection/yolo_cpp/src/aim_solver_node.cpp)
- [`cv_auto_aim_interfaces/msg/Target.msg`](/Users/Kushal/NYU/CV_Detection/cv_auto_aim_interfaces/msg/Target.msg)
- [`docs/RM_AUTO_AIM_PARITY_NEXT_STEPS.md`](/Users/Kushal/NYU/CV_Detection/docs/RM_AUTO_AIM_PARITY_NEXT_STEPS.md)

## Simple Mental Model

Old behavior:

```text
visible armor moved -> aim moved
```

Desired behavior:

```text
visible armor moved -> estimate robot body center -> predict useful armor -> aim corrected point
```

Why 3D matters:
- 2D sees a box jumping around the image.
- 3D sees an armor plate moving around a physical chassis.
- Once we know the chassis center, armor radius, spin direction, and armor height offset, aiming can stop chasing jitter.

## Definitions

Beyblading:
- The enemy chassis rotates quickly.
- The visible armor jumps around the robot center.
- The tracker can mistake this for the whole robot teleporting sideways.

Armor offset:
- Two armor plates are mounted about `+1 inch` above normal.
- Two armor plates are mounted about `-1 inch` below normal.
- `1 inch = 0.0254 m`.
- The solver must aim at the real plate center, not a fake shared height.

## Current Starting Point

Already available:
- Detector can publish armor corners and optional camera-frame PnP pose.
- Tracker can carry camera-frame 3D position and velocity.
- Processor can transform selected armor pose into a target frame and smooth it.
- Processor has a first-pass small-gyro fire gate.
- Solver can use 3D pose, velocity, bullet speed, latency, gravity, and STM attitude feedback.

Missing:
- A robot-center model.
- A configurable armor layout model.
- A stable armor slot index or slot phase estimate.
- Height correction for high / low armor plates.
- Predictive spin aim point selection.

## Target Model

Represent the enemy as:

```text
robot center position
robot center velocity
armor radius
spin angle
spin angular velocity
armor slot count
per-slot yaw offset
per-slot vertical offset
```

Minimum layout for this case:

```text
slot 0: height +0.0254 m
slot 1: height -0.0254 m
slot 2: height +0.0254 m
slot 3: height -0.0254 m
```

Make this configurable, because the real mapping may be front/back high and left/right low, or the reverse.

## Phase 0: Measure Before Modeling

Goal:
- Get real numbers before tuning the observer.

Checklist:
- Confirm PnP pose axes with a stationary target.
- Confirm `processor_target_frame` matches the solver convention.
- Record a short bag or runtime log for:
  - stationary robot, one visible armor
  - slow chassis rotation
  - fast beyblading
  - target driving while spinning
  - high armor visible
  - low armor visible
- Measure approximate armor radius from robot center to armor center.
- Confirm which two plates are high and which two are low.

Exit criteria:
- PnP range is believable.
- Yaw direction is known.
- The high / low armor mapping is written down.

## Phase 1: Add Armor Layout Config

Goal:
- Teach the processor what the robot geometry looks like.

Add parameters under `armor_processor_node`:

```yaml
armor_layout.slot_count: 4
armor_layout.radius_m: 0.20
armor_layout.vertical_offsets_m: [0.0254, -0.0254, 0.0254, -0.0254]
armor_layout.slot_yaw_offsets_rad: [0.0, 1.5708, 3.1416, 4.7124]
armor_layout.default_slot: -1
```

Implementation notes:
- Keep the first version internal to the processor.
- Do not change STM protocol for this phase.
- Add debug logs / markers before changing aim behavior.

Files likely to change:
- [`yolo_cpp/include/yolo_cpp/armor_processor_node.hpp`](/Users/Kushal/NYU/CV_Detection/yolo_cpp/include/yolo_cpp/armor_processor_node.hpp)
- [`yolo_cpp/src/armor_processor_node.cpp`](/Users/Kushal/NYU/CV_Detection/yolo_cpp/src/armor_processor_node.cpp)
- [`yolo_cpp/launch/basler.launch.py`](/Users/Kushal/NYU/CV_Detection/yolo_cpp/launch/basler.launch.py)
- [`yolo_cpp/launch/realsense.launch.py`](/Users/Kushal/NYU/CV_Detection/yolo_cpp/launch/realsense.launch.py)

Exit criteria:
- Processor loads armor layout parameters.
- RViz markers show expected high / low slot positions around a fake center.

## Phase 2: Estimate Chassis Center

Goal:
- Convert visible armor pose into estimated robot center pose.

Basic idea:

```text
armor position = chassis center + rotated armor radius + armor height offset
```

So:

```text
chassis center = armor position - rotated armor radius - armor height offset
```

First implementation:
- Use visible armor yaw around the camera / target frame as the spin phase estimate.
- Use configured `armor_layout.radius_m`.
- Use the nearest matching slot offset if a slot can be inferred.
- Smooth center position and center velocity separately from armor position.

Fallback behavior:
- If slot is unknown, use zero vertical offset.
- If spin phase is noisy, keep current 3D armor tracking behavior.

Exit criteria:
- During chassis rotation in place, estimated robot center moves much less than visible armor pose.
- During normal driving, center velocity follows the robot body, not armor plate jumps.

## Phase 3: Infer Armor Slot and Height Offset

Goal:
- Decide which of the four armor slots is currently visible, then apply the right vertical correction.

Slot inference signals:
- Current armor angle around estimated chassis center.
- Previous slot and expected spin direction.
- Jump size between visible armors.
- Armor number if the detector eventually provides reliable number classification.

Simple first version:
- Unwrap observed armor angle over time.
- Quantize phase to the nearest of four slots.
- Apply that slot's `vertical_offsets_m`.
- Publish debug state showing selected slot and offset.

Better later version:
- Maintain probability per slot.
- Penalize impossible jumps.
- Fuse armor number when available.

Exit criteria:
- High plates get `+0.0254 m` correction.
- Low plates get `-0.0254 m` correction.
- Slot estimate does not flap randomly while one armor remains visible.

## Phase 4: Detect Beyblading

Goal:
- Classify normal motion vs chassis spin.

Signals:
- Repeated armor yaw jumps around the estimated center.
- Consistent spin direction.
- Estimated angular velocity above threshold.
- Armor position changes large, but chassis center stays stable.

Suggested debug fields:
- `target_spinning`
- `spin_direction`
- `spin_rate_rad_s`
- `estimated_radius_m`
- `visible_slot`
- `slot_confidence`

Possible interface options:
- Extend `SpinInfo.msg`.
- Or add a new `ChassisState.msg` if debug state gets large.

Exit criteria:
- Normal driving does not trigger spin mode.
- Slow rotation produces low spin rate.
- Fast beyblading reliably triggers spin mode.

## Phase 5: Predict Aim Point During Spin

Goal:
- Aim at the best future armor or stable chassis-relative point, not the noisy currently visible plate.

Inputs:
- chassis center position
- chassis center velocity
- spin angle
- spin angular velocity
- armor layout
- bullet flight time
- solver latency

Prediction:

```text
future_time = solver_latency + bullet_flight_time
future_center = center + center_velocity * future_time
future_spin_angle = spin_angle + spin_rate * future_time
future_armor_position = future_center + slot_radius_at(future_spin_angle) + slot_height_offset
```

Aim choices:
- If a valid future armor is hittable, aim at that armor center.
- If the phase is bad, hold fire with `suggest_fire=false`.
- If spin estimate is low confidence, fall back to current visible armor pose.

Exit criteria:
- Aim point is stable during fast spin.
- Fire gate only allows shots inside a configurable hit window.
- Solver output does not chase every visible armor jump.

## Phase 6: Wire Solver and STM Behavior

Goal:
- Keep solver and STM simple.

Processor should publish:
- `processor/target.pose`: corrected aim point
- `processor/target.velocity`: corrected aim point velocity or chassis velocity
- `processor/target.suggest_fire`: spin-aware fire gate
- debug topic with spin and slot state

Solver should:
- consume `processor/target` normally
- keep 2D fallback when `has_pose=false`
- keep publishing radians on `bounding_box_angles`

STM bridge should:
- keep receiving radians from ROS
- keep converting TX yaw/pitch to degrees when `transmit_angles_in_degrees=true`
- use `suggest_fire` to choose fire vs hold-fire mode

Exit criteria:
- No STM protocol shape change.
- No unit regression: ROS yaw/pitch radians, STM TX degrees.

## Phase 7: Tests and Tuning

Unit tests:
- slot quantization from angle
- high / low offset application
- center estimate from known armor pose
- spin direction and angular velocity from synthetic phase sequence
- fallback when pose is missing

Runtime tests:
- stationary target
- target moving straight
- slow chassis rotation
- fast beyblading
- high armor only
- low armor only
- mixed high / low slot transitions
- target disappears then reappears

Metrics:
- center jitter in meters
- slot estimate stability
- spin detection latency
- false spin detections
- aim point jump size
- `suggest_fire` duty cycle during spin

## Suggested Implementation Order

1. Add armor layout config and debug markers.
2. Add chassis center estimation with no behavior change.
3. Add slot inference and vertical offset correction.
4. Publish corrected target pose behind a parameter, default off at first.
5. Add spin classifier and richer `SpinInfo`.
6. Add predictive spin aim point.
7. Turn corrected aim on by default after logs look sane.

## Key Parameters To Tune

```yaml
armor_layout.radius_m
armor_layout.vertical_offsets_m
armor_layout.slot_yaw_offsets_rad
spin.min_rate_rad_s
spin.max_center_jitter_m
spin.min_slot_confidence
spin.fire_window_rad
spin.prediction_latency_s
filter.center_position_alpha
filter.center_velocity_alpha
```

## Definition of Done

This work is done when:
- A spinning robot produces a stable estimated chassis center.
- The system identifies high vs low armor slots well enough to apply the right `1 inch` correction.
- During beyblading, aim does not snap to every visible armor jump.
- `suggest_fire` blocks bad spin phases.
- Solver and STM units stay unchanged: ROS radians, STM TX degrees.
- The old 2D fallback still works when 3D pose or slot confidence is missing.
