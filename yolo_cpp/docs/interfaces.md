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
- `has_corners`, `corners`: four image-space armor corners for PnP
- `armor_type`, `armor_number`: size class and optional armor id
- `has_pose`, `pose_camera`: optional camera-frame PnP result
- `distance_to_image_center`: pixel distance from the principal point or image center

This is the message shape the detector uses before any temporal tracking is applied. The extra 3D fields are optional so 2D-only detections still flow through the stack.

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
- `has_pose`, `pose_camera`: filtered camera-frame pose for the track
- `predicted_position_camera`: camera-frame 3D prediction
- `has_velocity`, `velocity_camera`: camera-frame velocity estimate

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
- optional 3D pose, velocity, range, tracking state, and fire suggestion

The tracker publishes a 2D-compatible target on `tracker/target`. The processor publishes the full filtered target on `processor/target`, and the 3D solver consumes that topic in the full launch files.

## SpinInfo

`SpinInfo.msg` is debug state from the small-gyro observer:

- `target_spinning`: whether repeated armor jumps indicate spinning motion
- `suggest_fire`: whether the observer currently allows firing
- `jump_count`, `yaw_diff`, `jump_period`, `time_after_jumping`: observer timing and jump metrics

## AimSolution

`AimSolution.msg` is the atomic solver output consumed by the STM bridge:

- `has_solution`, `used_3d`, `suggest_fire`
- yaw and pitch in radians plus degree copies for inspection
- estimated `flight_time`, `range`, and `bullet_speed`
- `has_angular_velocity`, `yaw_velocity`, `pitch_velocity`: rates in rad/s of
  the same absolute ballistic aim point, sharing its capture timestamp and
  `gimbal_inertial` frame; zero on acquisition, identity/slot changes, coasting,
  stale/invalid input, 2D/offset mode, or rate/innovation guard failure

Unavailable rates do not invalidate an otherwise valid angle solution. The
bridge requires rate capture age at most 100 ms and maps the rates to existing
SP velocity fields; acceleration fields remain zero. A zero wire rate can mean
stationary or unavailable, since the wire format has no rate-valid flag.
Rebuild both active ROS packages after changing this message.

## Controller auto aim acceptance

`controller_auto_aim` (`std_msgs/msg/Int8`) is `1` for ON, `0` for OFF, and
`-1` for UNKNOWN. It comes only from a CRC-valid 39-byte MCU ST packet:
`robot_status` at byte offset 26 has bit 7 (`0x80`) marking this metadata as
supported and bit 3 (`0x08`) reporting the effective logical auto aim gate.
Bits 0–2 retain referee power status. Without bit 7, the status is UNKNOWN;
SP mode, vision transmission, target detection, and referee power never imply ON.

The bridge expires ST status after 500 ms using monotonic receipt time and
clears it on disconnect. The detector separately expires ROS status after
500 ms; invalid scalar values become UNKNOWN. Bad CRC packets and SP attitude
traffic cannot refresh acceptance. This requires direct CDC or a link forwarding
ST intact; the legacy Sentry bridge path that discards `robot_status` cannot
provide it and remains UNKNOWN.

The preview displays `AUTO AIM ON/OFF/UNKNOWN` separately from `TX CONFIG`.
ON means the controller reports its logical gate armed; it does not prove target
tracking, motor power/motion, packet acceptance, or firing. `/stats.json` exposes
`auto_aim_status` (`on`, `off`, `unknown`) and `auto_aim_status_age_ms` (age of the
last received ROS status, `null` before any message). HTTP freshness is evaluated
when requested, even if camera frames stop. An already displayed frozen image
cannot repaint itself; use the live sidebar/status endpoint for current status.

## How the Interfaces Fit Together

The message flow is:

`Armor` -> `Armors` -> `TrackedArmor` -> `TrackedArmors` -> `Target` -> `AimSolution`

That progression mirrors the runtime:

1. The detector finds boxes.
2. The detector publishes the frame-wide detection set.
3. The tracker adds temporal identity and prediction.
4. The tracker publishes both the full tracked set and the single chosen target.
5. The processor transforms, filters, and fire-gates the selected target.
6. The solver turns the selected target into yaw, pitch, and controller-facing topics.
