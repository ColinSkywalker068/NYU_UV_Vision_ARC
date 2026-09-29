# Robot Tracking Manual Test Guide

Use this guide after the synthetic and ROS tests pass. Keep the launcher disabled and begin with
the observation profile, which does not create the aim solver or serial bridge.

## 1. Repeat the Mac Tests

From the repository root:

```bash
python3 yolo_cpp/tools/test_robot_tracking_scenario.py
python3 yolo_cpp/tools/test_robot_tracking_closed_loop_sim.py
python3 yolo_cpp/tools/test_observation_launch_contract.py
```

Generate a repeatable trace without any fire recommendation:

```bash
python3 yolo_cpp/tools/robot_tracking_closed_loop_sim.py \
  --scenario spin_stop_reverse \
  --dropout-probability 0.10 \
  --output /tmp/robot-tracking.jsonl
```

## 2. Safe Live Observation

On the Jetson, source the built ROS workspace and run:

```bash
ros2 launch yolo_cpp tracking_observe.launch.py \
  target_color:=red \
  robot_armor_radius_m:=0.26 \
  robot_armor_count:=4
```

This profile starts the camera, detector, armor tracker, and robot estimator. It explicitly omits
the solver and serial bridge. It therefore cannot publish aim solutions or open the Type-C serial
port.

Inspect the processed state:

```bash
ros2 topic echo /processor/target
```

Record evidence for replay and tuning:

```bash
ros2 bag record \
  /detector/armors \
  /tracker/tracked_armors \
  /processor/target \
  /debug/spin_info
```

The processor also prints one `TRACK_STATE` line per second. The line includes the active mode,
robot center, velocity, phase, angular velocity, coast count, and accepted/rejected observation
counts.

Expected behavior:

- A single stationary armor uses `mode=direct_plate` and `has_robot_state: false`.
- A non-spinning robot remains trackable through its directly observed plate. If two plates or a
  plate transition provide robot evidence, `has_robot_state` becomes true with angular velocity
  near zero.
- A spinning robot becomes `mode=robot` after the configured minimum updates and keeps the same
  robot center while visible plate IDs change.
- A plate from another robot outside `robot_center_gate_m` increments the rejected count instead
  of pulling the tracked center.

## 3. Solver Preview Without Robot Transmission

After observation results look correct, run the ordinary Basler launch with the solver enabled and
serial bridge omitted:

```bash
ros2 launch yolo_cpp basler.launch.py \
  start_solver:=true \
  start_serial_bridge:=false \
  use_3d_solution:=true \
  robot_estimator_enable:=true \
  enable_fire_recommendation:=false \
  solver_output_mode:=absolute
```

Inspect the hypothetical command:

```bash
ros2 topic echo /aim_solution
```

`AimSolution.suggest_fire` remains false. The Type-C serial device is not opened.

## Tracking Parameters

| Parameter | Purpose | Initial value |
|---|---|---:|
| `robot_estimator_enable` | Enables shared robot center and phase estimation | `true` |
| `robot_armor_radius_m` | Robot center to armor plane | `0.26` m |
| `robot_armor_count` | Equally spaced armor slots | `4` |
| `robot_center_gate_m` | Rejects plates inconsistent with the current robot | `0.45` m |
| `robot_plate_cluster_gate_m` | Keeps nearby robots from merging into one center | `0.18` m |
| `robot_estimator_min_updates` | Evidence required before future-plate prediction | `5` |
| `robot_estimator_coast_frames` | Missing frames allowed before model confidence drops | `8` |
| `solver_latency_s` | Measured camera-to-solver latency added to flight time | `0.03` s |

Measure the physical armor radius rather than tuning it by feel. Tighten the center gate only after
recorded logs show that valid plates remain inside it.

## Settings That Can Cause Real Motion

Real Type-C movement packets require all of these conditions:

```text
start_serial_bridge:=true
solver_output_mode:=absolute
transmit_enable:=true
Type-C firmware configured to accept vision control
```

Keep `transmit_enable:=false` until the actuator bench is physically safe. Keep
`enable_fire_recommendation:=false`; only a person should enable it for competition. Changing
Jetson launch or environment values requires rebuilding only when source code changed, then restart
the CV service. Changing Type-C firmware settings requires rebuilding and reflashing the Type-C.
