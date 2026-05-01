# rm_auto_aim Parity Next Steps

Scope: roadmap from current C++ 2D detector -> tracker -> solver pipeline in [`yolov8_ros_cpp`](/Users/Kushal/NYU/CV_Detection/yolov8_ros_cpp) toward fuller parity with [`TantalumKevin/rm_auto_aim`](https://github.com/TantalumKevin/rm_auto_aim).

Current state:
- We now have split C++ nodes for detection, tracking, and solver.
- We now have custom tracking messages in [`cv_auto_aim_interfaces`](/Users/Kushal/NYU/CV_Detection/cv_auto_aim_interfaces).
- We still operate in image space and publish yaw/pitch compatibility outputs for STM.
- We do **not** yet have 3D pose solve, TF/world-frame tracking, target velocity in world frame, anti-spin observer, or ballistic compensation.

Use this file as the handoff after 2D validation passes.

## Phase 0: Validate Current 2D Pipeline

Goal:
- Prove new 2D path is stable before adding 3D complexity.

Packages / files to exercise:
- [`yolov8_ros_cpp/launch/stm.launch.py`](/Users/Kushal/NYU/CV_Detection/yolov8_ros_cpp/launch/stm.launch.py)
- [`yolov8_ros_cpp/launch/realsense.launch.py`](/Users/Kushal/NYU/CV_Detection/yolov8_ros_cpp/launch/realsense.launch.py)
- [`yolov8_ros_cpp/src/armor_detector_node.cpp`](/Users/Kushal/NYU/CV_Detection/yolov8_ros_cpp/src/armor_detector_node.cpp)
- [`yolov8_ros_cpp/src/armor_tracker_node.cpp`](/Users/Kushal/NYU/CV_Detection/yolov8_ros_cpp/src/armor_tracker_node.cpp)
- [`yolov8_ros_cpp/src/aim_solver_node.cpp`](/Users/Kushal/NYU/CV_Detection/yolov8_ros_cpp/src/aim_solver_node.cpp)

Checklist:
- Build workspace with `cv_auto_aim_interfaces` and `yolov8_ros_cpp`.
- Launch `realsense.launch.py` and verify:
  - `detector/armors` publishes detections.
  - `tracker/tracked_armors` publishes stable IDs.
  - `tracker/target` holds target lock across nearby frames.
  - `bounding_box_angles` stays zero when no target exists.
- Launch `stm.launch.py` and verify:
  - STM bridge still transmits using `override` and `bounding_box_angles`.
  - `attitude_flags` still updates target color behavior.
  - no-target path yields `override=false` and zero angles.
- Test scenes:
  - one target, steady motion
  - two targets, crossing motion
  - short occlusion and reacquisition
  - wrong-color distractor
  - empty frame after valid target

Exit criteria:
- No topic shape regressions for STM.
- Track IDs remain stable through normal motion.
- Target switching follows lock policy instead of flickering.
- End-to-end latency is acceptable for current robot tuning.

## Phase 1: Add 3D-Ready Detector Outputs

Goal:
- Keep detector architecture, but extend outputs from 2D-only armor observations to pose-ready observations.

Main changes:
- Extend `cv_auto_aim_interfaces` messages instead of creating a second interface package.
- Add detector fields needed for PnP:
  - armor corner points in image space
  - armor type or size class
  - source timestamp and frame id discipline
- Add camera calibration consumption to detector:
  - subscribe to `sensor_msgs/CameraInfo`
  - cache intrinsics and distortion coefficients
- Add pose solve in detector:
  - use armor corner points
  - use known armor geometry
  - compute pose in camera frame with `solvePnP`

Suggested interface additions:
- `Armor.msg`
  - `geometry_msgs/Point32[4] corners`
  - `uint8 armor_type`
  - `geometry_msgs/Pose pose_camera`
  - `bool has_pose`
- Keep current bbox/center fields for 2D compatibility.

Files likely to change:
- [`cv_auto_aim_interfaces/msg/Armor.msg`](/Users/Kushal/NYU/CV_Detection/cv_auto_aim_interfaces/msg/Armor.msg)
- [`cv_auto_aim_interfaces/msg/Armors.msg`](/Users/Kushal/NYU/CV_Detection/cv_auto_aim_interfaces/msg/Armors.msg)
- [`yolov8_ros_cpp/src/armor_detector_node.cpp`](/Users/Kushal/NYU/CV_Detection/yolov8_ros_cpp/src/armor_detector_node.cpp)

Exit criteria:
- Detector can publish valid 3D pose in camera frame for detected armors.
- 2D outputs remain unchanged for STM compatibility.

## Phase 2: Replace Image-Space Tracking with Pose-Aware Tracking

Goal:
- Move tracker from bbox-center persistence to target-state persistence that understands camera-frame pose.

Main changes:
- Track on 3D target position, not only image center.
- Keep 2D association fallback while pose quality is poor.
- Carry velocity estimate in tracker outputs.
- Distinguish measurement state from prediction state.

Suggested interface additions:
- `TrackedArmor.msg`
  - `geometry_msgs/Point position_camera`
  - `geometry_msgs/Vector3 velocity_camera`
  - `bool has_velocity`
- `Target.msg`
  - same position / velocity fields for selected target

Tracker work:
- Maintain per-track filtered position and velocity.
- Fuse 2D bbox association with 3D consistency gates.
- Continue lock / hysteresis behavior from current 2D tracker.

Files likely to change:
- [`yolov8_ros_cpp/include/yolov8_ros_cpp/tracker.hpp`](/Users/Kushal/NYU/CV_Detection/yolov8_ros_cpp/include/yolov8_ros_cpp/tracker.hpp)
- [`yolov8_ros_cpp/src/tracker.cpp`](/Users/Kushal/NYU/CV_Detection/yolov8_ros_cpp/src/tracker.cpp)
- [`yolov8_ros_cpp/src/armor_tracker_node.cpp`](/Users/Kushal/NYU/CV_Detection/yolov8_ros_cpp/src/armor_tracker_node.cpp)

Exit criteria:
- Selected target has stable camera-frame position and velocity.
- Occlusion recovery still works.
- 2D compatibility outputs still behave correctly.

## Phase 3: Add World-Frame Processor for Real rm_auto_aim Parity

Goal:
- Match upstream architectural step where detections are transformed and tracked in world or turret-relevant coordinates.

Main changes:
- Add new node, likely `armor_processor_node`.
- Subscribe to tracked armors plus TF.
- Transform camera-frame poses into gimbal/world frame.
- Publish world-frame target state and filtered velocity.

Required dependencies:
- `tf2`
- `tf2_ros`
- `geometry_msgs`
- likely `Eigen3`

Suggested new outputs:
- `processor/target`
  - pose in world or gimbal frame
  - velocity in same frame
  - selected armor ID
  - timestamped target state for solver use

Design note:
- Do **not** overload `armor_tracker_node` with TF logic.
- Keep tracker pure on target identity and state.
- Put frame transforms in dedicated processor layer.

Exit criteria:
- World-frame target motion is consistent while robot/camera moves.
- Timing and frame IDs stay coherent through detector -> tracker -> processor.

## Phase 4: Add Small-Gyro / Spinning Target Observer

Goal:
- Reach core upstream behavior for rotating enemy handling.

Main changes:
- Add target motion classifier:
  - normal motion
  - spinning / small-gyro behavior
- Add observer or model for rotating armor set around robot center.
- Switch aim point from currently visible armor to predicted future armor position when spin mode is active.

Expected new concepts:
- robot center estimate
- radius estimate
- angular velocity estimate
- armor index or armor ordering around target

Best structure:
- Implement this inside processor layer or a dedicated observer module, not inside detector.

Exit criteria:
- In spinning-target tests, chosen aim point remains stable and predictive instead of chasing visible plate jitter.

## Phase 5: Add Ballistic Solver and Final Aim Model

Goal:
- Move from image-space or raw target yaw/pitch to actual shot solution parity.

Main changes:
- Replace current `aim_solver_node` angle projection with solver that consumes:
  - target pose
  - target velocity
  - bullet speed
  - turret or gimbal orientation
  - fire delay / pipeline latency
- Compensate for:
  - target motion during latency
  - projectile flight time
  - gravity drop if needed

Inputs already partly available:
- `attitude_flags` from STM
- future turret/world target state from processor

Recommended output strategy:
- Keep old STM compatibility topics.
- Add debug solver topics with:
  - solved yaw
  - solved pitch
  - predicted hit time
  - target frame used for solution

Exit criteria:
- Solver output tracks moving targets better than direct center-angle projection.
- System still falls back cleanly when 3D state is missing.

## Order to Follow

Recommended order:
1. Finish Phase 0 and freeze 2D baseline.
2. Do Phase 1 and Phase 2 together in one branch if calibration source is already available.
3. Add Phase 3 as separate processor node.
4. Add Phase 4 after world-frame target state is trustworthy.
5. Add Phase 5 last.

Reason:
- 3D parity depends on good calibrated detections first.
- Small-gyro and ballistics built before pose / TF layer will create churn.

## Definition of “Good Enough to Move to 3D”

Move on only if:
- 2D detector/tracker/solver path survives repeated launch and shutdown.
- STM integration shows no protocol regressions.
- target lock behaves predictably under multi-target scenes.
- no-target behavior stays zeroed and safe.
- basic tracking tests pass consistently.

If any of those fail, keep fixing 2D first. Do not stack 3D work on unstable 2D behavior.
