# Local pose and tracking upgrade

Branch: `codex/autoaim-pose-tracking`. Work in progress; no robot has been connected or flashed for this upgrade.

## Contracts being validated

- Detector images retain their capture timestamp. Missing/invalid time does not become a new measurement stamped “now.” DeepStream maps buffer PTS through its pipeline clock even when preview is off. The native camera timestamp is still host receipt time; exposure/USB latency needs a measured calibration.
- `gimbal/attitude` carries one receive timestamp assigned in the bridge, shared by tracker and solver. The SP wire has no device timestamp: this remains an estimate, not clock synchronization. `attitude_receive_delay_s` subtracts a measured fixed telemetry delay. A batched read publishes only its newest SP sample.
- Tracker rotates the full PnP pose into `gimbal_inertial` before 3D association/filtering. X is right, Y down, Z forward at control zero; positive control yaw is right and pitch is up. Camera mounting rotation and translation belong in tracker extrinsics. This compensates gimbal rotation, not chassis translation; moving-chassis odometry is a remaining integration item.
- Interpolation rejects telemetry gaps over 50 ms; edge extrapolation is limited to 30 ms and telemetry expires after 250 ms. Images require increasing, nonzero capture time. A coasted track retains its last actual measurement time.
- Processor keeps separate robot estimates by geometric association plus known color/number. A lone stationary armor stays in direct-plate mode. It does not treat predicted/coasting plates as new measurements or smooth the tracker output a second time.
- Solver consumes the already transformed position without adding camera attitude again. Prediction includes measured frame age, configured residual transport delay, flight time and a separately configured actuator delay. The reported slew time uses the sentry limits (120/60 deg/s and 360/240 deg/s² yaw/pitch); it is a rest-to-rest bound, not a measured motor-response guarantee.
- Serial bridge consumes one atomic `aim_solution` message. Stale, future, replayed, nonfinite or offset commands are rejected. Separate legacy angle/override topics remain for previews; old serial-topic behavior requires explicit `use_legacy_command_topics:=true`.

## Model decision

ARUW's 2026 four-corner model is the first integration candidate. It has an explicit 16-class label contract and a simple end-to-end output; it also had lower CPU inference time in this small check. RobotPilots 2026 remains a strong challenger and produced proposals on two additional Tongji frames. This is **not** a measured accuracy ranking: the public clips are unlabeled, may overlap training data, and are not our Basler camera.

| Candidate | Verified input/output | Local evidence | Integration decision |
| --- | --- | --- | --- |
| ARUW 2026 YOLO26n-pose | FP32 `1×3×384×640` → `1×300×14` | Blank: zero proposals; proposals on 21/24 Tongji and 24/24 RobotPilots sampled frames; median CPU inference about 45 ms | Initial explicit `model_type:=aruw_2026_pose` adapter |
| RobotPilots 2026 infantry v8 | FP32 IO despite fp16 filename; `1×3×480×640` → `1×21×6300` | Blank: zero proposals; 23/24 and 24/24 frames; median about 60 ms | Challenger; separate decoder required |
| Tongji 2025 | OpenVINO IR `1×3×640×640` → `1×25200×22` | Weight files and contract inspected | Estimator reference; additional runtime conversion needed for TensorRT |
| TUP 2023 | ONNX `1×3×416×416` → `1×3549×21` | Download/hash/shape verified | Older baseline; explicit model license and full preprocessing contract unresolved |

CPU timings were measured in an ARM64 Linux Docker container with ONNX Runtime 1.22.1, two inference threads, on this Mac. They do not predict Jetson TensorRT latency. Results: [machine-readable smoke report](model-cpu-smoke-2026-09-30.json).

Native Mac acceleration is now available through ONNX Runtime Core ML. On the M5,
ARUW measured 5.09 ms median with GPU-enabled execution versus 8.96 ms native CPU;
all 71 above-threshold outputs matched across 48 sampled frames. The diagnostic
compute plan assigned most operations to GPU, with CPU fallback for output
selection. See the [native benchmark](model-native-mac-2026-09-30.json) and
[reproducible video replay guide](../../tools/model-eval/README.md). The replay runs
the production C++ decoder/tracker and renders ordinary video files with corners,
IDs, coasting and pixel error. It does not invent 3D commands from uncalibrated clips.

ARUW's upstream `ColorImage` contract is **BGR**. Its model path resizes floating-point BGR directly to 640×384, without letterboxing. Our adapter deliberately uses that path. RobotPilots uses RGB, centered letterboxing with padding 124, and split color/type outputs. Treating either as our old two-class OBB model is incorrect.

The ARUW adapter orders corners BL/TL/TR/BR and derives the tracked rectangle from
their bounds, matching the upstream parser. A context box offset cannot move the
centering point away from the predicted armor corners; this has a regression test.
Public spin footage also exposed multiple model rows for one physical plate. The
adapter removes near-identical same-color quads (IoU > 0.85), with a regression
that preserves adjacent plates. Full replay still shows identity/selection churn
on spin footage. Reducing duplicate proposals alone does not prove stable robot
association; the image-only replay deliberately lacks calibrated 3D robot state.

The current branch also contains an optional 11-state EKF for robot center,
translation, phase, angular speed, radius and alternating plate geometry. It
weights uncertain planar normals less strongly and retains direct tracking for a
lone stationary plate. Processor defaults select it for ongoing branch testing;
the baseline estimator remains available through `robot_estimator.use_ekf:=false`.
Initial estimator and ROS pipeline regressions pass. Noisy projected-corner,
multiple-robot and closed-loop actuator validation remain required before this
branch is a deployment recommendation.

Download pinned weights outside the repository:

```bash
python3 tools/model-eval/fetch_candidates.py --output /tmp/armor-models --candidate aruw-2026
```

The fetcher verifies expected size and Git blob hash, records SHA256, and inspects ONNX metadata without executing a pickle. It does not create a TensorRT engine. Engines must be built for the destination Jetson's TensorRT version.

For ARUW evaluation set `model_type:=aruw_2026_pose`, `task:=pose`, `imgsz_width:=640`, `imgsz_height:=384`. Native TensorRT additionally requires BGR8 camera output. DeepStream requires pre-mux resizing off. Native TensorRT/CUDA/DeepStream execution has not yet been validated for this branch.

PnP requires **real keypoints and calibration matching the source image resolution**. Box-derived corners and guessed FOV intrinsics are insufficient for robot-state estimation. Poor/degenerate quadrilaterals, negative depth, excessive range and large reprojection error are rejected. Ambiguous planar normals cannot confirm a spinning robot, although usable position can still support direct-plate tracking.

## Safety and deployment

Keep `transmit_enable:=false`, `enable_fire_recommendation:=false`, and `solver_output_mode:=absolute` for preview. TX disabled still permits the solver to publish its hypothetical command. Receiving attitude is required for compensated tracking. A missing command with visible detections can indicate missing attitude, stale capture time, incorrect calibration, or invalid geometry; do not solve it by enabling TX.

The repository-local `services/stm-stack.env` remains the saved runtime configuration. A config-only change needs `cv restart`. Changes to C++ or ROS message definitions need a build of **both** `cv_auto_aim_interfaces` and `yolo_cpp`, followed by restart. These Jetson-only changes preserve the SP/ST wire format and do not require a Type C flash. Actual controller acceptance and actuator limits still depend on its installed firmware and RC mode.

## Reproduce local validation

With Docker running on the host device:

```bash
tools/run-local-tracking-tests.sh
```

This builds the ROS interfaces and runtime, then runs the C++ geometry/model/estimator tests, the actual ROS pose pipeline, and the production bridge with virtual serial ports. No robot device is mounted. JUnit results remain under `.local-test-output/build/yolo_cpp/test_results/`. The script builds its own test image on first use; `CV_TEST_IMAGE` and `CV_TEST_OUTPUT` override the image and output folder.

The production pose test generates corners for a stationary plate while the camera pans, pitches and rolls. It runs the actual ARUW decoder, PnP, tracker, processor and solver. It checks absolute yaw error below 0.001 rad, apparent target velocity below 0.01 m/s, direct-plate fallback, and fire suggestions disabled. It does not assert perception accuracy on real frames or a validated physical motor model.

Latest [checkpoint results](pose-pipeline-checkpoint-2026-09-30.json): 50 ROS checks passed; 12 existing Python tests and the Linux service-config regression passed. The synthetic camera-pan run produced 79 valid commands, maximum yaw error `1.558e-7 rad`, and maximum apparent target speed `0.0001321 m/s`.

## Remaining work before deployment approval

- Complete production tests for spinning/nonspinning robots, distractors, noisy corners, dropped frames and actuation using actual sentry limits; the old Python simulator is not evidence that the live ROS path works.
- Evaluate an uncertainty-aware robot estimator against the existing baseline; check radius/height variation and phase ambiguity rather than blindly importing a team's whole runtime.
- Wire/calibrate mounting extrinsics, sensor receipt delays and camera intrinsics for the specific robot. Verify moving-chassis frame handling.
- Check the model on held-out positive/negative Basler recordings. CPU smoke counts are not recall or false-positive rates.
- Validate TensorRT engine build and replay throughput on the camera-less Jetson. Later, validate live Basler capture and telemetry with TX disabled before any operator-controlled motion test.

## Upstream references and licenses

- [ARUW Project Otto](https://gitlab.com/aruw/vision/project-otto/-/tree/154035fc34ffb34f99120ffa8e8a2bb098bc6306): GPL-3.0 source; downloaded model metadata declares AGPL-3.0. Our adapter implements its documented tensor contract; model weights are not committed.
- [RobotPilots 2026](https://github.com/SZURPVision/26_NNDeployment_Lib_and_Detection_Models/tree/ccb5dfe09696edeea445ffb8f99adaa7c383ec3a): MIT source; model metadata declares AGPL-3.0. Keep these distinctions when distributing artifacts.
- [Tongji SuperPower 2025](https://github.com/TongjiSuperPower/sp_vision_25/tree/bd9f5e798fa3c6dd3b483ae6627796afb41c608d): MIT source, reference for center/phase/radius/alternating-height state estimation. Its state uses different axes from ours.
- [TUP 2023](https://github.com/tup-robomaster/TUP-Vision-2023-Based/tree/40b8269ef8b0b9c3475911a06d6cd13a6e1257f2): no explicit weight license located; do not treat public downloadability as a redistribution license.
- [Sumedh's CV learnings](https://forums.arc-robotics.org/t/sumedh-s-various-cv-learnings/247): informs capture/inference/telemetry separation, replay, exposure tuning, and the stationary-target-under-camera-motion check.
