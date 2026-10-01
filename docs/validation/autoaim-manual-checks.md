# ARUW tracking: manual deployment and measurements

<!-- SCOPE: Operator handoff for the locally tested pose/tracking branch; preview first, no automatic firmware deployment. -->
<!-- DOC_KIND: how-to -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Deploying codex/autoaim-pose-tracking to a Jetson for operator-supervised evaluation. -->
<!-- SKIP_WHEN: Running only the offline Mac video replay. -->
<!-- PRIMARY_SOURCES: services/stm-stack.sh, services/stm-stack-entrypoint.sh, services/uv.sh, yolo_cpp/launch/basler.launch.py, yolo_cpp/src/aim_solver_node.cpp, yolo_cpp/test/test_closed_loop_pipeline.cpp -->

## Quick navigation

| Need | Reference |
| --- | --- |
| Model evidence and limitations | [Local upgrade report](autoaim-local-upgrade.md) |
| Native Mac inference and overlay videos | [Replay guide](../../tools/model-eval/README.md) |
| USB and wireless SSH | [SSH guide](../project/ssh-access.md) |
| Container prerequisites | [Container runbook](../../CPP_CONTAINER_RUNBOOK.md) |

## Agent entry

This guide covers the active `yolo_cpp` and `cv_auto_aim_interfaces` packages.
It is an evaluation procedure, not evidence of a validated physical controller.
No hardware was connected or flashed during the local upgrade. The branch must
first be transferred to the Jetson or merged into its selected checkout.

## Saved configuration and changes

The live file is **`services/stm-stack.env` inside the Jetson checkout**. A common
checkout is `/home/ultraviolet/Documents/CV_Detection`; verify the actual path.
`services/stm-stack.env.example` is a template and does not change the service.
The service installation records the repo-local env path; an explicit
`CV_STM_ENV_FILE` override can select another file. Confirm the service definition
with `systemctl cat stm-stack` when migrating an old installation.

| Change | Required action on the Jetson |
| --- | --- |
| Saved env arguments, preview, exposure, enemy color | Save, then `cv restart` |
| CV C++, ROS messages, or launch files | `uv build yolo_cpp`, then `cv restart` |
| ONNX model or TensorRT/JetPack version | Build/validate an engine for that Jetson; never reuse an engine from a different runtime blindly |
| Type C C source or sentry limits | Separate `just build sentry`; changes take effect only after a separately authorized flash |

`uv build yolo_cpp` builds **both** message and runtime packages. Use the saved
container image compatible with that Jetson's JetPack; the Mac ROS test image is
not the Jetson deployment image. `cv start` does not repair or rebuild packages.

`cv start` and `cv restart` now wait for the ROS graph and check for fresh, valid
Type C SP attitude samples when the running bridge has RX or TX enabled.
`cv health` repeats the check without restarting anything. A present serial port
alone does not pass. The check only subscribes to decoded telemetry; it never
sends packets, changes safety settings, calibrates sensors, resets the MCU, or
flashes firmware. If readiness fails, the service remains running for inspection
and the command returns a failure status. A first-time engine build may still
be progressing: inspect `cv logs`, then retry `cv health`.
`CV_STM_STARTUP_TIMEOUT_S` controls the ROS startup wait (default 45 seconds);
the telemetry check allows up to eight additional seconds. Camera calibration
and synchronized MCU source timestamps remain separate requirements.

## ARUW model and parser

Fetch the pinned model with `python3 tools/model-eval/fetch_candidates.py --output /tmp/armor-models --candidate aruw-2026`.
Copy the verified ONNX file to a persistent directory in the Jetson checkout,
then set `CV_STM_MODEL_PATH` to its absolute path. The file produced by the fetcher
is `downloads/aruw-2026/5_12_icon_detector.onnx` under the selected output directory.
Keep weights and generated engines out of git.

After the first successful DeepStream build, read the exact serialized engine
path from `cv logs` and set `CV_STM_MODEL_PATH` to that file. DeepStream may
generate a name such as `5_12_icon_detector.onnx_b1_gpu0_fp32.engine` rather
than the requested `.onnx.engine` name. Restart and verify a successful
`deserialized trt engine` message, so later starts reuse the validated engine.
Keep the original ONNX for rebuilding after a TensorRT/JetPack change; an engine
validated on one Jetson runtime is not automatically portable to another.

For initial DeepStream evaluation, set `CV_STM_DETECTOR_BACKEND=deepstream`,
`CV_STM_ALLOW_NATIVE_TENSORRT_FALLBACK=false`,
`CV_STM_DEEPSTREAM_PRE_MUX_RESIZE=false`, and `CV_STM_SHOW=web`.
Inside the existing `CV_STM_EXTRA_LAUNCH_ARGS` value, set each of these **once**:

`model_type:=aruw_2026_pose task:=pose basler_width:=640 basler_height:=384 half:=false confidence_threshold:=0.5`

The launch names `basler_width`/`basler_height` select model input dimensions.
With full-sensor capture, they are not the camera-calibration resolution.
The launch argument `model_type` maps to the detector node's ROS parameter `type`.
The adapter uses BGR/255, stretching, the model's 16-class mapping, and ordered
BL/TL/TR/BR corners. An OBB parser or RGB/letterbox preprocessing is incompatible.

DeepStream can build an engine from ONNX on first load. Inspect its logs and
compare outputs before enabling reduced precision (`half:=true`). The native
TensorRT path requires an engine plus `basler_pixel_format:=bgr8`; its ARUW path
does not support raw Bayer preprocessing. Neither GPU backend was executed on a
Jetson in the Mac validation run.

Do not use `cv model-orin`, `cv model-agx`, or `cv basler-defaults` to select ARUW:
those shortcuts restore older model/camera defaults.

## Keep hypothetical commands separate from movement

Retain these arguments in the saved env for every preview session:

`transmit_enable:=false receive_enable:=true enable_fire_recommendation:=false solver_output_mode:=absolute preview_command_is_offset:=false preview_absolute_error_arrow:=true publish_zero_angles_when_no_target:=false`

**`transmit_enable:=true` is the setting that permits real SP movement packets.**
Absolute/offset chooses command meaning; it is not the arming switch. Disabled
fire recommendations do not disable gimbal movement. `cv yaw-smoke` is a separate
real-transmission tool and must not be used for a preview.

Telemetry reception is needed for camera-motion compensation even with TX off.
Use the verified Type C USB CDC path in `CV_STM_SERIAL_PORT`. The
[controller acceptance gates](../../README.md#safety-critical-tx-and-controller-gates)
are independent of this switch; RC search/scan can move the robot without CV TX.
Keep actuator power disabled during initial validation.

After configuration, use `cv preflight`, `cv start` (or `cv restart`), `cv health`,
and `cv logs`. Verify the running values inside `cv shell`:

- `ros2 param get /stm32_serial_bridge_node transmit_enable` must be false.
- `ros2 param get /aim_solver_node enable_fire_recommendation` must be false.
- `ros2 param get /armor_detector_node type` must be `aruw_2026_pose`.
- `ros2 topic echo /aim_solution` shows `used_3d`, `status`, ages, predicted horizon,
  minimum slew time, selected slot, and hypothetical angles.

For USB preview, run `ssh -N -L 18080:127.0.0.1:8080 ultraviolet@192.168.55.1`
on the host device, then open `http://127.0.0.1:18080/`. Substitute the verified
wireless hostname/IP to tunnel over Wi-Fi. The preview is not an acknowledgment
that any packet was transmitted or accepted.

## Uncalibrated preview for development

When measured intrinsics are unavailable, an operator can explicitly supply
**provisional** values to exercise PnP, tracking and solver output. These are
not calibration or evidence of accurate range, heading, spin, or motor response.
Keep `calibration_provisional:=true` with guessed intrinsics so the preview
clearly labels them. This flag does **not** change TX. `transmit_enable` is the
sole configuration switch for controller transmission. Fire recommendations
remain separately opt-in and blocked while calibration is provisional.
Normal freshness, finite-value and protocol checks still reject invalid commands.
For preview-only work keep both switches false. Save the env, then `cv restart`.

For the 1920x1200 full-sensor Basler development setup, the temporary assumption
is a 79.2-degree horizontal FOV, square pixels and a centered principal point:

```text
calibration_provisional:=true camera_fx:=1160.440656 camera_fy:=1160.440656 camera_cx:=960.0 camera_cy:=600.0 camera_calibration_width:=1920 camera_calibration_height:=1200 camera_distortion:=[0.0,0.0,0.0,0.0,0.0] hfov_degrees:=79.2 vfov_degrees:=54.681993
```

The focal length above is `1920 / (2*tan(79.2 degrees/2))`; FOV and zero
distortion are assumptions, not manufacturer calibration. The existing object
point templates are 135x55 mm (small) and 225x55 mm (large); check that the
model's four corners correspond to these light-bar endpoints, not the full
outer plate. Those sizes do not determine lens focal length or distortion.
Do not copy this temporary camera profile to a different lens, ROI or resolution.
Replace it with measured intrinsics before treating the resulting angles or range as accurate.

The live overlay uses yellow for image-center-to-plate pixel error, magenta for
model keypoints, and cyan for actual atomic `aim_solution` command error. A
blocked solver displays its exact rejection reason. A fresh attitude label
reports receive age and whether time comes from the host or MCU. Pixel error
can remain visible while 3D solving is blocked; it is not a controller command.

## Brief detection gaps and smoothing

The tracker confirms after two hits, predicts through at most six missed frames
**and** 120 ms, and retains an unmatched identity for at most 16 frames and
350 ms so short gaps can reacquire the same ID. The first limit reached wins.
`target_lock_hysteresis_frames:=6` and `target_switch_min_score_delta:=0.08`
reduce switches caused by small confidence fluctuations. Color and armor-number
association checks remain in effect.

Tune `track_max_coast_frames`, `track_max_coast_s`, `track_drop_frames`, and
`track_drop_s` in `CV_STM_EXTRA_LAUNCH_ARGS`, then `cv restart`. The independent
solver measurement-age cutoff remains 150 ms; increasing tracker coast time
does not bypass it. Coasting never refreshes the last real measurement time,
and coasted targets cannot recommend fire.

`position_filter_tau_s:=0.035` and `velocity_filter_tau_s:=0.10` smooth 3D
measurements with time-based gains. Larger values smooth more but add lag;
zero disables that filter. Velocity uses successive real measurements over
their actual interval, including missed frames, to avoid a reacquisition kick.

The preview retains predicted targets in amber with **COASTING** and the age of
the real measurement. Raw detection boxes/keypoints still represent real model
outputs and can blink during a miss. The selected-target outline and command
continue through permitted short gaps; neither is held indefinitely.

## Calibration and timing before 3D claims

Provide a valid `camera_info` topic for the **actual source image**, or configure
`camera_fx`, `camera_fy`, `camera_cx`, `camera_cy`, `camera_calibration_width`,
`camera_calibration_height`, and `camera_distortion` in the launch arguments.
Keep arrays without spaces in `CV_STM_EXTRA_LAUNCH_ARGS`, for example
`camera_distortion:=[0.0,0.0,0.0,0.0,0.0]` only when zero distortion is measured
or the incoming image is already rectified. Do not substitute guessed FOV values.
Static dimensions must be explicitly supplied and match the detections' source
resolution. ROI, resizing, rotation/flip, or a new lens requires the corresponding
calibration; `CameraInfo` takes precedence over static values.

Measure the physical corners represented by the model and set
`small_armor_width_m`, `small_armor_height_m`, `large_armor_width_m`, and
`large_armor_height_m`. Wrong plate dimensions bias depth even with good boxes.
`pose_corner_sigma_model_px` defaults to 0.5 pixels and
`pose_max_relative_depth_uncertainty` to 0.05. They screen ill-conditioned tiny
plates using an approximate endpoint-noise depth bound. Calibrate the pixel-noise
assumption on labeled frames; this is not a certified statistical confidence
interval. Source-image upscaling also scales the assumed pixel noise. A small
reprojection residual alone is insufficient evidence of accurate depth.
ARUW's complete upstream pose estimator uses depth-camera data; the Basler path
here uses monocular PnP and is not the same estimator.

Set `camera_translation_m` and `camera_mount_yaw_pitch_roll` from the actual mount
(metres and radians). The fixed-orientation tracking frame uses X right, Y down,
Z forward; logical positive command yaw is right and pitch is up. Do not reapply
the Type C motor/IMU inversion settings on the Jetson.

SP telemetry has no device timestamp. `attitude_receive_delay_s` corrects a
measured receive delay; it is not clock synchronization. Frame age is already
included. `solver_latency_s` is residual transport delay and `actuator_delay_s`
is measured controller/plant response delay; do not count inference time twice
or copy the synthetic test's delay as a robot calibration.

The solver's `yaw_max_slew_deg_s`, `pitch_max_slew_deg_s`,
`yaw_max_accel_deg_s2`, and `pitch_max_accel_deg_s2` describe the installed
controller for diagnostics. **Changing these Jetson values does not change Type C
speed limits.** `reachable` is a rest-to-rest reference-time estimate, not proof
that a motor has arrived or a firing decision is safe.

## Record and close the simulation-to-robot gap

1. Collect held-out Basler recordings of stationary single plates, nonspinning
   robots, spinning robots, two nearby robots, blank scenes, glare, and motion
   blur. Label corners/identities on sampled frames. Tune exposure and confidence
   using missed detections and false positives; public demo counts are not accuracy.
2. With TX disabled, verify corner order and calibration. A displayed detection
   without usable PnP produces `status: missing_3d_pose` and no command in the
   default 3D mode. Explicit `use_3d_solution:=false` enables the separate
   `2d_preview` path. Require `used_3d: true` before assessing robot center/phase. The `tracking_observe`
   profile starts no serial bridge, so compensated tracking needs another source
   of typed attitude telemetry.
3. During separately supervised camera motion, a stationary plate should retain
   its inertial position and absolute command. Check signs, timestamps, roll,
   mounting offset, and depth before changing smoothing.
4. Measure actual reference-to-angle response, packet delay, and missed-frame
   intervals on the robot. Replace the assumed plant/delays in the local test
   with those measurements. Camera compensation presently covers gimbal rotation
   and its lever arm; moving-chassis translation still needs synchronized odometry.
5. Only after those checks should a controls lead authorize TX-enabled motion
   tests. Keep firing disabled. The local results do not authorize deployment.

Inside `cv shell`, record `ros2 bag record /detector/armors /tracker/tracked_armors /processor/target /gimbal/attitude /aim_solution /camera_info`.
Use the actual topics reported by `ros2 topic list` if remapped. Add the raw image
topic only after verifying recording overhead; `debug_publish_image_raw:=true`
changes bandwidth and may change backend timing. Detector target snapshots alone
cannot measure false negatives.

Inspect `TRACK_TIME`, `TRACK_STATE`, `POSE_QUALITY`, and `AIM` lines in `cv logs`, plus the typed
message ages and uncertainty. Frequent robot IDs, rejected poses, stale ages,
large changes in angular speed, or commands far ahead of measured attitude are
signals to examine the recording. Do not hide them by extending coasting.

## Type C reference tested locally

The closed-loop fixture is extracted from local Type C commit
`9553b7f750a7c202b88f4ddc3974f7aefd1357bc` on
`codex/autoaim-slew-reference-fix`, based on `6aa98c1`.
It preserves the actual final-step slew rate instead of setting it to zero early.
The old code exceeded its nominal acceleration limit under tiny alternating
commands. `just verify-consolidation` and `just build sentry` passed for the fix;
**it has not been flashed**. The SP/ST wire format did not change.

The tested sentry reference uses 120/60 deg/s yaw/pitch limits, 360/240 deg/s²
acceleration limits, and 0.20 target filter alpha. Check the installed firmware
rather than assuming another branch or main has these values. Re-extract with
`python3 tools/snapshot-typec-vision.py --typec-repo /path/to/controls_typec` and
rerun the local tests after controller changes. `--check` verifies the recorded
source revision and extracted-function hash without changing the fixture.

The fixture covers reference generation. Motor PID, encoder limits, wiring, RC
interlocks, real transport timing, and joint stops remain hardware validation.

## Maintenance

Update this guide when launch names, model contracts, frame conventions, firmware
limits, or safety defaults change. Run the launch-parameter and production pipeline
tests; verify every linked path. Last updated: 2026-10-01.

## Recording command/attitude alignment

`tracking_log_enable:=true` (default) adds alignment records to the bridge's
`vision_serial_runtime_*.log` in the existing `VISION_LOG_DIR` / `vision_logs`
directory. It works with TX disabled and continues when the operator enables
TX. Save the env and run `cv restart` to change the setting.

- `tracking_alignment`: command frame timestamp, acceptance, TX configuration,
  solver status, absolute yaw/pitch, frame/measurement/prediction age, latest
  measured attitude and its timestamps, freshness, and wrapped yaw error.
- `tracking_target`: the same frame timestamp plus target ID, coasting state,
  confidence, image center, inertial pose/velocity and camera orientation.
- `serial_tx`: actual write result. A successful write does not prove that the
  MCU accepted the command or that an actuator moved.
- `serial_rx`: raw SP/ST feedback. All records now include a Jetson ROS timestamp
  in nanoseconds. The current wire protocol still has no MCU source clock.

Join target and command records by `frame_ns`; callback arrival order is not
an alignment guarantee. Measured attitude in an alignment row is the latest
received sample, not an interpolated exposure-time sample. Both timestamps and
age are retained so stale feedback cannot be mistaken for valid tracking error.
Angles/errors are radians. Keep the raw records to inspect command versus
measured-heading oscillation and to estimate relative delay offline.

Each process run rotates its log at 64 MiB and retains three backups (`.log.1`
through `.log.3`), plus its current file. Older runs are separate files; archive
or remove them after collecting a test. A quick read-only summary is:

```bash
python3 tools/analyze-tracking-alignment.py 'vision_logs/vision_serial_runtime_*.log*'
```

Use the actual log directory printed at startup if it differs. Copy all files
from the relevant run, including rotated backups, for later analysis. The
summary reports errors and timing, not a diagnosis of motor PID stability.
The commissioning yaw reference limit is now 60 degrees/s; pitch remains
60 degrees/s, with yaw/pitch acceleration caps unchanged at 360/240 degrees/s².
