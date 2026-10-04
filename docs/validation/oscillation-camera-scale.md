# Oscillation and Camera Scale Investigation

<!-- SCOPE: Evidence and prepared changes for the stationary-plate oscillation investigation; not a calibrated camera profile or deployment approval. -->
<!-- DOC_KIND: record -->
<!-- DOC_ROLE: working -->
<!-- READ_WHEN: Diagnosing repeated crossings of a single plate or preparing the provisional camera correction. -->
<!-- SKIP_WHEN: Looking for normal startup commands. -->
<!-- PRIMARY_SOURCES: tools/diagnose-camera-motion.py, yolo_cpp/src/detection_postprocess.cpp, yolo_cpp/src/armor_detector_node.cpp, yolo_cpp/src/aim_prediction.cpp -->

## Quick Navigation

| Need | Source |
|---|---|
| Prepared camera values | [Inert provisional profile](../../yolo_cpp/config/calibration/ramon-basler-provisional-20261002.json) |
| Offline fit | [Camera-motion diagnostic](../../tools/diagnose-camera-motion.py) |
| Capture another session | [Timing capture](timing-session-capture.md) |
| Deployment and calibration rules | [Manual checks](autoaim-manual-checks.md) |

## Agent Entry

This is conditional evidence from session `20261002T202333Z-a2367adc`, not a
general Basler calibration. The profile file is inert; its camera values were
subsequently merged into the live env with operator approval. CV restarted with
TX=false and fire=false, both verified live. The operator later re-enabled TX
for testing. Never infer the current TX state from this deployment record.

## Findings

The active ARUW path stretches 1920x1200 source frames to 640x384 with no
padding, then restores X by 3 and Y by 3.125 exactly once. PnP consumes those
source coordinates with intrinsics explicitly sized for 1920x1200. No duplicate
resize or missing aspect-ratio conversion was found.

The old provisional focal lengths are 1160.440656 pixels, implying 79.2 degrees
horizontal FOV. A shared 3.15 multiplier produces 3655.3880664 pixels and
29.4301 degrees HFOV. The principal point, square pixels and zero distortion
remain assumptions. Known armor dimensions do not independently establish
focal length when range and plate orientation are unknown.

The diagnostic fits alternating half-second blocks and evaluates the other
blocks against the training set's fixed direction. Windows are seconds after
capture start; frame capture timestamps choose the samples. This test assumes
the target and camera pivot remain stationary. Operator confirmation of that
assumption is still absent; target/chassis translation and telemetry angular
scale can confound an effective focal-length estimate.

| Window / axis | Original held-out angular RMS | Focal correction | Timing-only fit |
|---|---:|---:|---:|
| 63–69 s / pitch | 4.411 degrees | 0.043 degrees | 4.421 degrees |
| 74–82 s / yaw | 5.830 degrees | 0.045 degrees | 5.830 degrees |
| 108–115 s / yaw | 5.440 degrees | 0.044 degrees | 5.473 degrees |

All three windows selected 3.15 and zero additional shift on a 10 ms grid
when scale and timing were fitted together. The grid searched focal multipliers
0.5–5 and time shifts -150–150 ms. This is evidence against a large timing
offset as the sole explanation, not proof of source-clock synchronization.
Samples still used host receive time for attitude.

The reported 3D world position also oscillates before prediction. Consequently,
false target velocity can amplify the error through prediction. Simply reducing
acceleration cannot correct this camera-to-angle scale error. A smaller
acceleration limit also reduces available braking.

The recorded MCU accepted vision commands. Its reference-following error was
approximately 3.7 degrees RMS in the yaw oscillation windows and 2.1 in pitch.
A simple lag fit suggests about 210–215 ms effective yaw phase lag and 110 ms
pitch lag, with residual errors around 1.5–1.7 degrees. Those values combine
motor response and other dynamics; do not insert them as pure transport delays.
Pitch angle and speed integral gains are zero in the inspected Sentry source;
yaw speed has integral gain 70. Pitch oscillation therefore cannot be attributed
to a configured pitch integral term alone. These source values are not live PID
state measurements.

## Prepared Correction and Braking Experiment

The JSON profile contains the proposed camera values and a preview-only TX/fire
configuration. It is deliberately not automatically loaded. For an eventual
approved trial, merge its arguments into `CV_STM_EXTRA_LAUNCH_ARGS`, replacing
existing occurrences while retaining all unrelated saved settings. Check for
an overriding CameraInfo source. Restart only when approved, then verify live
parameters and TX=false before a manually supervised test. Do not change
Type C limits in the same first comparison.

The existing Type C profile already bounds approach speed by the minimum of
maximum speed, 8 times reference error, and square-root stopping speed. Its
distance is measured from the generated reference, not the actual motor.
A simulation-only candidate combines a tanh approach curve and an 80 ms
braking-delay margin. The Type C script
`scripts/tests/compare_vision_approach.py` extracts the current production C
function and compiles it on the host; it never builds or flashes firmware.

In a synthetic underdamped plant (natural frequency 20 rad/s, damping 0.35),
incorrect angular gain 3.15 left both profiles oscillating after 12 seconds.
The candidate reduced the final three-second RMS from 3.41 to 2.13 degrees on
yaw and 2.77 to 1.92 on pitch. With correct angular scale, the existing profile
settled within 0.25 degrees in 0.70/0.67 seconds (yaw/pitch); the candidate took
0.72/0.82 seconds. Moving-target lag also slightly increased. Therefore the
candidate was initially retained as an experiment. The operator subsequently
requested deployment; the tanh curve and 80 ms margin were enabled for Sentry
and flashed with verification. Subsequent operator-requested limits are yaw
45 deg/s and 240 deg/s², pitch 35 deg/s and 160 deg/s². The numeric simulation
comparison above describes the earlier 27/20 deg/s configuration, not these
later limits. No measured-velocity damping or PID gains were changed.

64 axis/sign/gain/damping/motion/profile scenarios checked reference velocity
and acceleration bounds and mirror symmetry. They do not validate real motor
PID behavior. The older CV closed-loop fixture still contains 60 deg/s limits
and a synthetic plant; passing those tests is not validation of current hardware.

If oscillation persists after the camera correction, use measured angular
velocity relative to target velocity to investigate damping and stopping
distance. Avoid adding derivative-on-error blindly: target changes can kick
that derivative. Identify motor response with a supervised, bounded test
before changing PID gains or adding a second feedback loop around the existing
motor controller.

## Repeat the Offline Check

Later runtime logs exposed a separate vertical offset: `compensate_gravity`
remained true with default bullet speed 18 m/s. Over 2,856 accepted samples
(54.7 seconds), the solver added a median 1.755 degrees upward at estimated
range 2.025 m. Reconstructing its gravity term matched command pitch within
numerical precision. The measured pitch was approximately 0.137 degrees from
the raised command while the plate center remained about 122 source pixels
below the crosshair. Fire recommendations being false does not disable this
gravity term. For a camera-centering test, explicitly select
`compensate_gravity:=false` and restart only when approved. That change has not
been deployed by this investigation.

Use a NumPy-enabled Python environment. Run
`python3 tools/test-diagnose-camera-motion.py` for synthetic identification checks.
Run `python3 tools/diagnose-camera-motion.py --help` for the log/window interface.
The tool reads files and writes a report only; no ROS/device/config access.
It rejects mixed target identities and coasted samples, and reports search
boundary fits rather than silently accepting extrapolated estimates.

Full intrinsic/distortion calibration still needs multiple known-pattern views;
see [OpenCV's calibration tutorial](https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html).
Without a board, stationary target observations at several manually established
gimbal headings can validate effective angular scale. Keep the chassis and
target fixed, retain the same lens/focus/ROI, record both axes and both motion
directions, and evaluate a separate held-out sweep. This cannot establish full
distortion, metric depth, or camera-to-gimbal extrinsics.

## Maintenance

Update after a controlled camera-scale trial or motor system-identification
session. Verify profile dimensions/FOV consistency, diagnostic synthetic tests,
and the distinction between prepared and deployed settings.

Last Updated: 2026-10-02
