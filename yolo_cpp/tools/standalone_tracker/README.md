# Standalone Tracker Replay

This is a macOS-friendly harness for testing the tracker without ROS, camera capture, YOLO,
TensorRT, DeepStream, or OpenCV. It compiles only:

- `yolo_cpp/src/tracker.cpp`
- `yolo_cpp/src/tracking_math.cpp`
- `yolo_cpp/tools/standalone_tracker/tracker_replay.cpp`

Run the built-in dropout/hold scenario:

```bash
./yolo_cpp/tools/standalone_tracker/run_tracker_replay.sh
```

Run the corner sign check:

```bash
./yolo_cpp/tools/standalone_tracker/run_tracker_replay.sh --scenario corners
```

Run a multi-frame detector dropout with the default one-frame coast:

```bash
./yolo_cpp/tools/standalone_tracker/run_tracker_replay.sh --scenario long_dropout
```

Trace the tracker -> processor -> solver -> bridge handoff:

```bash
./yolo_cpp/tools/standalone_tracker/run_tracker_replay.sh --scenario long_dropout --trace-control
./yolo_cpp/tools/standalone_tracker/run_tracker_replay.sh --scenario long_dropout --trace-control --publish-zero-angles-when-no-target false
./yolo_cpp/tools/standalone_tracker/run_tracker_replay.sh --scenario long_dropout --trace-control --track-max-coast-frames 3
```

Replay a captured or hand-written CSV:

```bash
./yolo_cpp/tools/standalone_tracker/run_tracker_replay.sh --scenario csv --input /path/to/frames.csv
```

CSV format:

```text
frame,dt,center_x,center_y,width,height
0,0.0167,190,365,40,20
1,0.0167,none,,,,
2,0.0167,210,350,40,20
```

The output prints one row per frame with selected track state, coasting count, tracked center,
predicted center, and solver-compatible yaw/pitch signs. With the current math:

- left of image center means positive yaw
- right of image center means negative yaw
- above image center means negative pitch
- below image center means positive pitch

The harness mirrors `apply_angle_gain`, so `--angle-gain` values below `1.0` are clamped to `1.0`
just like the active solver code.

If a frame has no selected target, the split ROS solver would publish zero angles when
`publish_zero_angles_when_no_target=true`.
