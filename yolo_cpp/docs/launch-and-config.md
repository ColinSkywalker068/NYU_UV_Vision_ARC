# Launch and Config

Related pages: [Overview](./overview.md) | [Pipeline](./pipeline.md) | [Nodes](./nodes.md) | [Interfaces](./interfaces.md) | [Legacy Pipeline Notes](./legacy-pipeline-notes.md) | [Development Notes](./development-notes.md)

## Launch Files

- [`stm.launch.py`](../launch/stm.launch.py): full modular stack with detector, tracker, processor, 3D solver, and STM bridge
- [`realsense.launch.py`](../launch/realsense.launch.py): RealSense-focused detector, tracker, processor, and solver stack without the serial bridge
- [`zerocopy.launch.py`](../launch/zerocopy.launch.py): component-container launch for the combined perception path

## Common Commands

Build just this package:

```bash
colcon build --packages-select yolo_cpp --symlink-install
```

Launch the full stack:

```bash
uv run yolo_cpp stm.launch.py
```

Launch the RealSense path:

```bash
uv run yolo_cpp realsense.launch.py
```

## Main Knobs

The launch files expose the settings that matter most during bring-up:

- camera selection such as `camera_type`, `realsense_*`, and `basler_*`
- detector backend selection through `detector_backend`
- fallback control through `allow_native_tensorrt_fallback`
- tracker tuning such as `track_min_hits`, `track_max_coast_frames`, and `track_drop_frames`
- 3D pose through `enable_3d_pose`, `camera_info_topic`, `camera_frame_id`, and optional static `camera_fx/fy/cx/cy`
- tracker 3D gating through `association_max_3d_distance_m`
- processor frame selection through `processor_target_frame`
- solver geometry through `hfov_degrees`, `vfov_degrees`, `use_3d_solution`, `solver_latency_s`, and `default_bullet_speed_mps`
- serial settings such as `serial_port`, `serial_baudrate`, and `serial_verbose`
- STM command units through `transmit_angles_in_degrees`

The DeepStream template shipped with the package lives at [`config/deepstream_bestmerge_nvinfer.txt`](../config/deepstream_bestmerge_nvinfer.txt).

## Which Launch Should I Use?

Use `stm.launch.py` when you want the real field-facing stack with controller integration. Its default solver mode is `absolute`, so STM RX yaw/pitch radians are added to the solved aim offsets, then the bridge converts the outgoing yaw/pitch to degrees for STM TX.

Use `realsense.launch.py` when you want a smaller loop for camera, inference, tracking, 3D processor tuning, and aiming without serial transport in the way. Its default solver mode is `offset`, since there may be no STM attitude feedback.
