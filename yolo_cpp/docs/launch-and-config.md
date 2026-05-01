# Launch and Config

Related pages: [Overview](./overview.md) | [Pipeline](./pipeline.md) | [Nodes](./nodes.md) | [Interfaces](./interfaces.md) | [Legacy Pipeline Notes](./legacy-pipeline-notes.md) | [Development Notes](./development-notes.md)

## Launch Files

- [`stm.launch.py`](../launch/stm.launch.py): full modular stack with detector, tracker, solver, and STM bridge
- [`realsense.launch.py`](../launch/realsense.launch.py): lighter RealSense-focused modular stack without the serial bridge
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
- solver geometry through `hfov_degrees` and `vfov_degrees`
- serial settings such as `serial_port`, `serial_baudrate`, and `serial_verbose`

The DeepStream template shipped with the package lives at [`config/deepstream_bestmerge_nvinfer.txt`](../config/deepstream_bestmerge_nvinfer.txt).

## Which Launch Should I Use?

Use `stm.launch.py` when you want the real field-facing stack with controller integration. Use `realsense.launch.py` when you want a smaller loop for camera, inference, tracking, and aiming without serial transport in the way.
