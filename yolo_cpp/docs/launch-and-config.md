# Launch and Config

For the ARUW pose/tracking branch, use the [manual evaluation guide](../../docs/validation/autoaim-manual-checks.md)
for parser selection, source-resolution calibration, timing, quality limits, and
preview-only settings. Its setup differs from the older OBB model defaults.


Related pages: [Overview](./overview.md) | [Pipeline](./pipeline.md) | [Nodes](./nodes.md) | [Interfaces](./interfaces.md) | [Legacy Pipeline Notes](./legacy-pipeline-notes.md) | [Development Notes](./development-notes.md)

## Launch Files

- [`basler.launch.py`](../launch/basler.launch.py): Basler + DeepStream field stack with detector, tracker, processor, 3D solver, and STM bridge
- [`realsense.launch.py`](../launch/realsense.launch.py): RealSense + native TensorRT detector, tracker, processor, and solver stack without the serial bridge
- [`stm.launch.py`](../launch/stm.launch.py): compatibility alias that forwards to `basler.launch.py`
- [`zerocopy.launch.py`](../launch/zerocopy.launch.py): component-container launch for the combined perception path

## Common Commands

Build just this package:

```bash
colcon build --packages-select cv_auto_aim_interfaces yolo_cpp --symlink-install
```

Launch the Basler field stack:

```bash
uv run yolo_cpp basler.launch.py
```

Launch the RealSense path:

```bash
uv run yolo_cpp realsense.launch.py
```

## Main Knobs

The launch files expose the settings that matter most during bring-up:

- camera selection such as `camera_type`, RealSense `rs_*` launch args, and Basler `basler_*` launch args
- Basler high-FPS capture through `basler_pixel_format`, `basler_source_width`, `basler_source_height`, `basler_fps`, and `basler_uncapped_fps`
- DeepStream full-FOV resize through `pre_mux_resize`, `pre_mux_width`, and `pre_mux_height`
- DeepStream Bayer conversion through `bayer_conversion_backend`; `cuda` is the default and falls back to `cpu`/`bayer2rgb` when unavailable
- DeepStream conversion profiling through `deepstream_profile_stages` and `deepstream_profile_stage_interval`
- detector backend selection through `detector_backend`
- fallback control through `allow_native_tensorrt_fallback`
- default enemy color through `target_color` until STM or robot-id feedback overrides it
- color filtering through `use_model_color_classes` and `enable_color_verification`; defaults trust model class colors and keep HSV disabled
- target-frame capture through `save_target_frames`, `target_frame_save_max`, and `target_frame_jpeg_quality`; saved JPEGs and a CSV index go under `pictures/` next to `vision_logs/` by default
- preview through `show` and `preview_max_fps`; use `show:=false` to disable preview rendering, `show:=gui` for the local OpenCV window, or `show:=web` for the browser preview
- browser preview through `show:=web`, `preview_http_enabled`, `preview_http_bind_address`, `preview_http_port`, and `preview_http_jpeg_quality`; open `http://<robot-host>:8080/`, `/stream.mjpg`, `/snapshot.jpg`, or `/stats.json` from any same-network laptop without ROS
- tracker tuning such as `track_min_hits`, `track_max_coast_frames`, and `track_drop_frames`; defaults favor fast armor-plate switching during spin
- 3D pose through `enable_3d_pose`, `camera_info_topic`, `camera_frame_id`, and optional static `camera_fx/fy/cx/cy`
- tracker 3D gating through `association_max_3d_distance_m`
- processor frame selection through `processor_target_frame`
- solver geometry through `hfov_degrees`, `vfov_degrees`, `use_3d_solution`, `solver_latency_s`, and `default_bullet_speed_mps`
- solver fire recommendations through `enable_fire_recommendation`; it defaults to `false` and must be set to `true` explicitly for competition
- serial settings such as `serial_port`, `serial_baudrate`, and `serial_verbose`
- STM command units through `transmit_angles_in_degrees`

The DeepStream template shipped with the package lives at [`config/deepstream_bestmerge_nvinfer.txt`](../config/deepstream_bestmerge_nvinfer.txt).

The default tracker profile is tuned for fast rotating armor plates: `track_min_hits:=1`, `track_max_coast_frames:=1`, `track_drop_frames:=4`, `association_max_center_distance_ratio:=0.30`, `association_min_iou:=0.0`, `target_switch_min_score_delta:=0.02`, and `target_lock_hysteresis_frames:=0`. Increase the coast, lock, and score-delta values when you want steadier lock over faster switching.

## Basler High-FPS Source Formats

The Basler + DeepStream path defaults to `basler_pixel_format:=bayer_rggb8` with `basler_source_width:=1920` and `basler_source_height:=1200` for the `a2A1920-160ucBAS` full-resolution high-FPS case. This keeps USB traffic at raw 8-bit Bayer instead of full-resolution BGR. Keep `basler_use_full_sensor:=true` to preserve the full FOV. Enable `pre_mux_resize:=true` to resize that full-FOV frame before `nvstreammux`; with the default 640x640 model box, 1920x1200 becomes 640x400 and `nvstreammux` pads it to the model input. The default `bayer_conversion_backend:=cuda` uses the in-process CUDA Bayer resize element when the build includes CUDA and no full-resolution color/debug branch is active. `show:=gui` and `show:=web` use a cheap downscaled preview branch; HSV verification, raw image publishing, and target-frame saving still request full-resolution BGR and can force the CPU `bayer2rgb` path.

On the observed `a2A1920-160ucBAS` / serial `41923000` setup, direct Pylon probing reported `Binning*`, `Decimation*`, and `Scaling*` GenICam nodes as absent, so binning is not a supported runtime knob for this camera environment. `Width` and `Height` are still writable, but they are ROI/crop controls and should not be used when full FOV is required.

Useful source-format choices:

- `bayer_rggb8`: max-FPS color capture path; the GStreamer graph uses CUDA Bayer resize when available or `bayer2rgb` as the CPU fallback.
- `yuy2`: compatibility fallback when Bayer conversion is unavailable or the camera UserSet favors YUYV.
- `bgr8`: simple/debug path; full-resolution BGR can be bandwidth-limited near 60 FPS on USB3.
- `mono8`: raw grayscale path for camera-only throughput tests when color is not needed.

For a camera-only throughput probe on the robot, launch with `basler_uncapped_fps:=true` and compare the startup log lines for requested caps, negotiated caps, pixel format, and `BslResultingAcquisitionFrameRate`.

## Browser Preview

Set `show:=web` to render the annotated detector preview for the browser without opening a local GUI window:

```bash
uv run yolo_cpp basler.launch.py show:=web
```

Open the preview from a same-network laptop:

```text
http://rm-standard.local:8080/
```

The preview server exposes:

- `/`: browser page with the MJPEG stream and detector statistics
- `/stream.mjpg`: raw MJPEG stream for tools or dashboards
- `/snapshot.jpg`: latest rendered preview frame
- `/stats.json`: latest FPS, backend, camera, candidate count, target state, and detector latency

Use `show:=gui` when you want the local OpenCV window instead. The `gui` and `web` modes are intentionally separate. Use `preview_http_port:=<port>` if another service already owns `8080`. Keep `preview_http_bind_address:=0.0.0.0` for robot-LAN viewing; use `127.0.0.1` only for local-only testing.

On a headless Jetson, prefer `show:=web` because it does not require X11 display access.

## Which Launch Should I Use?

Use `basler.launch.py` when you want the real field-facing Basler stack with controller integration. Its default solver mode is `absolute`, so STM RX yaw/pitch radians are added to the solved aim offsets, then the bridge sends outgoing yaw/pitch radians to STM because the current controls firmware expects radians. `stm.launch.py` remains available only as a compatibility alias.

Use `realsense.launch.py` when you want a smaller RealSense loop for camera, native TensorRT inference, tracking, 3D processor tuning, and aiming without serial transport in the way. Its default solver mode is `offset`, since there may be no STM attitude feedback.
