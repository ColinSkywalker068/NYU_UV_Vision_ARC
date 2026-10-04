# Timing session capture

The recorders capture existing CV logs, HTTP preview stats, and Type C debugger
telemetry. They do not change TX, start/stop CV, or flash firmware. Capture files
are saved locally on the Jetson, independent of the browser preview.

## Capture fields

- Bridge log records gain same-host monotonic `steady_ns`, `bridge_frame_age_s`,
  and measured yaw/pitch rates. Existing ROS timestamps and source-timestamp flags
  remain. New C++ fields require a compatible CV rebuild and restart.
- `tools/record-tracking-session.py` copies newly appended bridge-log bytes and
  samples HTTP preview stats into a bounded session directory. It neither opens
  the serial port nor changes the service, TX, or firmware. It can collect the
  existing log format without deploying the new C++ fields.
- Type C repository `scripts/logger/typec_diagnostics.py` collects existing
  RTT channel 1 binary telemetry and channel 0 controller text through J-Link
  and installed OpenOCD. No firmware diagnostic changes are needed for those
  existing fields. It does not issue reset, halt, resume, flash, or control commands.
  Attaching a debugger is still hardware access and consumes RTT read pointers;
  run only one RTT reader and validate its effect during a supervised session.

## Record a session

From your Jetson CV checkout, one command records both sides for two minutes:

```bash
python3 tools/record-tracking-session.py --seconds 120 --typec-repo /home/ultraviolet/Documents/Controls_TypeC
```

Leave this terminal open during the test; Ctrl+C stops recording. The command
prints a unique `vision_logs/sessions/<session>` folder. Type C files are under
its `typec/` subdirectory, with attach progress in `typec-console.log`. A debugger
failure stops the combined capture and reports an error; it never resets the MCU
to recover. Test operation and TX remain under the operator's control.

For CV-only capture, omit `--typec-repo`. You can also record the two sides separately:

Use two terminals on the same Jetson to minimize host-clock ambiguity:

```bash
# CV checkout; collect the already running stack for two minutes.
python3 tools/record-tracking-session.py --seconds 120
```

```bash
# Type C checkout; J-Link SWD must be connected.
python3 scripts/logger/typec_diagnostics.py --openocd --seconds 120
```

Do not run a flash or another OpenOCD/RTT reader simultaneously. Neither command
starts CV or enables movement. Stop with Ctrl+C; no controller reset is issued.
The RTT collector stops on disconnect instead of resetting/recovering the MCU.
The CV collector defaults to 128 MiB of new bridge data and 120 seconds. Log
rotation faster than collection can lose data; log fragments at session edges
may contain incomplete records. Preserve the original runtime logs as well.

Share the CV `vision_logs/sessions/<session>` directory and Type C
`diagnostics/<session>` directory, with a short note describing when you moved
the plate and when control was enabled. RTT raw data can later be decoded with
`--replay path/to/telemetry.bin`; replay receive times are replay times, not the
original capture times. Use the original JSONL for timing analysis.

## Overlay Video Per Session

Run `python3 tools/record-tracking-session.py --seconds 120 --record-video` from
the Jetson CV checkout. Each invocation creates a new timestamped directory in
`vision_logs/sessions/` containing `overlay.mkv`, `video-frames.jsonl`,
`bridge.log`, `preview.jsonl`, and `manifest.json`. Play the MJPEG Matroska video
with VLC. The image's boxes and arrows are recorded; browser side panels are
represented by the saved preview stats. Frame index timestamps record host
arrival, not camera exposure or MCU source time.

Add `--typec-repo /home/ultraviolet/Documents/Controls_TypeC` to capture J-Link
RTT in the same session. Do not flash while that RTT recorder is running.
Ctrl+C stops the session. Neither recorder enables TX or starts/restarts CV.

The recorder requires `ffmpeg` on the Jetson host. It reads the existing HTTP
preview using Python and copies JPEG packets into Matroska without re-encoding
or opening the camera. This supports the NVIDIA FFmpeg package, which lacks
HTTP and multipart-JPEG input support. `--video-max-mib` defaults to 1024 and
bounds video independently from the bridge-log cap. A video limit ends video
capture while log capture continues. Inspect `video-console.log` and the manifest
if recording fails or ends early.

Verified on the running Jetson preview: an eight-second session produced 209
MJPEG frames at 1152x720, with a 7.943-second playable video. This verifies
recording, not real-time alignment with MCU source timestamps.

## Interpretation

- Compare target estimates, outgoing commands, controller references, measured
  angles/rates, vision gate, emergency latch, RC state, and motor-online flags.
- `accepted=1` in a Jetson log means bridge acceptance, not MCU acknowledgment.
  Serial write success also is not an acknowledgment.
- RTT receive fields are snapshots; the new-data latch may clear between samples.
  Do not count those samples as a packet-loss measurement.
- `steady_ns` is a same-host interval clock, ROS time is separate, and MCU uptime
  is another clock. Correlate host receive times with uncertainty; do not directly
  subtract MCU uptime from Jetson timestamps or claim source-clock synchronization.
- The first second of RTT attachment is marked `capture_warmup`; the ring can
  contain old frames. Initial sequence gaps are reported separately. Later RTT
  sequence gaps are not proof of CV USB packet loss.
- An advancing telemetry sequence with fixed MCU timestamps is marked stale by
  the RTT collector. Sequence gaps, corruption, and emergency/gate observations
  are retained for analysis.
- Check whether a stationary plate has a moving world-frame estimate before
  blaming motor gains. Stable command plus oscillating measured angle suggests
  a different cause than commands that oscillate before reaching the controller.
- Separate the received vision angle, applied Type C reference, and measured
  motor angle. A reference that reaches the requested angle while the motor
  remains short is a motor-following problem; reducing reference braking alone
  does not remove that stationary error.
- The older `pitch_saturation_ratio` measures the angle PID output, not the final
  motor command or speed-loop saturation. Low angle-loop saturation cannot rule
  out an actuator limit. New Type C actuator diagnostics require their own build
  and flash; use the matching Type C logger and preserve its original JSONL.
- Validate decoded roll independently of pitch. Type C SP names its quaternion
  Y Euler angle roll and its X Euler angle pitch; scalar yaw/pitch remain
  authoritative. See the [serial boundary contract](../../yolo_cpp/docs/nodes.md#stm32serialbridgenode).

The live solver reachability settings can differ from the flashed Type C limits.
Check both before interpreting reachability; changing solver metadata does not
change the MCU's motor limits. Do not silently change them in a logging patch.
