[![pipeline status](https://gitlab.com/nyu-robomaster/CV_Detection/badges/main/pipeline.svg)](https://gitlab.com/nyu-robomaster/CV_Detection/-/commits/main)

# CV Detection

NYU Robomaster ROS 2 C++ auto-aim vision pipeline for camera capture, YOLO/TensorRT/DeepStream armor detection, temporal tracking, 3D target processing, ballistic aim solving, and STM32 serial output.

## For Readers

Several repository docs start with machine-readable metadata and navigation tables because agents use them to route work safely. If you are reading manually, use each doc's `Quick Navigation` table and skip past `SCOPE`, `DOC_KIND`, `Agent Entry`, and maintenance blocks. The fastest human paths are [Quick Start](#quick-start), [Project runbook](docs/project/runbook.md), [Jetson reflash guide](docs/project/jetson-reflash.md), and [Launch and config](yolo_cpp/docs/launch-and-config.md).

## Active Runtime

The runtime is centered on:

- `yolo_cpp/`: C++ ROS 2 nodes, launch files, DeepStream config, tests, package docs, and active model assets.
- `cv_auto_aim_interfaces/`: custom ROS 2 messages shared by detector, tracker, processor, solver, and STM bridge.
- `services/`: Jetson/container helper scripts, STM stack service, preflight, health, and benchmark tools.
- `Docker/`: active container images and Docker bring-up notes.
- `docs/` and `yolo_cpp/docs/`: current project, architecture, runbook, and package documentation.

## Quick Start

On the robot, use the `cv` operator command for the headless Basler + DeepStream + STM32 service.
Install or refresh the shortcut once:

```bash
sudo ln -sf /home/ultraviolet/CV_Detection/services/stm-stack.sh /usr/local/bin/cv
```

Run the competition stack:

```bash
cv install
cv preflight
cv restart
cv logs
```

Common operator commands:

```bash
cv status
cv health
cv preview-gui
cv preview-web
cv preview-off
cv basler-defaults
cv model-orin
cv model-agx
cv yaw-smoke
```

If the `cv` symlink is not installed yet, use the helper directly from the repo root:

```bash
./services/stm-stack.sh install
./services/stm-stack.sh preflight
./services/stm-stack.sh restart
./services/stm-stack.sh logs
```

For development shells and manual ROS launches, use the `uv` wrapper from the repo root:

```bash
uv docker
uv build yolo_cpp
uv run yolo_cpp orin.launch.py
uv run yolo_cpp basler.launch.py
uv run yolo_cpp realsense.launch.py
```

The robot-local fallback model path is configured at:

```text
yolo_cpp/config/default_model_path.txt
```

Override it with `CV_STM_MODEL_PATH` or a launch `model_path:=...` argument. `cv model-orin` and `cv model-agx` now normalize the service to the same Basler/DeepStream defaults and select `yolo_cpp/models/kushalv2-nano.engine`; `model-nano` remains a backward-compatible alias for `model-orin`.

## View Camera and Bounding Boxes Remotely

The existing web preview shows camera frames with detector bounding boxes and
the selected target overlay. It does not require Foxglove or a desktop display.
Connect the Basler camera and use a compatible model first. A running service
alone does not prove the detector is producing frames.

1. **Jetson:** run `cv preflight`, then `cv preview-web`. This persists web
   preview mode and **restarts the full CV/STM service**, not just the viewer.
   Use only with the robot safely disabled; the normal STM bridge remains part
   of this stack. If `cv` is not installed, use `./services/stm-stack.sh`
   from the repository root in its place.
2. **Jetson:** run `cv logs` and check for camera/inference errors. Run
   `hostname -I` to find its current Wi-Fi address. To distinguish interfaces,
   use `ip -br -4 addr`; do not use Docker bridge addresses.
3. **Mac, same Wi-Fi:** open `http://JETSON_WIFI_IP:8080/` in a browser,
   replacing `JETSON_WIFI_IP` with that address. Confirm frames keep changing
   and boxes appear when a supported target is detected.

Alternatively, tunnel the preview over either Wi-Fi or USB SSH:

- **Mac, USB:** run `ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18080:127.0.0.1:8080 ultraviolet@192.168.55.1`.
- **Mac, Wi-Fi:** use the same command with the Jetson's Wi-Fi IP instead of
  `192.168.55.1`. Substitute your Linux username or saved SSH alias if different.
- Keep that terminal open and visit `http://127.0.0.1:18080/` on the Mac.
  Press Ctrl+C to close the tunnel; this does not stop CV.

USB forwarding requires a working USB network/SSH connection, not merely a
charging cable. Wi-Fi requires reachability between clients; sharing an SSID
does not guarantee this on networks with client isolation. If SSH works but
direct HTTP is blocked, use the tunnel.

**Security:** the current preview binds to `0.0.0.0:8080` and has no login or
TLS. Use a trusted robot network; do not expose it to the internet. An SSH
tunnel protects the tunnel connection but does not disable the direct LAN
listener. This is separate from Nav's restricted Foxglove port 8765.

**Troubleshooting:** on the Jetson, run `curl --fail --max-time 3 -o /dev/null http://127.0.0.1:8080/`
to check HTTP reachability and `cv logs` for startup failures. A reachable page
without fresh frames is not a successful camera test. A missing camera, model
failure, or stale installed binary must be resolved first. After source
updates, rebuild the active stack with `uv build yolo_cpp` in the configured
development environment before restarting; pulling source alone does not
update compiled code.

**Turn off:** `cv preview-off` persists preview-off and restarts the full
service. Preview mode persists across service restarts; automatic startup
still depends on whether the service is enabled.

## Command Overlay and Stationary Aiming Checks

The preview draws a white image-center crosshair. Command text reads
`/bounding_box_angles` (yaw, pitch in radians), the same topic consumed by
the STM bridge; it is not a serial-transmission acknowledgment. The cyan
arrow is opt-in: use `solver_output_mode:=offset use_3d_solution:=false
preview_command_is_offset:=true` together as launch arguments for a 2D
relative-error test. Set those arguments in `CV_STM_EXTRA_LAUNCH_ARGS` in
`/etc/cv-detection/stm-stack.env` for the service, then restart it.
Do not enable the offset arrow for absolute solver output.

Positive yaw draws right; positive pitch draws up, matching the Type C
logical command frame. Near-zero correction draws a cyan
ring. Arrows are bounded to the image; command numbers remain unscaled.
Invalid commands, missing target authorization, or command/authorization
updates older than 250 ms hide the arrow. This timeout only affects the
display; it is not an MCU watchdog. The center is the image center, not a
verified camera-to-gimbal boresight calibration.

Before motor tests, verify radians, axis signs, and whether firmware expects
absolute angles, incremental offsets, or rates. Absolute solving rejects
missing/nonfinite attitude or attitude received more than 250 ms ago. Camera
frames carry a local steady-clock capture marker, and the solver interpolates
a one-second attitude history at the resulting frame timestamp. The default
`attitude_sync_tolerance_ms:=30.0` applies only when the frame is just outside
the stored history; frames bracketed by two attitude samples are interpolated.
Anything farther outside the history fails closed.

The solver emits throttled `AIM_SYNC` log lines with `aligned`, `rejected`,
alignment mode, and nearest-sample offset. Inspect them with
`journalctl -u stm-stack -f | grep AIM_SYNC`. For an offline dump, record
`/detector/armors`, `/tracker/target`, `/attitude_data`, `/aim_solution`,
`/bounding_box_angles`, and `/override` with `ros2 bag record`.

The capture marker records when a frame enters host processing rather than a
hardware exposure timestamp, and STM attitude is timestamped when received by
the Jetson. This removes inference-time mixing but does not remove camera or
serial transport latency. Calibrate camera intrinsics and boresight, then test
stationary targets during controlled gimbal motion before increasing rate
limits or prediction. STLINK supports controller debugging but does not
replace runtime feedback; halting the MCU can interrupt its control/watchdog
loops. Keep actuators disabled for debugger bring-up.

### Type C Sentry SP/ST Integration

The bridge uses the SP/ST transport from upstream `cv-dropback-fix` commit
`4c7a568` (protocol introduced in `e053dd4`), with local safety corrections.
It sends 29-byte SP packets with absolute yaw/pitch in radians and zero
velocity/acceleration fields. ST status never overwrites SP attitude, and ST
shooter heat is not interpreted as bullet speed. No 1.35 angle gain is used.

TX defaults off. `receive_enable:=true transmit_enable:=false` opens the
serial device for telemetry only. Specify `serial_port:=/dev/ttyACM0` or the
verified stable device path. `transmit_enable:=true` with offset mode or
degree units is rejected. Target loss or command/authorization expiry after
250 ms requests mode 0; this cannot replace the MCU's own safety watchdog.
Commands must be finite and within the SP firmware's angle range.

#### Safety-critical TX and controller gates

`transmit_enable` is the host-side arming switch. The normal CV bridge can send
real controller-facing SP commands only when both of these launch arguments are
present:

```text
solver_output_mode:=absolute
transmit_enable:=true
```

Keep `transmit_enable:=false` for every preview-only session. Absolute solving,
the displayed command, STM attitude reception, and the command-minus-attitude
preview arrow continue to work while TX is disabled. `cv yaw-smoke` is a
separate hardware command tool and must also be treated as real transmission.

The Type C sentry firmware applies a second, independent acceptance gate:

- With the normal DBUS remote, left switch **up** enables vision control.
- Right switch **middle** selects gimbal auto-rotate and disables vision control.
- Emergency stop, `ROBOT_STOP`, or a dial value above `300` rejects vision control.
- A fresh sentry bridge `SX` command with `scan_control_valid=1` overrides the
  physical left-switch choice; its `allow_vision_control` bit becomes the gate.
- With the remote offline, the committed sentry config permits vision control
  only while the vision link is online, but keeps offline automatic firing
  disabled (`SENTRY_REMOTE_OFFLINE_AUTON_ALLOW_FIRE=0`).

Real gimbal movement therefore requires the host TX condition **and** a Type C
acceptance condition. SP mode `0` requests no control, mode `1` requests gimbal
aiming, and mode `2` additionally requests firing subject to the firmware's
separate firing gates.

The systemd service loads `/etc/cv-detection/stm-stack.env` every time it
starts, including after a reboot. Enable automatic boot startup without
changing the saved preview or launch settings with:

```bash
cv enable
```

Use `cv disable` to stop automatic boot startup without deleting the saved
environment file. Configuration-only changes require only:

```bash
cv restart
```

After changing CV C++ or launch source, rebuild and restart:

```bash
cd /home/ultraviolet/CV_Detection
uv build yolo_cpp
cv restart
```

After changing Type C source or `application/robot_configs/robot_sentry.h`,
build the sentry target explicitly. With actuators disabled, stop CV before the
flash so it releases the MCU serial device, then restart CV afterward:

```bash
cd /home/ultraviolet/controls_typec
just build sentry
cv stop
just flash pyocd sentry --serial 032200157416303030303032
cv start
```

The inspected `controls_typec` sentry config has yaw IMU direction +1,
pitch IMU direction -1, and both gimbal motor directions reversed. Those
conversions are already applied at the MCU's logical-coordinate boundary:
do not copy its raw IMU/motor inversions into the Jetson again. The external
logical convention is positive yaw right and positive pitch up. A flashed
binary/config match and physical direction check are still required.

The stationary preview can remain in 2D offset mode with TX disabled. Before
actuation, use absolute mode, verified fresh feedback and calibrated camera
geometry; disable the offset arrow for absolute output. No firmware flashing
is needed merely to adopt the existing SP/ST wire format.

Hardware-free regression: in a sourced ROS container run
`python3 yolo_cpp/tools/test_sp_bridge_pty.py`. It uses a pseudo-terminal and
isolated ROS domain 89; it never opens the physical board. Tests cover SP
encoding/CRC, SP attitude, ST isolation, bad CRC rejection, stale-command
mode 0, and target loss.
Run `python3 yolo_cpp/tools/test_solver_sentry.py` in the same sourced
container to verify sentry right/up command signs and absolute-mode feedback
expiry in isolated ROS domain 90. Neither test needs a camera or physical MCU.

## Documentation

- [Documentation map](docs/README.md)
- [C++ runtime overview](yolo_cpp/docs/overview.md)
- [Pipeline stages](yolo_cpp/docs/pipeline.md)
- [ROS nodes](yolo_cpp/docs/nodes.md)
- [Launch and config](yolo_cpp/docs/launch-and-config.md)
- [Container and bring-up runbook](CPP_CONTAINER_RUNBOOK.md)
- [Project runbook](docs/project/runbook.md)

## Legacy Archive

Historical Python tooling, including the former `yolov8_ros/` package, Mac/CoreML prototypes, old setup notes, stale model variants, and retired Docker files are archived under `legacy/`. Treat them as reference material unless a task explicitly targets legacy behavior.
