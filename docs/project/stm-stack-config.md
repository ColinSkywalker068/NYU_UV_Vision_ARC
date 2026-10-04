# Jetson STM stack configuration

<!-- SCOPE: Persistent Jetson STM stack environment configuration, boot setup, previews, transmission gates, rebuilds, and Type C flashing boundaries. -->
<!-- DOC_KIND: how-to -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when installing, editing, applying, or reviewing services/stm-stack.env. -->
<!-- SKIP_WHEN: Skip when you only need detector implementation details or model training. -->
<!-- PRIMARY_SOURCES: services/stm-stack.env.example, services/stm-stack.service, services/stm-stack.sh, services/stm-stack-entrypoint.sh -->

> **Status:** Current operational guide
> **Last Updated:** 2026-09-22

## Quick navigation

- [Repository README](../../README.md)
- [Operations runbook](runbook.md)
- [Jetson SSH access](ssh-access.md)
- [Tracked env example](../../services/stm-stack.env.example)

## Agent entry

| Signal | Value |
|--------|-------|
| Purpose | Canonical guide for the persistent Jetson CV/STM environment and its safety gates. |
| Read When | You need to install, edit, restart, preview, enable boot startup, rebuild, or determine whether flashing is required. |
| Canonical | Yes |
| Runtime Source | `services/stm-stack.env` on the Jetson |
| Tracked Example | `services/stm-stack.env.example` |
| Primary Sources | `services/stm-stack.service`, `services/stm-stack.sh`, `services/stm-stack-entrypoint.sh` |

The running configuration is stored on the Jetson at:

```text
services/stm-stack.env
```

The live file is inside the checkout and ignored by Git. Its settings survive
reboots and source updates; the tracked safe template remains separate. On Ramon
the full path is `/home/ultraviolet/Documents/CV_Detection/services/stm-stack.env`.
`cv install` renders the systemd unit with the actual checkout and env paths.
An existing legacy `/etc/cv-detection/stm-stack.env` is copied when first migrating
to the repo location; do not keep editing that old copy afterwards.

The repository keeps a safe, documented template at:

```text
services/stm-stack.env.example
```

The systemd unit in `services/stm-stack.service` loads the live file through
an absolute `EnvironmentFile` pointing to this file whenever the service starts.

## Runtime and checkout overrides

`CV_STM_REPO_ROOT` selects the actual checkout; Ramon uses
`/home/ultraviolet/Documents/CV_Detection`. `CV_STM_IMAGE` selects the compatible
container. `CV_STM_ROS_DISTRO` selects its ROS installation: `humble` remains
the default for JetPack 6, while Ramon's JetPack 7.2 / DeepStream 9.1 image uses
`jazzy`. See [the runtime recipe](../../Docker/README.md#ramon-jetpack-72--deepstream-91).
Changing the runtime SDK requires rebuilding the workspace and model engine;
ordinary launch/env changes only require `cv restart`.

## Initial setup

From the repository root on a newly configured Jetson:

```bash
sudo install -D -m 0644 services/stm-stack.env.example services/stm-stack.env
sudo ln -sf /home/ultraviolet/CV_Detection/services/stm-stack.sh /usr/local/bin/cv
cv install
cv preflight
cv enable
cv restart
```

`cv install` preserves an existing live env file. Fresh installations copy the
tracked safe example, including disabled transmission and fire recommendations. The explicit `install`
command above is for a new machine or an intentional reset. Back up a tuned live
file before replacing it.

## Edit and apply settings

Edit the live configuration:

```bash
sudoedit services/stm-stack.env
```

Apply configuration-only changes:

```bash
cv restart
cv status
cv logs
```

The service is enabled at boot with `cv enable`; it reloads the same live env
file after every reboot. `cv disable` disables automatic startup without
deleting the saved settings.

These helper commands update `CV_STM_SHOW` in the live env and restart the
service:

```bash
cv preview-web
cv preview-gui
cv preview-off
```

For a USB SSH tunnel from the operator computer:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18080:127.0.0.1:8080 \
  ultraviolet@192.168.55.1
```

Then open `http://127.0.0.1:18080/`.

The overlay and browser sidebar show `AUTO AIM ON/OFF/UNKNOWN` from the MCU's
logical vision acceptance gate, separately from `TX CONFIG`. ON means the gate
is armed, not that a target is being followed or firing is allowed. UNKNOWN
means missing, stale, or unsupported controller status. This needs the matching
Type C status firmware; there is no new env toggle. See the
[status contract](../../yolo_cpp/docs/interfaces.md#controller-auto-aim-acceptance).

## Transmission safety

For safe preview, the launch arguments must include:

```text
solver_output_mode:=absolute
preview_absolute_error_arrow:=true
transmit_enable:=false
receive_enable:=true
```

With `transmit_enable:=false`, the preview can compute and draw the absolute
command, current attitude, error, and arrow while the normal bridge does not
send controller-facing movement commands.

Changing only `solver_output_mode` does not enable transmission. Real SP
movement transmission from the Jetson requires both:

```text
solver_output_mode:=absolute
transmit_enable:=true
```

Treat `transmit_enable:=true` as an arming change. Type C firmware still has
independent acceptance gates:

- Normal DBUS left switch up allows vision control.
- Right switch middle, emergency stop, `ROBOT_STOP`, or dial above 300 blocks it.
- A fresh SX command with `scan_control_valid=1` can override the physical
  left-switch choice through its `allow_vision_control` bit.
- `cv yaw-smoke` is a separate real-packet transmitter.

Confirm mechanical safety and controller gates before enabling TX.

## When a rebuild or flash is required

An env-only change needs only `cv restart`.

After changing CV C++ or ROS launch source (`uv` uses the saved
`CV_STM_IMAGE`, so the build and boot service use the same runtime):

```bash
cd /home/ultraviolet/CV_Detection
uv build yolo_cpp
cv restart
```

After changing Type C firmware source or its compile-time sentry configuration:

```bash
cd /home/ultraviolet/controls_typec
just build sentry
# Flash only during a controlled hardware-safe procedure:
just flash pyocd sentry --serial 032200157416303030303032
```

Editing the Jetson env does not rebuild or flash the STM32. Editing Type C
firmware does not take effect until that firmware is built and flashed.

## Bounded tracking gaps and reacquisition

After building the updated workspace, the tracker defaults to time-based
coasting/identity retention. Optional settings in `CV_STM_EXTRA_LAUNCH_ARGS`:

```text
track_max_coast_frames:=0 track_max_coast_s:=0.12
track_drop_frames:=0 track_drop_s:=0.35
pose_innovation_confirm_m:=0.20 pose_max_reprojection_rms_px:=5.0
```

Zero frame caps make the time budgets authoritative; positive caps retain the
legacy additional frame limit. Saved positive overrides are preserved; remove
or deliberately change them to use the new policy. The solver's independent
150 ms last-real-measurement limit remains, including pipeline latency.
Increasing coasting does not make old data fresh. Fire remains disabled on
coasted targets and opt-in overall. See [manual validation](../validation/autoaim-manual-checks.md#brief-detection-gaps-and-smoothing)
for quality gates, spin transitions and diagnostic interpretation.

This update changes tracked ROS messages, so rebuild both active packages via
`uv build yolo_cpp` before restarting; an env-only restart cannot apply source
changes. Neither installing nor updating this code edits the live env.

## Motion-aware tracking update

The solver publishes bounded yaw/pitch angular rates with its absolute 3D
command, and the bridge puts them in the existing SP velocity fields in
radians/second. No extra env switch enables this: trusted rates require fresh,
continuous measured tracking and become zero on acquisition, coasting, stale
input, or identity/armor-slot changes. Angles remain usable when rates are
unavailable. Sentry firmware validates those rates for motion-aware braking
and feeds the final bounded reference rate into both motor speed loops.

This requires `uv build yolo_cpp` (both active ROS packages), `cv restart`,
and the matching built/flashed Type C Sentry firmware. Saved env-only changes
still need only `cv restart`. The update preserves the env and existing rate,
acceleration, travel and motor-output limits; `transmit_enable` remains the
normal bridge TX switch. See [manual validation](../validation/autoaim-manual-checks.md#moving-target-following-motion-aware-braking-and-rate-feedforward)
for limits, fallback behavior and commissioning checks.

## Keeping the example current

When a setting should become the documented team default, update
`services/stm-stack.env.example` and this guide in Git. Do not commit secrets,
credentials, private keys, or machine tokens. The live Git-ignored `services/stm-stack.env` remains the
runtime source of truth for an individual robot.

## USB disconnect recovery

Set `CV_STM_SERIAL_PORT` to the controller's `/dev/serial/by-id/...` path.
A `/dev/ttyACM0` name can change after unplugging; the stable identity lets the
bridge reopen the same controller even when Linux assigns a different number.
The service mounts the host device directory so newly enumerated ports remain
visible inside its container.

The bridge detects USB hangups and fatal read/write errors, closes the old
connection, and retries once per second. It also retries when the controller is
absent at startup. Partial packets and queued input are discarded between
connections; commands are not queued for replay. Reconnecting preserves the
saved `transmit_enable` setting. With TX enabled, fresh commands can resume as
soon as the link recovers: use `transmit_enable:=false` before recovery tests.

`cv logs` reports disconnect/reconnect events and warns after two seconds
without valid SP/ST telemetry, then every five seconds. A port that opens
without telemetry can still indicate a controller stream or protocol problem;
reopening a port does not guarantee recovery of a stalled controller.
Fresh attitude is required for the absolute-command preview. Fire development
remains disabled with `enable_fire_recommendation:=false`.

The virtual-port regression is included in
`colcon test --packages-select yolo_cpp`. It covers a missing startup port,
repeated unplug/replug, CRC rejection, partial-frame isolation, silence
warnings, and zero emitted bytes with TX disabled. It opens no robot hardware.

## Editing over wireless VS Code SSH

For Orin2, connect VS Code Remote SSH to `jetson-orin2-wifi`
(`ultraviolet@rm-orin2.local`) and open `/home/ultraviolet/CV_Detection`.
Open `services/stm-stack.env` directly. An administrator can grant
the robot operator edit access once with:

```bash
sudo chgrp ultraviolet services/stm-stack.env
sudo chmod 0664 services/stm-stack.env
```

Preview helper changes preserve these file permissions. Save the env, then
run `cv restart` in the remote terminal. Check `cv status` and `cv logs`.
TX is controlled by `transmit_enable` in `CV_STM_EXTRA_LAUNCH_ARGS`;
changing target color, output mode, or web preview does not arm transmission.
Keep `enable_fire_recommendation:=false` for development. A live absolute
command preview also requires fresh Type C attitude telemetry.

For a wireless preview tunnel on the host device:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18082:127.0.0.1:8080 jetson-orin2-wifi
```

Open `http://127.0.0.1:18082/`.
