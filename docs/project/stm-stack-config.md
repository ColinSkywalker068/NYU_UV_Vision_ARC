# Jetson STM stack configuration

<!-- SCOPE: Persistent Jetson STM stack environment configuration, boot setup, previews, transmission gates, rebuilds, and Type C flashing boundaries. -->
<!-- DOC_KIND: how-to -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when installing, editing, applying, or reviewing /etc/cv-detection/stm-stack.env. -->
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
| Runtime Source | `/etc/cv-detection/stm-stack.env` on the Jetson |
| Tracked Example | `services/stm-stack.env.example` |
| Primary Sources | `services/stm-stack.service`, `services/stm-stack.sh`, `services/stm-stack-entrypoint.sh` |

The running configuration is stored on the Jetson at:

```text
/etc/cv-detection/stm-stack.env
```

`/etc` is part of the Jetson's root filesystem. It is outside this Git
repository, so edits to the live file are persistent across reboots but are not
committed or pushed.

The repository keeps a safe, documented template at:

```text
services/stm-stack.env.example
```

The systemd unit in `services/stm-stack.service` loads the live file through
`EnvironmentFile=-/etc/cv-detection/stm-stack.env` whenever the service starts.

## Initial setup

From the repository root on a newly configured Jetson:

```bash
sudo install -D -m 0644 services/stm-stack.env.example /etc/cv-detection/stm-stack.env
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
sudoedit /etc/cv-detection/stm-stack.env
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

After changing CV C++ or ROS launch source:

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

## Keeping the example current

When a setting should become the documented team default, update
`services/stm-stack.env.example` and this guide in Git. Do not commit secrets,
credentials, private keys, or machine tokens. The live `/etc` file remains the
runtime source of truth for an individual robot.

## Editing over wireless VS Code SSH

For Orin2, connect VS Code Remote SSH to `jetson-orin2-wifi`
(`ultraviolet@rm-orin2.local`) and open `/home/ultraviolet/CV_Detection`.
Open `/etc/cv-detection/stm-stack.env` directly. An administrator can grant
the robot operator edit access once with:

```bash
sudo chgrp ultraviolet /etc/cv-detection/stm-stack.env
sudo chmod 0664 /etc/cv-detection/stm-stack.env
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
