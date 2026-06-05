# Jetson VNC Access

<!-- SCOPE: Same-network Jetson remote desktop setup for operator configuration and debugging only. Does not define the robot runtime, ROS topics, or public internet access. -->
<!-- DOC_KIND: how-to -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when you need free non-RealVNC desktop access to one or more Jetsons on the same Wi-Fi or lab LAN. -->
<!-- SKIP_WHEN: Skip when you only need SSH, Docker, ROS launch, or STM stack operations. -->
<!-- PRIMARY_SOURCES: docs/project/runbook.md, docs/project/infrastructure.md, services/stm-stack.sh, services/start-container.sh, Docker/docker-compose.yml -->

> **Status:** Current operator guide
> **Last Updated:** 2026-06-05

## Quick Navigation

| Need | Section |
|------|---------|
| Recommended setup | [1. Recommended Pattern](#1-recommended-pattern) |
| Configure one Jetson | [3. Jetson Setup](#3-jetson-setup) |
| Connect from another computer | [4. Client Connection](#4-client-connection) |
| Configure multiple Jetsons | [5. Multi-Jetson Naming And Ports](#5-multi-jetson-naming-and-ports) |
| Safer SSH tunnel mode | [6. SSH Tunnel Mode](#6-ssh-tunnel-mode) |
| Headless or black-screen fixes | [7. Headless And Display Notes](#7-headless-and-display-notes) |
| Troubleshooting | [9. Troubleshooting](#9-troubleshooting) |

## Agent Entry

| Field | Guidance |
|-------|----------|
| Purpose | Free same-network remote desktop setup for Jetson configuration, camera inspection, and GUI debugging. |
| Read when | A Jetson needs desktop access from a laptop without RealVNC, especially when configuring several robots. |
| Skip when | SSH-only access is enough, or the task concerns Docker, ROS launch files, STM serial, or detector behavior. |
| Canonical | Yes, for VNC operator setup. |
| Read next | [runbook.md](runbook.md) for runtime operations, [infrastructure.md](infrastructure.md) for host inventory. |
| Primary sources | `docs/project/runbook.md`, `docs/project/infrastructure.md`, `services/stm-stack.sh`, `services/start-container.sh`, `Docker/docker-compose.yml` |

## 1. Recommended Pattern

Use `x11vnc` on each Jetson and a free VNC client on the laptop. This shares the Jetson's existing desktop instead of creating a separate RealVNC cloud account.

| Choice | Recommendation | Why |
|--------|----------------|-----|
| VNC server on Jetson | `x11vnc` | Free, package-managed, works with ordinary VNC clients, and mirrors the real display. |
| VNC client on macOS | Built-in Screen Sharing, TurboVNC Viewer, or TigerVNC Viewer | No RealVNC account required; TurboVNC's official Mac package is preferable when the Homebrew TigerVNC viewer has poor scaling behavior. |
| VNC client on Linux | Remmina or TigerVNC Viewer | Both are free and available through package managers. |
| VNC client on Windows | TigerVNC Viewer | Free viewer with direct `host::port` support. |
| Naming | Unique hostnames plus Avahi `.local` names | Keeps many Jetsons discoverable without memorizing changing IP addresses. |
| Ports | Keep `5900` on every Jetson | Same port is fine because each Jetson has a different hostname or IP. |
| Security | Prefer SSH tunnel on shared/campus Wi-Fi | VNC screen traffic is not a good public-network boundary. |

Use direct VNC only on a trusted robot/lab network. Do not port-forward VNC from a router to the internet.

## 2. Network Assumptions

| Requirement | Check |
|-------------|-------|
| Jetson and client are on the same Wi-Fi or lab LAN | `ping rm-jetson-01.local` or `ping <jetson-ip>` from the client. |
| The network allows client-to-client traffic | Some campus or guest Wi-Fi networks block peer devices; use a robot router, phone hotspot, Ethernet switch, or SSH/VPN path if this happens. |
| SSH is available | `ssh <user>@rm-jetson-01.local` or `ssh <user>@<jetson-ip>`. |
| A Jetson desktop session exists | `x11vnc` mirrors the active X display, commonly `:0` or `:1`; it needs an active X desktop or a dummy display. |

## 3. Jetson Setup

Run these commands on each Jetson over SSH or with a local keyboard and monitor.

### 3.1 Pick A Unique Hostname

Use a name that matches the robot or Jetson label:

```bash
sudo hostnamectl set-hostname rm-jetson-01
sudo reboot
```

After reboot, reconnect using the new name when `.local` resolution works:

```bash
ssh ultraviolet@rm-jetson-01.local
```

If `.local` does not resolve from the client, use the Jetson IP from:

```bash
hostname -I
```

### 3.2 Install Free VNC And Discovery Packages

```bash
sudo apt update
sudo apt install -y openssh-server avahi-daemon x11vnc
sudo systemctl enable --now ssh avahi-daemon
```

### 3.3 Set A VNC Password

Run this as the Jetson login user that owns the desktop session, usually `ultraviolet`:

```bash
mkdir -p ~/.vnc
x11vnc -storepasswd ~/.vnc/passwd
chmod 600 ~/.vnc/passwd
```

Use a real password. The VNC password is separate from the Linux login password.

### 3.4 Install A Startup Service

Create a reusable systemd template so the same service file works for any Jetson user:

```bash
sudo tee /etc/systemd/system/x11vnc@.service >/dev/null <<'EOF'
[Unit]
Description=x11vnc desktop sharing for %i
After=display-manager.service network-online.target
Wants=network-online.target

[Service]
Type=simple
User=%i
ExecStart=/usr/bin/x11vnc -display :0 -auth guess -forever -loop -noxdamage -repeat -shared -rfbauth /home/%i/.vnc/passwd -rfbport 5900
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF
```

Enable it for the current user:

```bash
ls -l /etc/systemd/system/x11vnc@.service
echo "$USER"
sudo systemctl daemon-reload
sudo systemctl enable --now "x11vnc@${USER}.service"
systemctl status "x11vnc@${USER}.service" --no-pager
```

The template filename is `x11vnc@.service`. The enabled instance is `x11vnc@<user>.service`, for example `x11vnc@ultraviolet.service`. If `echo "$USER"` prints `root`, replace `${USER}` with the normal desktop user, such as `ultraviolet`.

The service listens on VNC port `5900`. It is intentionally host-level operator access; it is not part of the ROS 2 runtime or Docker service. Do not add `-localhost no`; on `x11vnc` 0.9.16, `-localhost` is a boolean flag and the trailing `no` is treated as an invalid option that prevents the VNC port from opening.

## 4. Client Connection

Use one of these from the other computer on the same network:

| Client OS | Free Client | Connect To |
|-----------|-------------|------------|
| macOS | Screen Sharing | `vnc://rm-jetson-01.local:5900` or `vnc://<jetson-ip>:5900` |
| macOS | RealVNC Viewer | `rm-jetson-01.local::5900` when normal clipboard copy/paste matters |
| macOS | TurboVNC Viewer | `rm-jetson-01.local::5900` when a free native viewer with better scaling is preferred |
| macOS | TigerVNC Viewer | `rm-jetson-01.local::5900` |
| Linux | Remmina | Protocol `VNC`, server `rm-jetson-01.local:5900` |
| Linux | TigerVNC Viewer | `vncviewer rm-jetson-01.local::5900` |
| Windows | TigerVNC Viewer | `rm-jetson-01.local::5900` or `<jetson-ip>::5900` |

On macOS, the built-in Screen Sharing app is located under `/System/Library/CoreServices/Applications/Screen Sharing.app`. From Terminal, this also opens it:

```bash
open vnc://rm-jetson-01.local:5900
```

## 5. Multi-Jetson Naming And Ports

Give every Jetson a different hostname. Keep VNC on `5900` on each Jetson.

| Robot Or Board | Hostname | Direct VNC Target | SSH Target |
|----------------|----------|-------------------|------------|
| First Jetson | `rm-jetson-01` | `rm-jetson-01.local:5900` | `ultraviolet@rm-jetson-01.local` |
| Second Jetson | `rm-jetson-02` | `rm-jetson-02.local:5900` | `ultraviolet@rm-jetson-02.local` |
| Third Jetson | `rm-jetson-03` | `rm-jetson-03.local:5900` | `ultraviolet@rm-jetson-03.local` |

Use router DHCP reservations when the router is under team control. Use `.local` hostnames or a small team inventory when IP addresses change often.

If Avahi reports a name conflict, it may advertise a suffixed name such as `rm-standard-2.local` even though `hostname` still prints `rm-standard`. Check `systemctl status avahi-daemon --no-pager`; the active advertised name appears in the status line, for example `avahi-daemon: running [rm-standard-2.local]`.

## 6. SSH Tunnel Mode

Use this mode on shared Wi-Fi, campus Wi-Fi, or any network where direct VNC feels too exposed.

On the Jetson, edit the service template:

```bash
sudo nano /etc/systemd/system/x11vnc@.service
```

Add `-localhost` to the `ExecStart` line, then restart the service:

```bash
sudo systemctl daemon-reload
sudo systemctl restart x11vnc@$USER.service
```

From the client, open one tunnel per Jetson. Keep each terminal running while using VNC:

```bash
ssh -N -L 5901:localhost:5900 ultraviolet@rm-jetson-01.local
ssh -N -L 5902:localhost:5900 ultraviolet@rm-jetson-02.local
```

Then connect the VNC viewer to the local forwarded port:

| Jetson | Tunnel Command Uses | VNC Viewer Target |
|--------|---------------------|-------------------|
| `rm-jetson-01` | `5901:localhost:5900` | `localhost:5901` |
| `rm-jetson-02` | `5902:localhost:5900` | `localhost:5902` |

Direct same-Wi-Fi mode is simpler. SSH tunnel mode is safer because only SSH is exposed to the network.

If VNC client scaling and clipboard behavior remain frustrating on macOS, use the official TurboVNC Mac viewer package instead of the Homebrew TigerVNC formula. If the team is willing to install a non-VNC remote desktop server on the Jetson, NoMachine is another free local-network option with stronger resize and clipboard behavior, but it is outside this VNC-specific setup.

## 7. Headless And Display Notes

`x11vnc` mirrors the existing X display. If the Jetson has no monitor, no dummy HDMI plug, and no active desktop session, the VNC viewer may show a black screen or fail to connect.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Black screen | No active desktop session on display `:0` | Log in locally once, enable automatic login for lab use, or attach a dummy HDMI plug. |
| Tiny, stretched, or strange resolution after unplugging DisplayPort or HDMI | Jetson lost monitor EDID and guessed a fallback display mode | Use a dummy HDMI or DisplayPort plug, attach a monitor once, or set a persistent display mode in the desktop settings. |
| `x11vnc` cannot open display | Desktop is Wayland or no X server exists | Jetson desktop images usually use Xorg; if not, switch the login session to Xorg before relying on `x11vnc`. |
| VNC works only after local login | Display manager does not start the user session | Enable automatic login only on trusted lab robots, or use SSH plus command-line setup instead. |

When the active display is not `:0`, find it with:

```bash
ls /tmp/.X11-unix/
ps -ef | grep -E 'Xorg|Xwayland'
```

If the socket is `X1` and Xorg was launched with `/run/user/1000/gdm/Xauthority`, use:

```ini
ExecStart=/usr/bin/x11vnc -display :1 -auth /run/user/1000/gdm/Xauthority -forever -loop -noxdamage -repeat -shared -rfbauth /home/%i/.vnc/passwd -rfbport 5900
```

For purely headless administration, SSH is usually better than VNC. For visual camera setup or GUI debugging, a dummy HDMI plug is often the least fragile Jetson option.

## 8. Relation To CV Detection Runtime

VNC is for operator access only. It does not replace the competition service or ROS launch flow.

| Task | Use |
|------|-----|
| Configure Wi-Fi, camera vendor tools, desktop settings, or files | VNC or SSH |
| Start or restart the robot vision stack | [runbook.md](runbook.md) and `./services/stm-stack.sh restart` |
| Inspect service logs | `./services/stm-stack.sh logs` |
| Validate ROS nodes and topics | `./services/stm-stack.sh health` |
| Run OpenCV preview windows | VNC can display the host desktop when `CV_STM_SHOW=true`, but match runs should usually keep preview off. |

The Docker paths use host networking. Avoid assigning VNC-related port changes inside containers unless a future service explicitly owns that behavior.

## 9. Troubleshooting

| Problem | Check | Fix |
|---------|-------|-----|
| `.local` hostname does not work | `systemctl status avahi-daemon --no-pager` on the Jetson | Use the IP from `hostname -I`, restart Avahi, or check whether the client OS supports mDNS. |
| `.local` hostname resolves to a suffixed name such as `rm-standard-2.local` | Avahi status contains `Host name conflict` or `running [rm-standard-2.local]` | Use the suffixed name, flush the client mDNS cache, restart Avahi, or choose a unique hostname with `sudo hostnamectl set-hostname <name>`. |
| `Unit x11vnc@...service could not be found` | `ls -l /etc/systemd/system/x11vnc@.service` | Recreate the template from section 3.4, run `sudo systemctl daemon-reload`, and check that the unit name ends in `.service`, not `.sevice` or `.servcice`. |
| Client cannot connect | `ss -ltnp | grep 5900` on the Jetson | Start `x11vnc@$USER.service`, verify the Jetson IP, or use SSH tunnel mode. |
| Service is active but `ss -ltnp | grep 5900` is empty | `journalctl -u x11vnc@$USER.service -n 80 --no-pager` | Remove invalid options such as `-localhost no`, then verify the active display and auth file. |
| Connection times out on Wi-Fi | Ping and SSH also fail | The network may block peer devices; use a robot router, hotspot, or Ethernet. |
| macOS Screen Sharing opens a stale target | `nc -vz <jetson-name>.local 5900` succeeds, but Screen Sharing still fails | Quit Screen Sharing, reconnect by exact `.local:5900` or IP target, or use TigerVNC with `host::5900`. |
| Clipboard paste does not work | VNC connects but clipboard does not sync | Use TurboVNC or TigerVNC instead of macOS Screen Sharing. In Ubuntu Terminal, paste with `Ctrl+Shift+V`; plain `Ctrl+V` is not terminal paste. Do not add `-clipboard` unless `x11vnc -opts` confirms support; on `x11vnc` 0.9.16 it can be an invalid option that prevents port `5900` from opening. |
| Password fails | `ls -l ~/.vnc/passwd` | Rerun `x11vnc -storepasswd ~/.vnc/passwd` as the desktop user. |
| Viewer connects but cannot control | Viewer is in view-only mode or desktop permissions block input | Disable view-only mode in the client and restart `x11vnc`. |
| Desktop feels slow | Weak Wi-Fi or high display resolution | Use 5 GHz Wi-Fi or Ethernet, reduce desktop resolution, or prefer SSH for non-visual work. |

## 10. External References

| Source | Use |
|--------|-----|
| [NVIDIA Jetson VNC setup](https://developer.nvidia.com/embedded/learn/tutorials/vnc-setup) | Confirms same-network Jetson VNC workflow and Jetson display-session caveat. |
| [TigerVNC viewer manual](https://tigervnc.org/doc/vncviewer.html) | Client address syntax such as `host::port`. |
| [x11vnc manual](https://man.archlinux.org/man/x11vnc.1) | Server options such as `-auth guess`, `-forever`, and `-shared`. |
| [TurboVNC documentation](https://turbovnc.org/Documentation/Documentation) | Official Mac viewer package and viewer option reference. |
| [NoMachine Jetson remote desktop guide](https://jetsonhacks.com/2023/12/03/nomachine-jetson-remote-desktop/) | Non-VNC fallback for Jetson desktop access when scaling and clipboard UX matter more than staying on VNC. |
| [Ubuntu VNC community guide](https://help.ubuntu.com/community/VNC) | Security background and SSH port-forwarding pattern. |

## Maintenance

**Update Triggers:**
- Team standard Jetson usernames, hostnames, router setup, or Wi-Fi topology changes.
- The robot image moves away from an Xorg desktop or stops shipping a graphical session.
- A repository service begins owning VNC, RDP, VPN, or remote desktop configuration.
- Competition operations decide to require SSH-tunneled VNC only.

**Verification:**
- [ ] All repository links resolve.
- [ ] Commands are tested on at least one current Jetson image before match-day use.
- [ ] Port `5900` is not exposed through public router forwarding.
- [ ] Multi-Jetson hostnames match the team inventory.
- [ ] `Last Updated` reflects the last operational verification date.

**Last Updated:** 2026-06-05
