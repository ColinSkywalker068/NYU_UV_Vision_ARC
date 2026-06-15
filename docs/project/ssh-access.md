# Jetson SSH Access

<!-- SCOPE: SSH access patterns for CV Detection Jetson hosts only. Does not define runtime, Docker, ROS, or VNC behavior. -->
<!-- DOC_KIND: how-to -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when you need to SSH into a robot host over same-Wi-Fi mDNS names or the wired USB/Ethernet link. -->
<!-- SKIP_WHEN: Skip when you only need VNC, Docker runtime commands, or ROS 2 launch guidance. -->
<!-- PRIMARY_SOURCES: docs/project/runbook.md, docs/project/infrastructure.md, docs/project/jetson-reflash.md, docs/project/jetson-vnc.md -->

> **Status:** Current operator guide
> **Last Updated:** 2026-06-15

## Quick Navigation

| Need | Section |
|------|---------|
| Same-Wi-Fi SSH | [2. Wi-Fi SSH](#2-wi-fi-ssh) |
| Wired SSH | [3. Wired SSH](#3-wired-ssh) |
| Troubleshooting | [4. Troubleshooting](#4-troubleshooting) |

## Agent Entry

| Field | Guidance |
|-------|----------|
| Purpose | Canonical SSH entry points for the current robot hosts. |
| Read When | You need to reach a Jetson shell before using Docker, ROS, camera, or service commands. |
| Skip When | You already have shell access or only need desktop sharing. |
| Canonical | Yes |
| Read Next | [Operations Runbook](runbook.md), [Infrastructure](infrastructure.md), [Jetson Reflash](jetson-reflash.md) |
| Primary Sources | `docs/project/runbook.md`, `docs/project/infrastructure.md`, `docs/project/jetson-reflash.md`, `docs/project/jetson-vnc.md` |

## 1. SSH User

Use the `ultraviolet` account on all current robot hosts:

```bash
ssh ultraviolet@<host>
```

## 2. Wi-Fi SSH

Use the `.local` hostnames when your laptop and the robot are on the same Wi-Fi network or lab LAN, and the network allows client-to-client traffic.

```bash
ssh ultraviolet@rm-standard.local
ssh ultraviolet@rm-sentry.local
```

If the hostname does not resolve, confirm both devices are on the same network. Many guest or campus Wi-Fi networks block peer-to-peer traffic, so `.local` SSH will fail even when both devices have internet access.

## 3. Wired SSH

Use the wired address when you are connected through the robot's direct cable link.

```bash
ssh ultraviolet@192.168.55.1
```

This path does not depend on Wi-Fi name resolution. Use it when the robot is connected by wire and the direct link is available.

## 4. Troubleshooting

| Symptom | Likely Cause | Check |
|---------|--------------|-------|
| `.local` host does not resolve | Not on the same Wi-Fi or mDNS blocked | Verify both machines are on the same Wi-Fi/lab LAN, then retry `ssh ultraviolet@rm-standard.local` or `ssh ultraviolet@rm-sentry.local`. |
| `.local` SSH hangs or times out | Client-to-client traffic blocked | Switch to the wired link and use `ssh ultraviolet@192.168.55.1`. |
| Wired SSH fails | Cable or host-side link issue | Confirm the robot is actually on the direct wired connection and the link is up. |

## Maintenance

**Update Triggers:**
- Robot hostnames change
- The wired direct-link address changes
- SSH user or authentication flow changes

**Verification:**
- [ ] `.local` hostnames match the deployed Wi-Fi robot names
- [ ] Wired address matches the direct cable link
- [ ] Quick navigation links resolve

