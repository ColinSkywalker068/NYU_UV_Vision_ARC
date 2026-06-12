# Jetson Orin Nano Developer Kit Reflash Guide

<!-- SCOPE: Optimized one-time or recovery reflash procedure for Jetson Orin Nano Developer Kit robot hosts used by CV Detection. Focuses on restoring the known-good L4T 36.2 host stack for the current DeepStream 6.4 Docker image. -->
<!-- DOC_KIND: how-to -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when a Jetson Orin Nano Developer Kit must be reset, downgraded, or aligned with the current DeepStream 6.4 competition container. -->
<!-- SKIP_WHEN: Skip when you only need VNC access, normal Docker preflight, ROS launch parameters, or source-level detector debugging. -->
<!-- PRIMARY_SOURCES: Team Orin Nano setup notes supplied 2026-06-08, services/setup-orin-nano.sh, services/stm-stack.sh, services/start-container.sh, Docker/Dockerfile.cplusplus, docs/project/runbook.md, NVIDIA JetPack and DeepStream compatibility docs. -->

> **Status:** Current recovery guide for the DeepStream 6.4 stack
> **Last Updated:** 2026-06-08

## Quick Navigation

| Need | Section |
|------|---------|
| Decide whether to use this guide | [1. Use Case](#1-use-case) |
| Prepare the GameLab Linux PC and Jetson | [3. Flash Host Preparation](#3-flash-host-preparation) |
| SDK Manager selections | [4. SDK Manager Flash](#4-sdk-manager-flash) |
| Minimal post-flash robot setup | [5. Post-Flash Setup](#5-post-flash-setup) |
| Validate the Docker/DeepStream stack | [6. Runtime Validation](#6-runtime-validation) |
| Optional CAN, PWM, and desktop setup | [7. Optional Hardware And Desktop Steps](#7-optional-hardware-and-desktop-steps) |
| Troubleshooting | [8. Troubleshooting](#8-troubleshooting) |

## Agent Entry

| Field | Guidance |
|-------|----------|
| Purpose | Restore a Jetson Orin Nano Developer Kit to the host runtime generation expected by the current CV Detection DeepStream 6.4 Docker image. |
| Read when | The Nano was reset, upgraded too far, or shows Nano-only DeepStream/NvBuf/CUDA failures while the same container works on AGX. |
| Skip when | The host is already on L4T 36.2 and you only need `cv preflight`, logs, or launch tuning. |
| Canonical | Yes, for Orin Nano reflash and first boot recovery. |
| Read next | [runbook.md](runbook.md) for normal operations, [jetson-vnc.md](jetson-vnc.md) for desktop access after the base stack works. |
| Primary sources | Team setup notes, `services/setup-orin-nano.sh`, `services/stm-stack.sh`, `services/start-container.sh`, `Docker/Dockerfile.cplusplus`, NVIDIA JetPack and DeepStream compatibility docs. |

## 1. Use Case

Use this guide for the current competition image:

```text
nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon
```

That image provides DeepStream 6.4, TensorRT 8.6, CUDA runtime 12.2, ROS 2 Humble, and Pylon support. The host Jetson should therefore stay in the matching JetPack/L4T family rather than the latest JetPack release.

| Target | Expected Version |
|--------|------------------|
| Jetson Linux / L4T | R36.2.0 |
| JetPack | 6.0 Developer Preview |
| DeepStream source | Docker image, not SDK Manager host install |
| DeepStream inside container | 6.4.0 |
| CUDA driver/runtime pairing | 12.2 / 12.2 inside validation output |
| TensorRT inside container | 8.6 |

Do not select a generic "latest JetPack 6.x" SDK Manager target when the goal is to keep the current DeepStream 6.4 image. Newer JetPack hosts can expose different NVIDIA driver and multimedia runtime behavior to the same container.

This guide assumes the NVIDIA Developer Kit carrier board: the board has a microSD slot and uses the J14 Force Recovery header. A production Orin Nano module on a custom carrier may use different recovery pins, storage, and SDK Manager target wording.

## 2. Before You Wipe The Nano

The reflash wipes the robot host. Save robot-specific state first.

```bash
mkdir -p ~/nano-backup
cp -a /etc/cv-detection ~/nano-backup/cv-detection-etc 2>/dev/null || true
cp -a ~/CV_Detection/yolo_cpp/models ~/nano-backup/models 2>/dev/null || true
cp -a ~/CV_Detection/vision_logs ~/nano-backup/vision_logs 2>/dev/null || true
```

Also save any local Pylon package archives, model exports, TensorRT engines, and operator notes that are not committed to git. Do not copy service account passwords, Wi-Fi passwords, or private tokens into repository files; retrieve those from the team credential store during setup.

## 3. Flash Host Preparation

Use the GameLab Linux PC or another native Ubuntu Linux host. macOS is fine for SSH and SD-card imaging, but it is not the supported path for Jetson recovery flashing because SDK Manager and the USB recovery flow expect a Linux host.

1. Insert the microSD card into the Jetson Orin Nano Developer Kit carrier board.
2. Short J14 pins `9` and `10` to enter Force Recovery mode.
3. Power the Jetson from the wall supply.
4. Connect Jetson USB-C to the GameLab Linux PC with a USB-C to USB-A cable.
5. Confirm the Linux PC sees the Jetson in recovery mode:

```bash
lsusb | grep -i nvidia
```

If `lsusb` shows no NVIDIA recovery device, re-check the jumper, cable, and power sequence before opening SDK Manager.

## 4. SDK Manager Flash

Open NVIDIA SDK Manager on the GameLab Linux PC. Use the team NVIDIA account from the credential store when prompted.

| SDK Manager Field | Select |
|-------------------|--------|
| Product Category | `Jetson` |
| Host Machine | Enabled only if the GameLab PC needs SDK Manager host tools; otherwise leave host extras minimal. |
| Target Hardware | `Jetson Orin Nano Developer Kit` when available; otherwise the closest `Jetson Orin Nano` / `Jetson Orin Nano 8GB Developer Kit` target that exposes SD-card flashing |
| SDK Version | `JetPack 6.0 Developer Preview` / `L4T 36.2` |
| DeepStream | Disabled |
| OEM Configuration | Pre-config |
| Username | `ultraviolet` |
| Password | Team robot password |
| Storage Device | SD Card / microSD |

Disabling SDK Manager DeepStream is intentional. The robot service runs DeepStream from the Docker image. The host install only needs the Jetson OS, NVIDIA driver/BSP stack, container runtime support, and ordinary system packages.

After SDK Manager finishes:

1. Remove the Force Recovery jumper.
2. Power-cycle the Jetson.
3. SSH from the GameLab PC:

```bash
ssh ultraviolet@192.168.55.1
```

Verify the host generation before installing robot software:

```bash
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core
```

Expected signal:

```text
# R36 (release), REVISION: 2.0
nvidia-l4t-core 36.2...
```

## 5. Post-Flash Setup

Install only the pieces needed for the CV runtime first. Skip browser, Snap, VNC, Jetson-IO, CAN, and desktop customization until the Docker/DeepStream stack validates.

```bash
sudo apt update
sudo apt install -y git ca-certificates curl can-utils busybox nano vim nvidia-l4t-dla-compiler nvidia-l4t-gstreamer nvidia-l4t-multimedia
```

Set up repository access. Generate a new SSH key only when the reflashed Nano does not already have one backed up.

```bash
ssh-keygen -t ed25519 -C "orin-nano"
cat ~/.ssh/id_ed25519.pub
```

Add the public key to the team GitLab account, then clone:

```bash
cd ~
git clone git@gitlab.com:nyu-robomaster/CV_Detection.git
cd CV_Detection
```

Run the existing setup script, but treat it as a bootstrap helper rather than a perfect source of truth:

```bash
./services/setup-orin-nano.sh
```

If Docker installation was already completed or the script stops after installing Docker packages, continue with the explicit runtime steps below.

```bash
sudo ln -sf /home/ultraviolet/CV_Detection/services/uv.sh /usr/local/bin/uv
sudo docker pull nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon
./services/stm-stack.sh install
```

Restore `/etc/cv-detection/stm-stack.env` only after reviewing it for stale host-specific paths. The service installer creates a default file when none exists.

## 6. Runtime Validation

Run preflight without requiring STM serial while the bench setup is still incomplete:

```bash
CV_STM_REQUIRE_SERIAL=false ./services/stm-stack.sh preflight
```

Start the foreground stack for direct log inspection:

```bash
CV_STM_REQUIRE_SERIAL=false ./services/stm-stack.sh run
```

In another shell, validate the NVIDIA container runtime view:

```bash
sudo docker exec -it cv-stm-stack bash -lc '
set -x
cat /etc/nv_tegra_release
deepstream-app --version-all
gst-inspect-1.0 nvinfer | grep -E "Filename|Version"
gst-inspect-1.0 pylonsrc | head -40
'
```

Target output:

```text
R36, REVISION: 2.0
DeepStreamSDK 6.4.0
CUDA Driver Version: 12.2
CUDA Runtime Version: 12.2
TensorRT Version: 8.6
nvinfer plugin Version 6.4.0
pylonsrc available
```

If an engine was generated on a newer JetPack/L4T host, rebuild or regenerate it on the reflashed Nano before using it for competition validation. The host and container should both be aligned before trusting DeepStream or TensorRT crash results.

## 7. Optional Hardware And Desktop Steps

Only run these after [6. Runtime Validation](#6-runtime-validation) passes.

| Step | When To Run | Command Or Doc |
|------|-------------|----------------|
| Wi-Fi setup | The robot is not on the lab network | `sudo nmtui` |
| VNC setup | Desktop access is needed after SSH works | [jetson-vnc.md](jetson-vnc.md) |
| Jetson-IO PWM | Controls or other hardware require 40-pin PWM configuration | `sudo /opt/nvidia/jetson-io/jetson-io.py` |
| CAN register setup | The Jetson communicates over CAN hardware | Use the CAN section of `services/setup-orin-nano.sh` and verify with controls. |
| Browser or Snap workarounds | A desktop browser is needed for manual setup | Keep out of the baseline CV recovery path. |

The CV detector-to-STM field stack uses serial by default, not CAN. Do not spend reflash time on CAN/PWM unless the specific robot hardware needs it.

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| SDK Manager cannot see the Nano | Recovery mode did not latch, USB cable is charge-only, or the jumper/power order is wrong | Re-seat the J14 9-10 jumper, power-cycle, and re-run `lsusb | grep -i nvidia`. |
| SDK Manager only shows newer JetPack versions | The archived JetPack 6.0 Developer Preview target is hidden or unavailable | Use SDK Manager archive/older-version controls or install the Jetson Linux 36.2 driver package path on a Linux host. Do not choose latest JetPack 6.x for this recovery flow. |
| `cat /etc/nv_tegra_release` shows `REVISION: 4.x` after flashing | The Nano was flashed to a newer JetPack or boot firmware was not downgraded | Reflash with JetPack 6.0 Developer Preview / L4T 36.2 and ensure Force Recovery mode is active. |
| `docker` requires `sudo` | User is not in the Docker group or the session has not refreshed | Run `sudo usermod -aG docker ultraviolet`, then log out and back in. `services/stm-stack.sh` can still fall back to `sudo docker`. |
| `pylonsrc` is missing in preflight | Pylon SDK or GStreamer plugin is absent from the image or mount | Use [../../CPP_CONTAINER_RUNBOOK.md](../../CPP_CONTAINER_RUNBOOK.md) to bake or install Pylon into the C++ image. |
| `nvinfer` missing inside container | Wrong image or DeepStream plugin path | Confirm `CV_STM_IMAGE` is `nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon` and re-run preflight. |
| `colcon build` fails with `undefined reference to nvdla::...` or `libnvdla_compiler.so: file too short` | JetPack 6.x defaults to an empty 0-byte stub file for the NVDLA compiler on Orin Nano (since Nano lacks DLA hardware, though TensorRT still references it) | On the host Jetson terminal (outside Docker), run: `sudo apt-get update && sudo apt-get install -y nvidia-l4t-dla-compiler`. This replaces the 0-byte stub with the real compiler and resolves the link/load errors. |
| CUDA driver/runtime mismatch remains | Host is not L4T 36.2 or the wrong container image is running | Re-check host release and `deepstream-app --version-all` inside `cv-stm-stack`. |

## Maintenance

**Update Triggers:**
- The field container moves away from DeepStream 6.4 / TensorRT 8.6.
- The team standardizes on a newer JetPack/L4T host generation.
- `services/setup-orin-nano.sh`, `services/stm-stack.sh`, or the default image tag changes.
- SDK Manager selections, account flow, storage choice, or Force Recovery hardware steps change.

**Verification:**
- [ ] All links resolve.
- [ ] SDK Manager selections still target L4T 36.2 for the current DeepStream 6.4 image.
- [ ] Post-flash commands match `services/setup-orin-nano.sh` and `services/stm-stack.sh`.
- [ ] No private passwords, tokens, or service account secrets are committed.
- [ ] Runtime validation output matches the current expected container stack.

**Last Updated:** 2026-06-08
