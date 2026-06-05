# Infrastructure: CV Detection

<!-- SCOPE: Server inventory, network and DNS configuration, port allocation, deployed services, artifact management, CI/CD pipeline, and host requirements only. -->
<!-- DOC_KIND: reference -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read when you need deployment topology, Docker service inventory, host device requirements, CI/CD facts, or artifact image names. -->
<!-- SKIP_WHEN: Skip when you need operator procedures, ROS 2 architecture, node behavior, or feature-level design. -->
<!-- PRIMARY_SOURCES: Docker/docker-compose.yml, Docker/Dockerfile.run-cv, Docker/Dockerfile.cplusplus, services/start-container.sh, services/stm-stack.service, services/stm-stack.sh, services/stm-stack-entrypoint.sh, .github/workflows/main.yml, .gitlab-ci.yml -->

> **Status:** Current repository inventory
> **Last Updated:** 2026-05-28

## Quick Navigation

- [Operations Runbook](runbook.md)
- [Jetson VNC Access](jetson-vnc.md)
- [Setup Guide](../Setup/Setup.md)
- [Software Install Guide](../Install_Software/Install_Software.md)
- [Jetson Connection Guide](../Connecting_Jetson/Connecting_Jetson.md)
- [Pre-commit Guide](../Pre-commit.md)

## Agent Entry

| Signal | Value |
|--------|-------|
| Purpose | Canonical inventory for CV Detection deployment hosts, Docker services, device mounts, CI/CD, and artifact images. |
| Read When | You need to know what runs where, which host devices are required, which images are used, or which automation exists. |
| Skip When | You need command procedures, troubleshooting steps, ROS 2 node architecture, or detector/tracker implementation details. |
| Canonical | Yes |
| Next Docs | [Operations Runbook](runbook.md), [Setup Guide](../Setup/Setup.md) |
| Primary Sources | `Docker/docker-compose.yml`, `Docker/Dockerfile.run-cv`, `Docker/Dockerfile.cplusplus`, `services/start-container.sh`, `services/stm-stack.service`, `services/stm-stack.sh`, `services/stm-stack-entrypoint.sh`, `.github/workflows/main.yml`, `.gitlab-ci.yml` |

## 1. Server Inventory

| Role | Known Path | Service Entrypoint | Host Facts Declared In Repo | Notes |
|------|------------|--------------------|-----------------------------|-------|
| Robot compute host | `/home/ultraviolet/CV_Detection` | `services/stm-stack.service` | Ubuntu or Jetson-class Linux host with Docker, NVIDIA runtime, host networking, camera/USB/device access, and optional `/opt/pylon` Basler SDK | Exact hostname and IP address are deployment-specific and are not declared in the repository. |

## 2. Domain And DNS

No domain, DNS, reverse proxy, TLS, or public ingress configuration is declared in the current repository. The active Docker paths use `network_mode: host` or `docker run --network host` for local robot networking.

## 3. Port Allocation

| Source | Service | Network Mode | Published Ports | Notes |
|--------|---------|--------------|-----------------|-------|
| `Docker/docker-compose.yml` | `cv-container` | Host network | None declared with `ports:` | ROS 2/DDS, camera, display, D-Bus, Avahi, and serial access are exposed through host networking and mounted devices rather than explicit container port mappings. |
| `services/start-container.sh` | `uv-*` Docker containers | Host network | None declared in the script | Container names default to `uv-` plus a sanitized image name unless `UV_CONTAINER_NAME` is set. |

## 4. Deployed Services

| Service | Runtime Path | Image Or Build Source | GPU | Devices And Host Access | Startup Behavior |
|---------|--------------|-----------------------|-----|-------------------------|------------------|
| `cv-container` | `Docker/docker-compose.yml` | Builds `Docker/Dockerfile.run-cv` with `BASE_IMAGE=${CV_BASE_IMAGE:-nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon}` | `runtime: nvidia` | `/dev/video0`, `/dev/bus/usb`, `/tmp/argus_socket`, `/dev`, X11 socket, D-Bus, Avahi, Docker socket, Jetson release files, `/home/ultraviolet` | `Docker/Dockerfile.run-cv` sources `install/setup.bash` if present, runs `uv clean`, runs `uv build`, then launches `uv run yolo_cpp realsense.launch.py show:=false`. |
| `stm-stack.service` | `services/stm-stack.service` | Runs `nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon` through `services/start-container.sh` and `services/stm-stack-entrypoint.sh` | Uses `--runtime nvidia` on Jetson hosts | Host network, privileged mode, `/tmp/argus_socket`, `/dev`, `/dev/bus/usb`, D-Bus, Avahi, Docker socket, `/home`, optional `/opt/pylon` | Removes the previous `cv-stm-stack` container, validates the prebuilt workspace/model/profile prerequisites, starts headless `basler.launch.py` by default, can run `realsense.launch.py` with `CV_STM_LAUNCH_PROFILE=realsense`, and restarts every 5 seconds on failure. |

## 5. Artifact Repository

| Artifact | Declared Reference | Source |
|----------|--------------------|--------|
| C++ CV Docker image | `nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon` | `services/uv.sh`, `services/run-cplusplus.sh`, `Docker/docker-compose.yml`, `Docker/Dockerfile.run-cv` |
| Python legacy RealSense wrapper image | `nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-realsense-wrapper-ultra` | `services/uv.sh`, `Docker/Dockerfile.cplusplus`, `Docker/build_cplusplus.sh` |
| Systemd STM stack image | `nyuultraviolet/cv-dockerfile:jp60-pytorch21-humble-tensorrt-cplusplus-pylon` | `services/stm-stack.service`, `services/stm-stack.sh`, `services/setup-orin-nano.sh` |
| Navigation ARM64 image | `nyuultraviolet/cv-dockerfile:nav-arm64` | `services/uv.sh`, `services/setup-orin-nano.sh` |

No credentials, private registry URL, or artifact retention policy is declared in the repository.

## 6. CI/CD Pipeline

| Platform | Config | Trigger | Runner Or Image | Behavior |
|----------|--------|---------|-----------------|----------|
| GitHub Actions | `.github/workflows/main.yml` | Push to `main` | `ubuntu-latest` | Checks out the repo, bumps a git tag with `mathieudutour/github-tag-action@v6.0`, and creates a GitHub release with `ncipollo/release-action@v1`. |
| GitLab CI | `.gitlab-ci.yml` | GitLab pipeline trigger for the current branch | `mohanlu/cv_detection_ci:latest` | Runs `pre-commit run --all-files`, then commits and pushes formatter changes back to the branch when the working tree changes. |

No CI deploy job for the robot host is declared in the repository.

## 7. Host Requirements

| Requirement | Repository Evidence | Notes |
|-------------|---------------------|-------|
| Operating system | `services/start-container.sh`, `services/setup-orin-nano.sh` | Jetson Linux is detected through `/proc/device-tree/model`; generic Linux and macOS container paths also exist for development. |
| Docker | `services/setup-orin-nano.sh`, `Docker/docker-compose.yml`, `services/start-container.sh` | Host needs Docker Engine. The setup script installs the Docker Compose plugin on Ubuntu. |
| NVIDIA container runtime | `Docker/docker-compose.yml`, `services/start-container.sh` | Production robot paths use `runtime: nvidia` or `docker run --runtime nvidia`. |
| GPU stack | `Docker/Dockerfile.cplusplus` | C++ DeepStream image installs DeepStream 6.4, CUDA-adjacent runtime libraries, TensorRT-dependent detector support, and GStreamer plugins. |
| Camera and USB devices | `Docker/docker-compose.yml`, `services/start-container.sh` | Compose mounts `/dev/video0`, `/dev/bus/usb`, `/tmp/argus_socket`, and `/dev`; the script discovers `/dev/video0` through `/dev/video9` on Jetson. |
| STM32 serial | `yolo_cpp/launch/basler.launch.py`, `yolo_cpp/src/stm32_serial_bridge_node.cpp` | Basler field stack defaults to `/dev/ttyUSB0` at `115200` baud; the RealSense service profile does not require serial by default. |
| Basler Pylon SDK | `Docker/Dockerfile.cplusplus`, `Docker/build_cplusplus.sh`, `Docker/bake_cplusplus_pylon_image.sh`, `services/start-container.sh` | Basler service support expects `pylonsrc`; the default runtime image is the pushed Pylon-enabled C++ image, while local validation can still use `cv-detection-cplusplus-pylon:local`, a mounted `/opt/pylon`, or a build with `PYLON_PACKAGE` and `PYLON_GSTREAMER_PACKAGE` under `Docker/pylon/`. |
| Display access | `Docker/docker-compose.yml`, `services/start-container.sh` | GUI/debug display paths use `DISPLAY`, `/tmp/.X11-unix`, `/tmp/.docker.xauth`, and `XAUTHORITY`. |
| Repository location | `services/stm-stack.service`, `Docker/Dockerfile.run-cv` | Runtime commands assume `/home/ultraviolet/CV_Detection`. |

## Maintenance

**Update Triggers:**
- Docker Compose service, image, device, network, or volume changes
- `services/start-container.sh` runtime flag changes
- `services/stm-stack.service` or setup script changes
- New CI/CD workflow files or release automation changes
- Host device, GPU, camera, serial, or Basler SDK requirements changing

**Verification:**
```bash
docker ps
docker compose -f Docker/docker-compose.yml config
systemctl status stm-stack.service
```

---
