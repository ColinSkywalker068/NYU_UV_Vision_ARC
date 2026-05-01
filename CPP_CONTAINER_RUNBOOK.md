# C++ camera + inference in container

## What works today

- **C++ `camera_node`**: camera capture and `image_raw` publishing.
- **C++ detector / tracker / solver**: active runtime path lives in `yolo_cpp`.

## 1) Start container with GUI + devices

Run on host:

```bash
xhost +local:root
./services/run-cplusplus.sh
```

Optional cleanup on host after container exits:

```bash
xhost -local:root
```

## 1.5) Basler / Pylon requirement

The step that usually goes wrong is **not** the ROS build itself. The issue is that Basler support is only
enabled when the Pylon SDK is already visible at `colcon build` time. Right now there are two supported ways
to make that happen:

1. Install Pylon into the image during `docker build` using a local Basler package, usually `tar.gz`.
2. Mount or install Pylon into `/opt/pylon` inside the running container **before** running `colcon build`.

If neither of those happened, `colcon build` still succeeds, but `yolo_cpp` prints a warning during CMake
configure and compiles with the Basler backend disabled.

### Option A: install a local `tar.gz` into the image

Put the Basler package somewhere under the Docker build context, for example:

```bash
mkdir -p Docker/pylon
cp /path/to/your/pylon_*.tar.gz Docker/pylon/
```

Then build the image:

```bash
cd Docker
./build_cplusplus.sh nyuultraviolet/cv-dockerfile \
  jp60-pytorch21-humble-tensorrt-realsense-wrapper-ultra \
  jp60-pytorch21-humble-tensorrt-cplusplus \
  pylon/your-package-name.tar.gz
```

You can also pass the same path via environment variable:

```bash
cd Docker
PYLON_PACKAGE=pylon/your-package-name.tar.gz ./build_cplusplus.sh
```

### Option B: install the `tar.gz` manually inside a running container

Start the container, make the `.deb` visible inside it, then install it before `colcon build`.

If the file is already on the host, one easy path is to copy it into the repo first so the container can see it:

```bash
mkdir -p Docker/pylon
cp /path/to/your/pylon_*.tar.gz Docker/pylon/
./services/run-cplusplus.sh
```

Inside the container:

```bash
mkdir -p /tmp/pylon
tar -xzf /home/ultraviolet/CV_Detection/Docker/pylon/your-package-name.tar.gz -C /tmp/pylon --strip-components=1
find /tmp/pylon -maxdepth 2 -name 'pylon*.sh'
chmod +x "$(find /tmp/pylon -maxdepth 2 -name 'pylon*.sh' | head -n 1)"
"$(find /tmp/pylon -maxdepth 2 -name 'pylon*.sh' | head -n 1)" --accept-eula --default
ldconfig
ls /opt/pylon
```

If the package installs somewhere other than `/opt/pylon`, export the root explicitly before building:

```bash
export PYLON_ROOT=/opt/pylon
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYLON_ROOT/lib
```

## 2) Build packages inside container

**C++ only:**

```bash
cd /home/ultraviolet/CV_Detection
source /opt/ros/humble/setup.bash
colcon build --packages-select yolo_cpp --symlink-install
```

Important: Basler support is only compiled if the Pylon SDK is visible inside the container at build time,
typically at `/opt/pylon`. `services/run-cplusplus.sh` now auto-mounts a host `/opt/pylon` into the container
when that directory exists on the host. If Pylon is missing during `colcon build`, the build succeeds but the
Basler backend is disabled and `camera_type:=basler` will throw at runtime.

Verification signal:

- Good: CMake prints `Basler Pylon SDK found. Basler backend enabled.`
- Bad: CMake prints `Basler Pylon SDK not found. Basler backend will be disabled.`

After installing Pylon, do a clean rebuild so CMake re-detects it:

```bash
cd /home/ultraviolet/CV_Detection
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --packages-select yolo_cpp --symlink-install
```

## 3) Source ROS and workspace overlay

Run in every new shell:

```bash
source /opt/ros/humble/setup.bash
source /home/ultraviolet/CV_Detection/install/setup.bash
```

Quick package check:

```bash
ros2 pkg list | rg yolo_cpp
```

## 4) C++ runtime launch

```bash
ros2 launch yolo_cpp stm.launch.py \
  detector_backend:=deepstream \
  camera_type:=basler
```

This starts the active C++ runtime path.

## 5) Optional runtime checks (new terminal in same container)

```bash
source /opt/ros/humble/setup.bash
source /home/ultraviolet/CV_Detection/install/setup.bash
ros2 node list
ros2 topic list
ros2 topic hz /image_raw
```

## Notes

- If `Package 'yolo_cpp' not found`, re-run the source commands in section 3.
