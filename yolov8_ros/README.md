# yolov8_ros

## Running the modular YOLO node and serial bridge

**Use the same code for both `ros2 run` and `python -m`:**

1. Build with **symlink install** so the installed package points at the source tree:
   ```bash
   cd /path/to/CV_Detection
   source /opt/ros/<distro>/setup.bash
   colcon build --packages-select yolov8_ros --symlink-install
   source install/setup.bash
   ```

2. Run the node either way (both use the same `yolov8_ros` package from the workspace):
   ```bash
   # Option A: ros2 run (uses install path, which is symlinked to source)
   ros2 run yolov8_ros yolov8_modular_node --ros-args -p camera_type:=realsense

   # Option B: python -m (must be run after source install/setup.bash so same package is loaded)
   python3 -m yolov8_ros.yolov8_node --camera_type realsense
   ```

If you build **without** `--symlink-install`, `ros2 run` uses copied files under `install/` and can get out of date until you run `colcon build` again. With `--symlink-install`, edits in `yolov8_ros/yolov8_ros/*.py` are used by both `ros2 run` and `python -m` as long as you have run `source install/setup.bash` in that shell.

## Serial bridge

```bash
ros2 run yolov8_ros stm32_serial_bridge --ros-args -p serial_port:=/dev/ttyACM0
# or
python3 -m yolov8_ros.stm32_serial_bridge --serial_port /dev/ttyACM0
```

Same rule: build with `--symlink-install` and source `install/setup.bash` so both invocations use the same code.
