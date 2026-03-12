[![pipeline status](https://gitlab.com/nyu-robomaster/CV_Detection/badges/main/pipeline.svg)](https://gitlab.com/nyu-robomaster/CV_Detection/-/commits/main)
# CV Detection

This is the NYU Robomaster's code for the Robomasters Robotics Competition.

## Table of Contents

- [Pipeline Documentation](./Documentation/Pipeline/Pipeline.md)
- [Resources to get started](./Documentation/Resources/Resources.md)
- [Setup Pre-commit](./Documentation/Pre-commit.md)
- [YOLOv8 Node Documentation](./docs/yolov8_node/index.md) — nodes, parameters, topics, launch files

---

## Quick Start

To easily run the detection stack, use the `uv` tool wrapper:

1. **Start the Docker container:**
   ```bash
   uv docker
   ```
2. **Build the ROS 2 packages:**
   ```bash
   uv build
   ```
3. **Run the launch files:**
   - To launch with STM configs and Basler camera (standard start):
     ```bash
     uv run yolov8_ros stm.launch.py
     ```
   - To launch with RealSense configs:
     ```bash
     uv run yolov8_ros realsense.launch.py
     ```

---

### Node Documentation

Detailed technical documentation for every node and module in the `yolov8_ros` package, including all ROS parameters, topics, and launch file arguments, is available in **[docs/yolov8_node/](./docs/yolov8_node/index.md)**.


### Update yml files

Incase the current yml files do not setup the conda environment, create a issue and commit a new yml file that works for that OS. Specify what was breaking in the PR.

#### Linux or Mac

```shell
conda env export --no-builds | grep -v "prefix" > environment.yml
```

#### Windows

```shell
conda env export --no-builds | findstr -v "prefix" > environment.yml
```
