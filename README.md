[![pipeline status](https://gitlab.com/nyu-robomaster/CV_Detection/badges/main/pipeline.svg)](https://gitlab.com/nyu-robomaster/CV_Detection/-/commits/main)
# CV Detection

This is the NYU Robomaster's code for the Robomasters Robotics Competition.

## Table of Contents

- [Pipeline Documentation](./Documentation/Pipeline/Pipeline.md)
- [Resources to get started](./Documentation/Resources/Resources.md)
- [Setup Pre-commit](./Documentation/Pre-commit.md)
- [rm_auto_aim Parity Next Steps](./docs/RM_AUTO_AIM_PARITY_NEXT_STEPS.md)

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
     uv run yolov8_ros_cpp stm.launch.py
     ```
   - To launch with RealSense configs:
     ```bash
     uv run yolov8_ros_cpp realsense.launch.py
     ```

---

### Runtime Documentation

Current runtime roadmap is available in **[docs/RM_AUTO_AIM_PARITY_NEXT_STEPS.md](./docs/RM_AUTO_AIM_PARITY_NEXT_STEPS.md)**.


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
