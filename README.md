[![pipeline status](https://gitlab.com/nyu-robomaster/CV_Detection/badges/main/pipeline.svg)](https://gitlab.com/nyu-robomaster/CV_Detection/-/commits/main)
# CV Detection

This is the NYU Robomaster's code for the Robomasters Robotics Competition.

## Table of Contents

- [C++ Pipeline Docs](./yolo_cpp/docs/overview.md)
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
   - To launch the Basler + DeepStream + STM32 field stack:
     ```bash
     uv run yolo_cpp basler.launch.py
     ```
   - To launch the RealSense + native TensorRT bring-up stack:
     ```bash
     uv run yolo_cpp realsense.launch.py
     ```
   - `stm.launch.py` remains as a compatibility alias for `basler.launch.py`.

---

### Runtime Documentation

Current runtime roadmap is available in **[docs/RM_AUTO_AIM_PARITY_NEXT_STEPS.md](./docs/RM_AUTO_AIM_PARITY_NEXT_STEPS.md)**.
Package pipeline docs live in **[yolo_cpp/docs/overview.md](./yolo_cpp/docs/overview.md)**.
The older illustrated pipeline writeup now lives in **[legacy/docs/Pipeline/Pipeline.md](./legacy/docs/Pipeline/Pipeline.md)**.


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
