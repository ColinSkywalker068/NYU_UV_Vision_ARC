# yolo_cpp Overview

Related pages: [Pipeline](./pipeline.md) | [Nodes](./nodes.md) | [Interfaces](./interfaces.md) | [Basler SP/ST Bringup](./basler-sp-st-bringup.md) | [Legacy Pipeline Notes](./legacy-pipeline-notes.md) | [Launch and Config](./launch-and-config.md) | [Development Notes](./development-notes.md) | [Documentation Coverage](./documentation-coverage.md)

`yolo_cpp` is the active C++ ROS 2 package for the vision pipeline. It owns the live detector, tracker, processor, solver, and STM bridge path that the team runs today, plus an older combined perception node that still exists for comparison and zero-copy experiments.

The package is built around a straightforward flow:

1. A camera backend provides frames from either RealSense or Basler.
2. The detector publishes armor candidates, corners, and optional PnP pose on `detector/armors`.
3. The tracker turns those detections into stable tracks on `tracker/tracked_armors` and picks a 2D-compatible target on `tracker/target`.
4. The processor transforms and filters the selected 3D target on `processor/target`.
5. The aim solver converts the selected target into `bounding_box`, `bounding_box_angles`, `override`, `suggest_fire`, `obb_angle`, and `aim_solution`.
6. The STM bridge forwards targeting output and listens for attitude, bullet speed, or color hints from the controller.

Use [`basler.launch.py`](../launch/basler.launch.py) when you want the full Basler + DeepStream active stack, including the SP/ST Type-C bridge. [`stm.launch.py`](../launch/stm.launch.py) remains as a compatibility alias. Use [`realsense.launch.py`](../launch/realsense.launch.py) when you want a lighter RealSense + native TensorRT detector, tracker, processor, and solver loop without the serial bridge.

If you need the exact container and build steps, the repo-level runbook is still the best source: [CPP_CONTAINER_RUNBOOK.md](../../CPP_CONTAINER_RUNBOOK.md). If you need the longer-term runtime direction, start with [RM_AUTO_AIM_PARITY_NEXT_STEPS.md](../../docs/RM_AUTO_AIM_PARITY_NEXT_STEPS.md). If you want the older illustrated, step-by-step explanation that predated the modular docs, see [Legacy Pipeline Notes](./legacy-pipeline-notes.md) and the archived original at [legacy/docs/Pipeline/Pipeline.md](../../legacy/docs/Pipeline/Pipeline.md).
