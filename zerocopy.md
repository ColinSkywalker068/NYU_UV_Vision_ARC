# YOLOv8 Zero-Copy Refactoring Plan

This document outlines the plan to refactor the `yolov8_ros` package to use a separate camera node and leverage ROS 2's zero-copy (intra-process) communication for improved performance.

## Stage 1: Create a Standalone Camera Node

**Goal:** Decouple the camera logic from the YOLOv8 node into its own dedicated, reusable node.

1.  **Create `camera_node.py`:**
    *   Create a new file `yolov8_ros/yolov8_ros/camera_node.py`.
    *   This file will host a new `CameraNode` class.
2.  **Migrate Camera Logic:**
    *   Move the `BaslerCamera` and `RealSenseCamera` initialization and frame-grabbing logic from `yolov8_node.py` into `CameraNode`.
    *   The `CameraNode` will take parameters to select the camera type (`basler`/`realsense`) and configure its settings.
3.  **Create Image Publisher:**
    *   The `CameraNode` will have a timer that dictates the frame-grabbing rate.
    *   In the timer's callback, the node will grab a frame from the camera.
    *   It will use `cv_bridge` to convert the OpenCV image into a `sensor_msgs/msg/Image` message.
    *   It will publish this message to a topic, e.g., `/image_raw`.

## Stage 2: Modify the YOLOv8 Node

**Goal:** Adapt the existing `YOLOv8Node` to consume images from the new `CameraNode` via ROS topics.

1.  **Remove Camera Code:**
    *   Delete all camera-specific initialization and control logic from `yolov8_node.py`.
2.  **Create Image Subscriber:**
    *   The main processing loop will no longer be driven by a timer calling a camera SDK. Instead, it will be driven by incoming messages.
    *   Create a subscriber to the `/image_raw` topic.
3.  **Update Callback Logic:**
    *   The subscription callback will receive `sensor_msgs/msg/Image` messages.
    *   It will use `cv_bridge` to convert the message back into an OpenCV image.
    *   The rest of the YOLOv8 processing logic (preprocessing, inference, publishing results) will run on this converted image.

## Stage 3: Implement Composable Nodes for Zero-Copy

**Goal:** Configure both nodes to run in the same process to enable zero-copy data transfer.

1.  **Refactor Nodes for Composability:**
    *   Ensure both `CameraNode` and `YOLOv8Node` are structured in a way that they can be loaded as components. This typically involves ensuring the main logic is within the node class and the `main` function is standard.
2.  **Create a Launch File:**
    *   Create a new launch file, `launch/zerocopy.launch.py`.
    *   This launch file will define a `ComposableNodeContainer`.
    *   It will then load both the `CameraNode` and `YOLOv8Node` as `ComposableNode` components into this container.
    *   This action ensures both nodes start within the same process, automatically enabling zero-copy communication between them.

## Stage 4: Testing and Validation

**Goal:** Verify that the new architecture works correctly and that zero-copy is active.

1.  **Launch the System:** Run the new `zerocopy.launch.py` file.
2.  **Verify Topics:**
    *   Use `ros2 topic list` to see all active topics.
    *   Use `ros2 topic echo /image_raw` to ensure the `CameraNode` is publishing images correctly.
    *   Use `ros2 topic echo /bounding_box` (and other output topics) to ensure the `YOLOv8Node` is processing images and publishing results.
3.  **Verify Zero-Copy:**
    *   Use `ros2 node info /yolov8_modular_node` and `ros2 node info /camera_node` to inspect the publishers and subscribers. The output can indicate if intra-process communication is being used.
