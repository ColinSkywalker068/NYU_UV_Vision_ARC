# Testing Plan for Zero-Copy Refactoring

This document provides the steps to build, run, and test the new decoupled, zero-copy architecture for the `yolov8_ros` package.

## 1. Prerequisites

*   A workstation with a sourced ROS 2 Humble environment.
*   The project cloned at `~/CV_Detection`.
*   A compatible camera (e.g., RealSense or Basler) connected to the workstation.
*   All required system dependencies for the camera (e.g., `librealsense-dkms`) are installed.

## 2. Build the Workspace

Before running, you must build the workspace to register the new nodes and launch file.

1.  **Navigate to the workspace root:**
    ```bash
    cd ~/CV_Detection
    ```

2.  **Build with colcon:**
    ```bash
    colcon build --symlink-install
    ```
    *Using `--symlink-install` is recommended as it allows you to change Python files without needing to rebuild.*

3.  **Source the workspace:**
    *Every new terminal you use for testing must be sourced.*
    ```bash
    source install/setup.bash
    ```

## 3. Test Procedure: Zero-Copy with Composable Nodes

This is the primary test to verify that the nodes run together in a single process, enabling zero-copy data transfer.

1.  **Run the Launch File:**
    Execute the `zerocopy.launch.py` launch file. You will likely need to pass the `camera_type` as a parameter.

    *For a RealSense camera:*
    ```bash
    ros2 launch yolov8_ros zerocopy.launch.py camera_type:=realsense
    ```

    *For a Basler camera:*
    ```bash
    ros2 launch yolov8_ros zerocopy.launch.py camera_type:=basler
    ```

2.  **Verification Steps:**
    *   **Verify Nodes are Running (in a new, sourced terminal):**
        ```bash
        ros2 node list
        ```
        You should see the container and the nodes running within it. The output will look something like this:
        ```
        /yolov8_container
        /camera_node
        /yolov8_modular_node
        ```

    *   **Verify Topics:**
        ```bash
        ros2 topic list
        ```
        Ensure the key topics are present:
        - `/image_raw`
        - `/bounding_box`
        - `/bounding_box_angles`

    *   **Check Camera Output:**
        ```bash
        ros2 topic echo /image_raw
        ```
        You should see a stream of `sensor_msgs/msg/Image` data. If there is no output, the `camera_node` may have failed to initialize.

    *   **Check YOLOv8 Output:**
        ```bash
        ros2 topic echo /bounding_box
        ```
        You should see a stream of `std_msgs/msg/Int32MultiArray` data. If there is no output, the `yolov8_node` is likely not receiving images or failing during processing.

    *   **Confirm Zero-Copy is Active:**
        This is the most important verification step for this task.
        ```bash
        ros2 topic info /image_raw
        ```
        The output **must** show that there is one intra-process subscription:
        ```
        Type: sensor_msgs/msg/Image
        Publisher count: 1
        Subscription count: 1
        Intra-process Subscription count: 1  <-- KEY INDICATOR
        ```

## 4. (Optional) Test Procedure: Separate Processes

This test demonstrates the "old" way of running things, where data is copied between processes. It is useful for debugging and for appreciating the performance benefits of the zero-copy approach.

1.  **Open two new, sourced terminals.**

2.  **In Terminal 1, run the `camera_node`:**
    ```bash
    ros2 run yolov8_ros camera_node --ros-args -p camera_type:=realsense
    ```

3.  **In Terminal 2, run the `yolov8_node`:**
    ```bash
    ros2 run yolov8_ros yolov8_modular_node
    ```

4.  **Verification:**
    *   The system should still work, and you should be able to see data on all topics.
    *   However, if you run the zero-copy confirmation command again:
        ```bash
        ros2 topic info /image_raw
        ```
        The output will now show **zero** intra-process subscriptions, confirming that data is being serialized and copied between the two processes.
        ```
        Type: sensor_msgs/msg/Image
        Publisher count: 1
        Subscription count: 1
        Intra-process Subscription count: 0  <-- NO ZERO-COPY
        ```
