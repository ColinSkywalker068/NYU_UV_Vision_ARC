# Plan: Migrating `yolov8_ros` from Python to C++

This document outlines a strategic plan for migrating the `yolov8_ros` package from its current Python implementation to a high-performance C++ version.

## 1. Analysis of the Python Implementation

The current system consists of four main components:
- **`yolov8_node.py`**: The main ROS 2 node. It orchestrates the other components, handles ROS parameters, subscribers, publishers, and contains the main processing loop.
- **`camera.py`**: A dedicated class to interface with a Basler camera using the `pypylon` library.
- **`detector.py`**: A wrapper for the `ultralytics` library, which runs the core YOLOv8 inference. It is already using a TensorRT `.engine` file, which is ideal.
- **`tracker.py`**: A custom object tracker using a `cv2.KalmanFilter` to predict object motion.

The primary performance bottlenecks are data copies between components (camera -> OpenCV -> PyTorch/TensorRT), Python's Global Interpreter Lock (GIL) preventing true parallelism, and the overhead of the `rclpy` client library compared to `rclcpp`.

## 2. C++ Project Structure

A new C++ ROS 2 package, `yolov8_ros_cpp`, should be created with the following structure:

```
yolov8_ros_cpp/
├── CMakeLists.txt
├── package.xml
├── launch/
│   └── yolov8_cpp.launch.py
├── include/yolov8_ros_cpp/
│   ├── visibility_control.h
│   ├── detector.hpp
│   ├── tracker.hpp
│   ├── camera.hpp
│   └── yolov8_node.hpp
└── src/
    ├── detector.cpp
    ├── tracker.cpp
    ├── camera.cpp
    ├── yolov8_node.cpp
    └── main.cpp      # Main function to spin the node
```

## 3. Core Dependencies

The `package.xml` and `CMakeLists.txt` must declare the following dependencies:
- **`rclcpp`**: The core C++ ROS 2 client library.
- **`image_transport`**: To efficiently subscribe to image topics.
- **`OpenCV`**: Required for the Kalman Filter (`cv::KalmanFilter`), image pre/post-processing, and drawing visualizations.
- **`TensorRT`**: For loading the `.engine` file and performing inference. The C++ API will be used directly.
- **`Pylon`**: The C++ SDK for the Basler camera. This will replace `pypylon`.
- **`ament_cmake_auto`**: Simplifies writing the `CMakeLists.txt` file.

## 4. Migration Plan: Component by Component

### Step 4.1: The `Camera` Class (`camera.hpp`/`.cpp`)
- **Goal:** Replicate the functionality of `BaslerCamera` in C++.
- **Interface:**
    ```cpp
    class Camera {
    public:
        Camera(int width, int height, double fps, ...);
        ~Camera();
        bool initialize();
        bool get_frame(cv::Mat& frame); // Pass by reference to avoid copies
        void release();
    private:
        // Pylon C++ SDK specific objects
        Pylon::CInstantCamera camera;
        Pylon::CImageFormatConverter converter;
    };
    ```
- **Implementation:** Use the official Pylon C++ SDK examples as a reference. The logic will be very similar to the Python version, but using C++ objects. This class should handle all camera configuration.

### Step 4.2: The `Detector` Class (`detector.hpp`/`.cpp`)
- **Goal:** Load the YOLOv8 TensorRT engine and perform inference directly using the C++ TensorRT API.
- **Interface:**
    ```cpp
    class Detector {
    public:
        Detector(const std::string& engine_path);
        ~Detector();
        // Detections can be a struct you define
        struct Detection {
            cv::Rect bbox;
            float score;
            int class_id;
            int track_id;
        };
        std::vector<Detection> run(const cv::Mat& image);
    private:
        // TensorRT C++ API objects
        nvinfer1::IRuntime* runtime;
        nvinfer1::ICudaEngine* engine;
        nvinfer1::IExecutionContext* context;
        // Buffers for input/output on the GPU
        void* buffers[2]; 
        cudaStream_t stream;
    };
    ```
- **Implementation:** This is the most complex part. You will need to:
    1.  Read the `.engine` file from disk.
    2.  Deserialize it to create a TensorRT engine.
    3.  Create an execution context.
    4.  Allocate memory on the GPU for input and output tensors.
    5.  Implement pre-processing (letterboxing, normalization) and post-processing (NMS, decoding bounding boxes) in C++. This can be done with OpenCV on the CPU or, for maximum performance, with custom CUDA kernels on the GPU.

### Step 4.3: The `Tracker` Class (`tracker.hpp`/`.cpp`)
- **Goal:** Re-implement the Kalman Filter logic from `ArmorPlateTracker`.
- **Interface:**
    ```cpp
    class Tracker {
    public:
        Tracker(float initial_x, float initial_y, float w, float h);
        cv::Point2f predict(float dt);
        void update(const cv::Point2f& measurement);
    private:
        cv::KalmanFilter kf;
    };

    // A wrapper to hold the tracker and its state
    struct Track {
        int id;
        std::unique_ptr<Tracker> tracker;
        int lost_frames = 0;
    };
    ```
- **Implementation:** The logic is a direct translation from Python's `cv2.KalmanFilter` to C++'s `cv::KalmanFilter`. The state and measurement matrices will be identical.

### Step 4.4: The Main `YoloNode` Class (`yolov8_node.hpp`/`.cpp`)
- **Goal:** Tie all components together in a ROS 2 node.
- **Interface:**
    ```cpp
    class YoloNode : public rclcpp::Node {
    public:
        YoloNode();
    private:
        void timer_callback();
        void id_listener_callback(const std_msgs::msg::Int32::SharedPtr msg);

        // Components
        std::unique_ptr<Camera> camera_;
        std::unique_ptr<Detector> detector_;
        std::map<int, Track> active_trackers_;

        // ROS Publishers/Subscribers
        rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr id_sub_;
        rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr bbox_pub_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr angle_pub_;
        rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr override_pub_;
        rclcpp::TimerBase::SharedPtr timer_;

        // Parameters
        // ... (use declare_parameter and get_parameter as in Python)
    };
    ```
- **Implementation:**
    1.  **Parameters:** Declare and retrieve all parameters in the constructor, just like in the Python node.
    2.  **Initialization:** Instantiate the `Camera` and `Detector` classes.
    3.  **`timer_callback`:** This will be the main loop.
        - Call `camera_->get_frame(frame)`.
        - Call `detector_->run(frame)` to get detections.
        - Implement the tracking logic to update `active_trackers_`.
        - Perform Kalman filter prediction and update.
        - Publish results on the ROS topics.

## 5. Build System (`CMakeLists.txt`)

A simplified `CMakeLists.txt` would look like this:

```cmake
cmake_minimum_required(VERSION 3.8)
project(yolov8_ros_cpp)

find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Find TensorRT and Pylon - this may require setting paths manually
find_package(TensorRT REQUIRED) 
find_package(Pylon REQUIRED)

ament_auto_executable(
    yolov8_ros_cpp_node
    src/main.cpp
    src/yolov8_node.cpp
    src/detector.cpp
    src/tracker.cpp
    src/camera.cpp
)

ament_target_dependencies(yolov8_ros_cpp_node
    rclcpp
    OpenCV
    ${TensorRT_LIBRARIES}
    ${Pylon_LIBRARIES}
)

# Include directories
target_include_directories(yolov8_ros_cpp_node PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${Pylon_INCLUDE_DIRS}
)

install(TARGETS
    yolov8_ros_cpp_node
    DESTINATION lib/${PROJECT_NAME}
)
# ... other install rules for launch files and headers
```

## 6. Launch File

Create a Python launch file (`yolov8_cpp.launch.py`) to start the C++ node. This allows you to continue passing parameters from a YAML file or the launch file itself.

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='yolov8_ros_cpp',
            executable='yolov8_ros_cpp_node',
            name='yolov8_node',
            output='screen',
            parameters=[{
                'model_path': '/path/to/your/best.engine',
                'conf': 0.1,
                'enable_tracking': True,
                # ... other parameters
            }]
        )
    ])
```

This plan provides a structured, component-based approach to the migration. The most challenging part will be the `Detector` class, which requires direct interaction with the TensorRT C++ API. It is highly recommended to start with NVIDIA's TensorRT examples (`trtexec` or sample YOLO implementations) as a foundation.
