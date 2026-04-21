cmake_minimum_required(VERSION 3.16)
project(yolov8_ros_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PkgConfig QUIET)

find_path(LIBREALSENSE2_INCLUDE_DIR librealsense2/rs.hpp)
find_library(LIBREALSENSE2_LIBRARY NAMES realsense2 librealsense2)
if(LIBREALSENSE2_INCLUDE_DIR AND LIBREALSENSE2_LIBRARY)
  set(HAVE_LIBREALSENSE2 TRUE)
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/local/include)
find_library(TENSORRT_LIBRARY NAMES nvinfer PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/lib)
find_library(CUDART_LIBRARY NAMES cudart PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND CUDART_LIBRARY)
  set(HAVE_TENSORRT TRUE)
endif()

find_path(PYLON_INCLUDE_DIR pylon/PylonIncludes.h PATHS /opt/pylon/include)
find_library(PYLON_BASE_LIBRARY NAMES pylonbase PATHS /opt/pylon/lib /opt/pylon/lib64)
find_library(PYLON_UTILITY_LIBRARY NAMES pylonutility PATHS /opt/pylon/lib /opt/pylon/lib64)
find_library(PYLON_GENAPI_LIBRARY
  NAMES GenApi_gcc_v3_1_Basler_pylon GenApi_gcc_v3_0_Basler_pylon GenApi_MD_VC141_v3_1_Basler_pylon
  PATHS /opt/pylon/lib /opt/pylon/lib64)
find_library(PYLON_GCBASE_LIBRARY
  NAMES GCBase_gcc_v3_1_Basler_pylon GCBase_gcc_v3_0_Basler_pylon GCBase_MD_VC141_v3_1_Basler_pylon
  PATHS /opt/pylon/lib /opt/pylon/lib64)
if(PYLON_INCLUDE_DIR AND PYLON_BASE_LIBRARY AND PYLON_UTILITY_LIBRARY)
  set(HAVE_PYLON TRUE)
endif()

add_library(yolov8_ros_cpp_components SHARED
  src/utils.cpp
  src/tracker.cpp
  src/deepstream_detector.cpp
  src/camera_node.cpp
  src/yolov8_node.cpp
)

target_include_directories(yolov8_ros_cpp_components PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(yolov8_ros_cpp_components
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
)
target_link_libraries(yolov8_ros_cpp_components ${OpenCV_LIBS})

if(HAVE_LIBREALSENSE2)
  target_compile_definitions(yolov8_ros_cpp_components PRIVATE HAVE_LIBREALSENSE2=1)
  target_include_directories(yolov8_ros_cpp_components PRIVATE ${LIBREALSENSE2_INCLUDE_DIR})
  target_link_libraries(yolov8_ros_cpp_components ${LIBREALSENSE2_LIBRARY})
endif()

if(HAVE_TENSORRT)
  target_compile_definitions(yolov8_ros_cpp_components PRIVATE HAVE_TENSORRT=1)
  target_include_directories(yolov8_ros_cpp_components PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(yolov8_ros_cpp_components ${TENSORRT_LIBRARY} ${CUDART_LIBRARY})
endif()

if(HAVE_PYLON)
  target_compile_definitions(yolov8_ros_cpp_components PRIVATE HAVE_PYLON=1)
  target_include_directories(yolov8_ros_cpp_components PRIVATE ${PYLON_INCLUDE_DIR})
  target_link_libraries(yolov8_ros_cpp_components
    ${PYLON_BASE_LIBRARY}
    ${PYLON_UTILITY_LIBRARY}
  )
  if(PYLON_GENAPI_LIBRARY)
    target_link_libraries(yolov8_ros_cpp_components ${PYLON_GENAPI_LIBRARY})
  endif()
  if(PYLON_GCBASE_LIBRARY)
    target_link_libraries(yolov8_ros_cpp_components ${PYLON_GCBASE_LIBRARY})
  endif()
endif()

rclcpp_components_register_nodes(yolov8_ros_cpp_components
  "yolov8_ros_cpp::CameraNode"
  "yolov8_ros_cpp::YOLOv8Node"
)

add_executable(camera_node src/camera_node_main.cpp)
target_link_libraries(camera_node yolov8_ros_cpp_components)
ament_target_dependencies(camera_node rclcpp)

add_executable(yolov8_modular_node src/yolov8_node_main.cpp)
target_link_libraries(yolov8_modular_node yolov8_ros_cpp_components)
ament_target_dependencies(yolov8_modular_node rclcpp)

add_executable(stm32_serial_bridge
  src/seasky_protocol.cpp
  src/stm32_serial_bridge_node.cpp
  src/stm32_serial_bridge_main.cpp
)
target_include_directories(stm32_serial_bridge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(stm32_serial_bridge rclcpp std_msgs)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS yolov8_ros_cpp_components camera_node yolov8_modular_node stm32_serial_bridge
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
