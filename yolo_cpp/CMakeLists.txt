cmake_minimum_required(VERSION 3.16)
project(yolo_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(PYLON_SEARCH_ROOTS /opt/pylon)
if(DEFINED ENV{PYLON_ROOT} AND NOT "$ENV{PYLON_ROOT}" STREQUAL "")
  list(PREPEND PYLON_SEARCH_ROOTS "$ENV{PYLON_ROOT}")
endif()
list(REMOVE_DUPLICATES PYLON_SEARCH_ROOTS)

find_package(ament_cmake REQUIRED)
find_package(cv_auto_aim_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(GSTREAMER QUIET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
  if(GSTREAMER_FOUND)
    set(HAVE_GSTREAMER TRUE)
  endif()
endif()

find_path(LIBREALSENSE2_INCLUDE_DIR librealsense2/rs.hpp)
find_library(LIBREALSENSE2_LIBRARY NAMES realsense2 librealsense2)
if(LIBREALSENSE2_INCLUDE_DIR AND LIBREALSENSE2_LIBRARY)
  set(HAVE_LIBREALSENSE2 TRUE)
endif()

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/local/include)
find_path(CUDART_INCLUDE_DIR cuda_runtime_api.h
  PATHS
    /usr/local/cuda/include
    /usr/local/cuda/targets/x86_64-linux/include
    /usr/local/cuda/targets/aarch64-linux/include
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/include/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY NAMES nvinfer PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/lib)
find_library(CUDART_LIBRARY NAMES cudart PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
if(TENSORRT_INCLUDE_DIR AND CUDART_INCLUDE_DIR AND TENSORRT_LIBRARY AND CUDART_LIBRARY)
  set(HAVE_TENSORRT TRUE)
endif()

find_path(DEEPSTREAM_INCLUDE_DIR nvdsmeta.h
  PATHS
    /opt/nvidia/deepstream/deepstream/sources/includes
    /opt/nvidia/deepstream/deepstream/sources/includes/gst-plugins)
find_library(DEEPSTREAM_META_LIBRARY NAMES nvds_meta
  PATHS /opt/nvidia/deepstream/deepstream/lib)
find_library(DEEPSTREAM_GST_META_LIBRARY NAMES nvdsgst_meta
  PATHS /opt/nvidia/deepstream/deepstream/lib)
if(DEEPSTREAM_INCLUDE_DIR AND DEEPSTREAM_META_LIBRARY AND DEEPSTREAM_GST_META_LIBRARY)
  set(HAVE_DEEPSTREAM TRUE)
endif()

find_path(PYLON_INCLUDE_DIR pylon/PylonIncludes.h
  PATHS ${PYLON_SEARCH_ROOTS}
  PATH_SUFFIXES include)
find_library(PYLON_BASE_LIBRARY NAMES pylonbase
  PATHS ${PYLON_SEARCH_ROOTS}
  PATH_SUFFIXES lib lib64)
find_library(PYLON_UTILITY_LIBRARY NAMES pylonutility
  PATHS ${PYLON_SEARCH_ROOTS}
  PATH_SUFFIXES lib lib64)
find_library(PYLON_GENAPI_LIBRARY
  NAMES GenApi_gcc_v3_1_Basler_pylon GenApi_gcc_v3_0_Basler_pylon GenApi_MD_VC141_v3_1_Basler_pylon
  PATHS ${PYLON_SEARCH_ROOTS}
  PATH_SUFFIXES lib lib64)
find_library(PYLON_GCBASE_LIBRARY
  NAMES GCBase_gcc_v3_1_Basler_pylon GCBase_gcc_v3_0_Basler_pylon GCBase_MD_VC141_v3_1_Basler_pylon
  PATHS ${PYLON_SEARCH_ROOTS}
  PATH_SUFFIXES lib lib64)
if(PYLON_INCLUDE_DIR AND PYLON_BASE_LIBRARY AND PYLON_UTILITY_LIBRARY)
  set(HAVE_PYLON TRUE)
  message(STATUS "Basler Pylon SDK found. Basler backend enabled.")
  message(STATUS "  PYLON_INCLUDE_DIR=${PYLON_INCLUDE_DIR}")
  message(STATUS "  PYLON_BASE_LIBRARY=${PYLON_BASE_LIBRARY}")
  message(STATUS "  PYLON_UTILITY_LIBRARY=${PYLON_UTILITY_LIBRARY}")
else()
  message(WARNING
    "Basler Pylon SDK not found. Basler backend will be disabled. "
    "Expected headers/libs under one of: ${PYLON_SEARCH_ROOTS}. "
    "Set PYLON_ROOT or mount/install /opt/pylon before building if you need camera_type=basler.")
endif()

add_library(yolo_cpp_components SHARED
  src/utils.cpp
  src/tracker.cpp
  src/camera_backend.cpp
  src/detection_postprocess.cpp
  src/detector_backend.cpp
  src/deepstream_detector.cpp
  src/deepstream_runtime.cpp
  src/armor_detector_node.cpp
  src/armor_tracker_node.cpp
  src/aim_solver_node.cpp
  src/camera_node.cpp
  src/yolo_node.cpp
  src/perception_node.cpp
)

target_include_directories(yolo_cpp_components PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
if(HAVE_GSTREAMER)
  target_include_directories(yolo_cpp_components PRIVATE ${GSTREAMER_INCLUDE_DIRS})
  target_link_directories(yolo_cpp_components PRIVATE ${GSTREAMER_LIBRARY_DIRS})
  target_compile_options(yolo_cpp_components PRIVATE ${GSTREAMER_CFLAGS_OTHER})
  target_compile_definitions(yolo_cpp_components PRIVATE HAVE_GSTREAMER=1)
  target_link_libraries(yolo_cpp_components ${GSTREAMER_LIBRARIES})
endif()
if(HAVE_DEEPSTREAM)
  target_compile_definitions(yolo_cpp_components PRIVATE HAVE_DEEPSTREAM=1)
  target_include_directories(yolo_cpp_components PRIVATE ${DEEPSTREAM_INCLUDE_DIR})
  target_link_libraries(yolo_cpp_components
    ${DEEPSTREAM_META_LIBRARY}
    ${DEEPSTREAM_GST_META_LIBRARY}
  )
endif()

ament_target_dependencies(yolo_cpp_components
  cv_auto_aim_interfaces
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
)
target_link_libraries(yolo_cpp_components ${OpenCV_LIBS})

if(HAVE_LIBREALSENSE2)
  target_compile_definitions(yolo_cpp_components PRIVATE HAVE_LIBREALSENSE2=1)
  target_include_directories(yolo_cpp_components PRIVATE ${LIBREALSENSE2_INCLUDE_DIR})
  target_link_libraries(yolo_cpp_components ${LIBREALSENSE2_LIBRARY})
endif()

if(HAVE_TENSORRT)
  target_compile_definitions(yolo_cpp_components PRIVATE HAVE_TENSORRT=1)
  target_include_directories(yolo_cpp_components PRIVATE ${TENSORRT_INCLUDE_DIR} ${CUDART_INCLUDE_DIR})
  target_link_libraries(yolo_cpp_components ${TENSORRT_LIBRARY} ${CUDART_LIBRARY})
endif()

if(HAVE_PYLON)
  target_compile_definitions(yolo_cpp_components PRIVATE HAVE_PYLON=1)
  target_include_directories(yolo_cpp_components PRIVATE ${PYLON_INCLUDE_DIR})
  target_link_libraries(yolo_cpp_components
    ${PYLON_BASE_LIBRARY}
    ${PYLON_UTILITY_LIBRARY}
  )
  if(PYLON_GENAPI_LIBRARY)
    target_link_libraries(yolo_cpp_components ${PYLON_GENAPI_LIBRARY})
  endif()
  if(PYLON_GCBASE_LIBRARY)
    target_link_libraries(yolo_cpp_components ${PYLON_GCBASE_LIBRARY})
  endif()
endif()

rclcpp_components_register_nodes(yolo_cpp_components
  "yolo_cpp::ArmorDetectorNode"
  "yolo_cpp::ArmorTrackerNode"
  "yolo_cpp::AimSolverNode"
  "yolo_cpp::CameraNode"
  "yolo_cpp::YoloNode"
  "yolo_cpp::PerceptionNode"
)

add_executable(camera_node src/camera_node_main.cpp)
target_link_libraries(camera_node yolo_cpp_components)
ament_target_dependencies(camera_node rclcpp)

add_executable(yolo_modular_node src/yolo_node_main.cpp)
target_link_libraries(yolo_modular_node yolo_cpp_components)
ament_target_dependencies(yolo_modular_node rclcpp)

add_executable(perception_node src/perception_node_main.cpp)
target_link_libraries(perception_node yolo_cpp_components)
ament_target_dependencies(perception_node rclcpp)

add_executable(armor_detector_node src/armor_detector_node_main.cpp)
target_link_libraries(armor_detector_node yolo_cpp_components)
ament_target_dependencies(armor_detector_node cv_auto_aim_interfaces rclcpp)

add_executable(armor_tracker_node src/armor_tracker_node_main.cpp)
target_link_libraries(armor_tracker_node yolo_cpp_components)
ament_target_dependencies(armor_tracker_node cv_auto_aim_interfaces rclcpp)

add_executable(aim_solver_node src/aim_solver_node_main.cpp)
target_link_libraries(aim_solver_node yolo_cpp_components)
ament_target_dependencies(aim_solver_node cv_auto_aim_interfaces rclcpp)

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
  TARGETS yolo_cpp_components camera_node yolo_modular_node perception_node
    armor_detector_node armor_tracker_node aim_solver_node stm32_serial_bridge
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY docs
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_detection_postprocess
    test/test_detection_postprocess.cpp
    src/detection_postprocess.cpp
  )
  if(TARGET test_detection_postprocess)
    target_include_directories(test_detection_postprocess PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${OpenCV_INCLUDE_DIRS}
    )
    ament_target_dependencies(test_detection_postprocess rclcpp)
    target_link_libraries(test_detection_postprocess ${OpenCV_LIBS})
  endif()
  ament_add_gtest(test_tracking_core
    test/test_tracking_core.cpp
    src/tracker.cpp
    src/utils.cpp
  )
  if(TARGET test_tracking_core)
    target_include_directories(test_tracking_core PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
      ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(test_tracking_core ${OpenCV_LIBS})
  endif()
endif()

ament_package()
