cmake_minimum_required(VERSION 3.16)
project(yolo_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(YOLO_CPP_BUILD_MANUAL_TOOLS "Build manual hardware smoke tools" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
  )
endif()

set(PYLON_SEARCH_ROOTS /opt/pylon)
if(DEFINED ENV{PYLON_ROOT} AND NOT "$ENV{PYLON_ROOT}" STREQUAL "")
  list(PREPEND PYLON_SEARCH_ROOTS "$ENV{PYLON_ROOT}")
endif()
list(REMOVE_DUPLICATES PYLON_SEARCH_ROOTS)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(cv_auto_aim_interfaces REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(
    GSTREAMER QUIET
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-base-1.0
    gstreamer-video-1.0)
  if(GSTREAMER_FOUND)
    set(HAVE_GSTREAMER TRUE)
  endif()
endif()

find_path(LIBREALSENSE2_INCLUDE_DIR librealsense2/rs.hpp)
find_library(LIBREALSENSE2_LIBRARY NAMES realsense2 librealsense2)
if(LIBREALSENSE2_INCLUDE_DIR AND LIBREALSENSE2_LIBRARY)
  set(HAVE_LIBREALSENSE2 TRUE)
endif()

set(TENSORRT_SEARCH_ROOTS
  /usr
  /usr/local
  /usr/src/tensorrt
  /usr/local/tensorrt)
if(DEFINED ENV{TENSORRT_ROOT} AND NOT "$ENV{TENSORRT_ROOT}" STREQUAL "")
  list(PREPEND TENSORRT_SEARCH_ROOTS "$ENV{TENSORRT_ROOT}")
endif()
list(REMOVE_DUPLICATES TENSORRT_SEARCH_ROOTS)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS
    ${TENSORRT_SEARCH_ROOTS}
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/include/aarch64-linux-gnu
    /usr/local/include
    /usr/src/tensorrt/include
  PATH_SUFFIXES include include/x86_64-linux-gnu include/aarch64-linux-gnu)
find_path(CUDART_INCLUDE_DIR cuda_runtime_api.h
  PATHS
    /usr/local/cuda/include
    /usr/local/cuda/targets/x86_64-linux/include
    /usr/local/cuda/targets/aarch64-linux/include
    /usr/include
    /usr/include/x86_64-linux-gnu
    /usr/include/aarch64-linux-gnu)
find_library(TENSORRT_LIBRARY NAMES nvinfer
  PATHS
    ${TENSORRT_SEARCH_ROOTS}
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
    /usr/src/tensorrt/lib
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu)
find_library(CUDART_LIBRARY NAMES cudart
  PATHS
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/x86_64-linux/lib
    /usr/local/cuda/targets/aarch64-linux/lib
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu)
if(TENSORRT_INCLUDE_DIR AND CUDART_INCLUDE_DIR AND TENSORRT_LIBRARY AND CUDART_LIBRARY)
  set(HAVE_TENSORRT TRUE)
  message(STATUS "TensorRT found. Native TensorRT backend enabled.")
  message(STATUS "  TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR}")
  message(STATUS "  TENSORRT_LIBRARY=${TENSORRT_LIBRARY}")
  message(STATUS "  CUDART_INCLUDE_DIR=${CUDART_INCLUDE_DIR}")
  message(STATUS "  CUDART_LIBRARY=${CUDART_LIBRARY}")
else()
  message(WARNING
    "TensorRT native backend disabled. Missing one or more dependencies:\n"
    "  TENSORRT_INCLUDE_DIR=${TENSORRT_INCLUDE_DIR}\n"
    "  TENSORRT_LIBRARY=${TENSORRT_LIBRARY}\n"
    "  CUDART_INCLUDE_DIR=${CUDART_INCLUDE_DIR}\n"
    "  CUDART_LIBRARY=${CUDART_LIBRARY}\n"
    "Rebuild after installing TensorRT/CUDA headers and libraries, or set TENSORRT_ROOT.")
endif()

include(CheckLanguage)
if(HAVE_GSTREAMER AND CUDART_INCLUDE_DIR AND CUDART_LIBRARY)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 87)
      endif()
    endif()
    set(HAVE_YOLO_CUDA_BAYER_RESIZE TRUE)
    message(STATUS "CUDA compiler found. CUDA Bayer resize backend enabled.")
  else()
    message(WARNING "CUDA compiler not found. CUDA Bayer resize backend will fall back to CPU Bayer conversion.")
  endif()
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

set(YOLO_CPP_COMPONENT_SOURCES
  src/utils.cpp
  src/pnp_solver.cpp
  src/tracking_math.cpp
  src/tracker.cpp
  src/camera_backend.cpp
  src/model_metadata.cpp
  src/detection_postprocess.cpp
  src/detector_backend.cpp
  src/deepstream_runtime_helpers.cpp
  src/deepstream_runtime.cpp
  src/vision_node_helpers.cpp
  src/gst_cuda_bayer_resize.cpp
  src/armor_detector_node.cpp
  src/armor_tracker_node.cpp
  src/armor_processor_node.cpp
  src/aim_solver_node.cpp
  src/perception_node.cpp
)
if(HAVE_YOLO_CUDA_BAYER_RESIZE)
  list(APPEND YOLO_CPP_COMPONENT_SOURCES src/cuda_bayer_resize.cu)
endif()

add_library(yolo_cpp_components SHARED
  ${YOLO_CPP_COMPONENT_SOURCES}
)

target_include_directories(yolo_cpp_components PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)
if(HAVE_GSTREAMER)
  target_include_directories(yolo_cpp_components PRIVATE ${GSTREAMER_INCLUDE_DIRS})
  target_link_directories(yolo_cpp_components PRIVATE ${GSTREAMER_LIBRARY_DIRS})
  foreach(GSTREAMER_CFLAG ${GSTREAMER_CFLAGS_OTHER})
    target_compile_options(
      yolo_cpp_components PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:${GSTREAMER_CFLAG}>)
  endforeach()
  target_compile_definitions(yolo_cpp_components PRIVATE HAVE_GSTREAMER=1)
  target_link_libraries(yolo_cpp_components ${GSTREAMER_LIBRARIES})
endif()
if(HAVE_YOLO_CUDA_BAYER_RESIZE)
  target_compile_definitions(yolo_cpp_components PRIVATE HAVE_YOLO_CUDA_BAYER_RESIZE=1)
  target_include_directories(yolo_cpp_components PRIVATE ${CUDART_INCLUDE_DIR})
  target_link_libraries(yolo_cpp_components ${CUDART_LIBRARY})
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
  ament_index_cpp
  rclcpp
  rclcpp_components
  geometry_msgs
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
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
  "yolo_cpp::ArmorProcessorNode"
  "yolo_cpp::AimSolverNode"
  "yolo_cpp::PerceptionNode"
)

add_executable(perception_node src/perception_node_main.cpp)
target_link_libraries(perception_node yolo_cpp_components)
ament_target_dependencies(perception_node rclcpp)

add_executable(armor_detector_node src/armor_detector_node_main.cpp)
target_link_libraries(armor_detector_node yolo_cpp_components)
ament_target_dependencies(
  armor_detector_node
  cv_auto_aim_interfaces
  geometry_msgs
  rclcpp
  sensor_msgs
  std_msgs
)

add_executable(armor_tracker_node src/armor_tracker_node_main.cpp)
target_link_libraries(armor_tracker_node yolo_cpp_components)
ament_target_dependencies(
  armor_tracker_node
  cv_auto_aim_interfaces
  geometry_msgs
  rclcpp
  std_msgs
)

add_executable(armor_processor_node src/armor_processor_node_main.cpp)
target_link_libraries(armor_processor_node yolo_cpp_components)
ament_target_dependencies(
  armor_processor_node
  cv_auto_aim_interfaces
  geometry_msgs
  rclcpp
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  visualization_msgs
)

add_executable(aim_solver_node src/aim_solver_node_main.cpp)
target_link_libraries(aim_solver_node yolo_cpp_components)
ament_target_dependencies(aim_solver_node cv_auto_aim_interfaces rclcpp std_msgs)

add_executable(fps_benchmark
  src/fps_benchmark_support.cpp
  src/fps_benchmark.cpp
)
target_link_libraries(fps_benchmark yolo_cpp_components ${OpenCV_LIBS})
ament_target_dependencies(fps_benchmark rclcpp)

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

if(YOLO_CPP_BUILD_MANUAL_TOOLS)
  add_executable(stm32_yaw_packet_smoke
    tools/stm32_yaw_packet_smoke.cpp
    src/seasky_protocol.cpp
  )
  target_include_directories(stm32_yaw_packet_smoke PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
endif()

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS yolo_cpp_components perception_node
    armor_detector_node armor_tracker_node armor_processor_node aim_solver_node fps_benchmark
    stm32_serial_bridge
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(YOLO_CPP_BUILD_MANUAL_TOOLS)
  install(
    TARGETS stm32_yaw_packet_smoke
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY docs
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_clang_format REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)

  ament_clang_format(CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../.clang-format)

  ament_add_gtest(test_detection_postprocess
    test/test_detection_postprocess.cpp
    src/model_metadata.cpp
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
    src/tracking_math.cpp
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
