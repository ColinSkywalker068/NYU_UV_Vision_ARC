#ifndef YOLOV8_ROS_CPP__DETECTION_POSTPROCESS_HPP_
#define YOLOV8_ROS_CPP__DETECTION_POSTPROCESS_HPP_

#include "yolov8_ros_cpp/detector_backend.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace yolov8_ros_cpp
{

struct EngineMetadata
{
  std::unordered_map<int, std::string> class_names;
  std::optional<std::string> task;
};

EngineMetadata read_engine_metadata_from_file(const std::string & model_path);

std::vector<Detection> parse_yolo_output_tensor(
  const float * tensor_data,
  const std::vector<int64_t> & shape,
  int img_w,
  int img_h,
  int num_classes,
  float conf_threshold,
  bool task_obb,
  const rclcpp::Logger * logger = nullptr);

}  // namespace yolov8_ros_cpp

#endif
