#ifndef YOLO_CPP__DETECTION_POSTPROCESS_HPP_
#define YOLO_CPP__DETECTION_POSTPROCESS_HPP_

#include "yolo_cpp/detector_backend.hpp"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace yolo_cpp
{

struct EngineMetadata
{
  std::unordered_map<int, std::string> class_names;
  std::optional<std::string> task;
  std::optional<std::string> version;
  bool end2end{false};
  bool int8_export{false};
  std::optional<bool> export_includes_nms;
};

EngineMetadata read_engine_metadata_from_file(const std::string & model_path);
std::string read_engine_plan_from_file(const std::string & model_path);

std::vector<Detection> parse_yolo_output_tensor(
  const float * tensor_data,
  const std::vector<int64_t> & shape,
  int img_w,
  int img_h,
  int num_classes,
  float conf_threshold,
  bool task_obb,
  bool end2end,
  std::optional<bool> export_includes_nms = std::nullopt,
  const rclcpp::Logger * logger = nullptr);

}  // namespace yolo_cpp

#endif
