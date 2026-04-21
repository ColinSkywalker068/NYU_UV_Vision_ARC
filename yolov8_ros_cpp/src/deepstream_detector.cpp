#include "yolov8_ros_cpp/deepstream_detector.hpp"

#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>

namespace yolov8_ros_cpp
{

DeepStreamDetector::DeepStreamDetector(
  std::string model_path, std::string task, std::string model_type, bool half,
  float conf_threshold)
: model_path_(std::move(model_path)),
  task_(std::move(task)),
  model_type_(std::move(model_type)),
  half_(half),
  conf_threshold_(conf_threshold)
{
}

std::vector<Detection> DeepStreamDetector::infer(const cv::Mat &, const rclcpp::Logger & logger)
{
  if (!warned_unavailable_) {
    RCLCPP_WARN(
      logger,
      "DeepStream/TensorRT inference is scaffolded but not fully integrated in this workspace yet. "
      "The node will preserve ROS interfaces and publish no-target fallbacks until the backend is completed.");
    warned_unavailable_ = true;
  }
  return {};
}

const std::string & DeepStreamDetector::backend_name() const
{
  return backend_name_;
}

}  // namespace yolov8_ros_cpp
