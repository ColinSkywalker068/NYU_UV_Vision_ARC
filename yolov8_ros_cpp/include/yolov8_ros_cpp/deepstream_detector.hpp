#ifndef YOLOV8_ROS_CPP__DEEPSTREAM_DETECTOR_HPP_
#define YOLOV8_ROS_CPP__DEEPSTREAM_DETECTOR_HPP_

#include <opencv2/core.hpp>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rclcpp
{
class Logger;
}

namespace yolov8_ros_cpp
{

struct Detection
{
  cv::Rect bbox_model;
  float confidence{0.0F};
  int class_id{0};
  std::optional<int> track_id;
  std::optional<float> obb_angle;
};

class DeepStreamDetector
{
public:
  DeepStreamDetector(
    std::string model_path, std::string task, std::string model_type, bool half,
    float conf_threshold);

  std::vector<Detection> infer(const cv::Mat & frame, const rclcpp::Logger & logger);
  const std::string & backend_name() const;

private:
  struct Impl;
  void ensure_initialized(const rclcpp::Logger & logger);

  std::string model_path_;
  std::string task_;
  std::string model_type_;
  bool half_{true};
  float conf_threshold_{0.25F};
  bool warned_unavailable_{false};
  bool initialized_{false};
  std::string backend_name_{"tensorrt_stub"};
  std::unordered_map<int, std::string> class_names_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace yolov8_ros_cpp

#endif
