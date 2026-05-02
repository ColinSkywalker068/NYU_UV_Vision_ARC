#ifndef YOLO_CPP__DEEPSTREAM_DETECTOR_HPP_
#define YOLO_CPP__DEEPSTREAM_DETECTOR_HPP_

#include "yolo_cpp/detector_backend.hpp"
#include "yolo_cpp/frame_packet.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace rclcpp
{
class Logger;
}

namespace yolo_cpp
{

/// Compatibility wrapper used by the legacy image-subscription YOLO node.
class DeepStreamDetector
{
public:
  /// Construct a detector that matches the older DeepStreamDetector API.
  DeepStreamDetector(
    std::string model_path, std::string task, std::string model_type, bool half,
    float conf_threshold);

  /// Run inference on an OpenCV BGR frame.
  std::vector<Detection> infer(const cv::Mat & frame, const rclcpp::Logger & logger);

  /// Return the active backend name used for logging.
  const std::string & backend_name() const;

private:
  std::unique_ptr<NativeTensorRTBackend> backend_;
};

}  // namespace yolo_cpp

#endif
