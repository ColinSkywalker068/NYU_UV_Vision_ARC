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

class DeepStreamDetector
{
public:
  DeepStreamDetector(
    std::string model_path, std::string task, std::string model_type, bool half,
    float conf_threshold);

  std::vector<Detection> infer(const cv::Mat & frame, const rclcpp::Logger & logger);
  const std::string & backend_name() const;

private:
  std::unique_ptr<NativeTensorRTBackend> backend_;
};

}  // namespace yolo_cpp

#endif
