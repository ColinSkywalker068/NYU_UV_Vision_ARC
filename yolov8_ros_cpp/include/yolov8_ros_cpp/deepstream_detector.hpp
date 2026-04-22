#ifndef YOLOV8_ROS_CPP__DEEPSTREAM_DETECTOR_HPP_
#define YOLOV8_ROS_CPP__DEEPSTREAM_DETECTOR_HPP_

#include "yolov8_ros_cpp/detector_backend.hpp"
#include "yolov8_ros_cpp/frame_packet.hpp"

#include <opencv2/core.hpp>

#include <memory>
#include <string>

namespace rclcpp
{
class Logger;
}

namespace yolov8_ros_cpp
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

}  // namespace yolov8_ros_cpp

#endif
