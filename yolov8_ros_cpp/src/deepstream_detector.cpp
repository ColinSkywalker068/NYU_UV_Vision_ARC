#include "yolov8_ros_cpp/deepstream_detector.hpp"

namespace yolov8_ros_cpp
{

DeepStreamDetector::DeepStreamDetector(
  std::string model_path, std::string task, std::string model_type, bool half,
  float conf_threshold)
: backend_(std::make_unique<NativeTensorRTBackend>(
    std::move(model_path), std::move(task), std::move(model_type), half, conf_threshold))
{
}

std::vector<Detection> DeepStreamDetector::infer(
  const cv::Mat & frame, const rclcpp::Logger & logger)
{
  FramePacket packet{};
  packet.frame = frame;
  packet.pixel_format = PixelFormat::kBgr8;
  packet.storage_kind = BackingStoreKind::kHost;
  return backend_->infer(packet, logger, nullptr).detections;
}

const std::string & DeepStreamDetector::backend_name() const
{
  return backend_->backend_name();
}

}  // namespace yolov8_ros_cpp
