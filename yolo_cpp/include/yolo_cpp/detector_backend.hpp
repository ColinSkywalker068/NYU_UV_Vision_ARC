#ifndef YOLO_CPP__DETECTOR_BACKEND_HPP_
#define YOLO_CPP__DETECTOR_BACKEND_HPP_

#include "yolo_cpp/frame_packet.hpp"
#include "yolo_cpp/utils.hpp"

#include <rclcpp/logger.hpp>

#include <optional>
#include <string>
#include <unordered_map>
#include <memory>
#include <vector>

namespace yolo_cpp
{

struct Detection
{
  cv::Rect bbox_model;
  float confidence{0.0F};
  int class_id{0};
  std::optional<int> track_id;
  std::optional<float> obb_angle;
};

struct DetectorStageTimings
{
  double preprocess_ms{0.0};
  double upload_ms{0.0};
  double inference_ms{0.0};
  double download_ms{0.0};
  double parse_ms{0.0};
};

struct DetectionBatch
{
  std::vector<Detection> detections;
  LetterboxMetadata letterbox;
};

class DetectorBackend
{
public:
  virtual ~DetectorBackend() = default;
  virtual DetectionBatch infer(
    const FramePacket & packet, const rclcpp::Logger & logger,
    DetectorStageTimings * timings = nullptr) = 0;
  virtual const std::string & backend_name() const = 0;
};

class NativeTensorRTBackend final : public DetectorBackend
{
public:
  NativeTensorRTBackend(
    std::string model_path, std::string task, std::string model_type, bool half,
    float conf_threshold);
  ~NativeTensorRTBackend() override;

  DetectionBatch infer(
    const FramePacket & packet, const rclcpp::Logger & logger,
    DetectorStageTimings * timings = nullptr) override;
  const std::string & backend_name() const override;

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
  bool end2end_{false};
  bool int8_export_{false};
  std::optional<bool> export_includes_nms_;
  std::string engine_version_;
  std::string backend_name_{"tensorrt_stub"};
  std::unordered_map<int, std::string> class_names_;
  std::unique_ptr<Impl> impl_;
};

class DeepStreamBackend final : public DetectorBackend
{
public:
  DeepStreamBackend(
    std::string model_path, std::string task, std::string model_type, bool half,
    float conf_threshold);

  DetectionBatch infer(
    const FramePacket & packet, const rclcpp::Logger & logger,
    DetectorStageTimings * timings = nullptr) override;
  const std::string & backend_name() const override;

private:
  NativeTensorRTBackend fallback_;
  mutable bool warned_{false};
  std::string backend_name_{"deepstream_fallback"};
};

std::unique_ptr<DetectorBackend> create_detector_backend(
  const std::string & backend_kind, const std::string & model_path, const std::string & task,
  const std::string & model_type, bool half, float conf_threshold);

}  // namespace yolo_cpp

#endif
