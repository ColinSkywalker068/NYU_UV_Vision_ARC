#ifndef YOLO_CPP__DETECTOR_BACKEND_HPP_
#define YOLO_CPP__DETECTOR_BACKEND_HPP_

#include <array>
#include <memory>
#include <optional>
#include <rclcpp/logger.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "yolo_cpp/frame_packet.hpp"
#include "yolo_cpp/utils.hpp"

namespace yolo_cpp
{

struct Detection
{
  /// Bounding box in letterboxed model coordinates.
  cv::Rect bbox_model;
  /// Detector confidence after class/objectness decoding.
  float confidence{0.0F};
  /// Model class id, usually color/team-specific for this project.
  int class_id{0};
  /// Optional model-provided track id from end-to-end exports.
  std::optional<int> track_id;
  /// Optional oriented-box angle in radians.
  std::optional<float> obb_angle;
  /// True when corners_model contains valid oriented-box corners.
  bool has_corners{false};
  /// Oriented armor corners in model coordinates.
  std::array<cv::Point2f, 4> corners_model{};
};

/// Per-stage timing emitted by detector backends for runtime logs.
struct DetectorStageTimings
{
  double preprocess_ms{0.0};
  double upload_ms{0.0};
  double inference_ms{0.0};
  double download_ms{0.0};
  double parse_ms{0.0};
};

/// Detections plus the letterbox transform needed to return to source pixels.
struct DetectionBatch
{
  std::vector<Detection> detections;
  LetterboxMetadata letterbox;
};

/// Common interface for native TensorRT and DeepStream detector implementations.
class DetectorBackend
{
public:
  virtual ~DetectorBackend() = default;

  /// Run inference for one frame packet.
  ///
  /// Args:
  ///   packet: Captured frame and source metadata.
  ///   logger: ROS logger for diagnostics.
  ///   timings: Optional timing sink populated by the backend.
  ///
  /// Returns:
  ///   Detection batch in model coordinates with letterbox metadata.
  virtual DetectionBatch infer(const FramePacket& packet, const rclcpp::Logger& logger,
    DetectorStageTimings* timings = nullptr) = 0;

  /// Prepare resources before the first frame so startup fails on build/load issues.
  virtual void prepare(const rclcpp::Logger& logger) = 0;

  /// Return a stable backend name for logs and debug output.
  virtual const std::string& backend_name() const = 0;
};

/// TensorRT detector backend that loads a serialized YOLO engine directly.
class NativeTensorRTBackend final : public DetectorBackend
{
public:
  /// Configure a native TensorRT backend.
  NativeTensorRTBackend(std::string model_path, std::string task, std::string model_type, bool half,
    float conf_threshold, bool layer_profile_enabled = false,
    std::string layer_profile_path = "", int layer_profile_interval = 1);
  ~NativeTensorRTBackend() override;

  DetectionBatch infer(const FramePacket& packet, const rclcpp::Logger& logger,
    DetectorStageTimings* timings = nullptr) override;
  void prepare(const rclcpp::Logger& logger) override;
  const std::string& backend_name() const override;

private:
  struct Impl;

  void ensure_initialized(const rclcpp::Logger& logger);

  std::string model_path_;
  std::string task_;
  std::string model_type_;
  bool half_{true};
  float conf_threshold_{0.25F};
  bool layer_profile_enabled_{false};
  std::string layer_profile_path_;
  int layer_profile_interval_{1};
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

/// DeepStream-compatible backend facade with a native TensorRT fallback path.
class DeepStreamBackend final : public DetectorBackend
{
public:
  /// Configure a DeepStream-backed detector.
  DeepStreamBackend(std::string model_path, std::string task, std::string model_type, bool half,
    float conf_threshold, bool layer_profile_enabled = false,
    std::string layer_profile_path = "", int layer_profile_interval = 1);

  DetectionBatch infer(const FramePacket& packet, const rclcpp::Logger& logger,
    DetectorStageTimings* timings = nullptr) override;
  void prepare(const rclcpp::Logger& logger) override;
  const std::string& backend_name() const override;

private:
  NativeTensorRTBackend fallback_;
  mutable bool warned_{false};
  std::string backend_name_{"deepstream_fallback"};
};

/// Create a detector backend from a launch/configured backend kind.
std::unique_ptr<DetectorBackend> create_detector_backend(const std::string& backend_kind,
  const std::string& model_path, const std::string& task, const std::string& model_type, bool half,
  float conf_threshold, bool layer_profile_enabled = false,
  const std::string& layer_profile_path = "", int layer_profile_interval = 1);

}  // namespace yolo_cpp

#endif
