#ifndef YOLO_CPP__DEEPSTREAM_RUNTIME_HPP_
#define YOLO_CPP__DEEPSTREAM_RUNTIME_HPP_

#include "yolo_cpp/detector_backend.hpp"
#include "yolo_cpp/frame_packet.hpp"

#include <rclcpp/logger.hpp>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace yolo_cpp
{

class DeepStreamRuntime
{
public:
  struct Config
  {
    std::string model_path;
    std::string config_template_path;
    std::string task{"obb"};
    int imgsz_width{640};
    int imgsz_height{640};
    int source_width{640};
    int source_height{640};
    int source_fps{60};
    float conf_threshold{0.45F};
    bool half{true};
    bool enable_color_verification{true};
    bool debug_publish_image_raw{false};
    std::string source_element{"pylonsrc"};
    std::string source_id{"basler"};
  };

  struct Result
  {
    DetectionBatch batch;
    DetectorStageTimings timings;
    std::optional<FramePacket> source_packet;
    double capture_ms{0.0};
  };

  using ResultCallback = std::function<void(Result &&)>;

  DeepStreamRuntime(
    rclcpp::Logger logger,
    Config config,
    ResultCallback callback);
  ~DeepStreamRuntime();

  const std::string & backend_name() const;

private:
  struct Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace yolo_cpp

#endif
