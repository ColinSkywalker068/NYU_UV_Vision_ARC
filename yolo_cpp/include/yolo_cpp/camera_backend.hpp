#ifndef YOLO_CPP__CAMERA_BACKEND_HPP_
#define YOLO_CPP__CAMERA_BACKEND_HPP_

#include "yolo_cpp/frame_packet.hpp"

#include <rclcpp/logger.hpp>

#include <memory>
#include <string>

namespace rclcpp
{
class Node;
}

namespace yolo_cpp
{

struct CameraConfig
{
  /// Backend selector: "basler" or "realsense".
  std::string camera_type{"basler"};
  int imgsz_width{640};
  int imgsz_height{640};
  double basler_exposure_us{5000.0};
  double basler_gain_db{5.0};
  int basler_fps{60};
  bool basler_uncapped_fps{true};
  bool basler_flip{false};
  bool basler_use_full_sensor{true};
  bool basler_centered_roi{true};
  int realsense_width{640};
  int realsense_height{480};
  int realsense_fps{60};
  double realsense_exposure{40.0};
  double realsense_gain{64.0};
  bool realsense_auto_exposure{false};
  bool realsense_auto_white_balance{true};
  double realsense_white_balance{6500.0};
  bool realsense_use_poll{false};
  bool realsense_use_max_native_fps{false};
};

/// Common camera capture interface used by detector and perception nodes.
class CameraBackend
{
public:
  virtual ~CameraBackend() = default;

  /// Open the camera and configure hardware parameters.
  virtual void initialize() = 0;

  /// Capture a frame into packet.
  ///
  /// Returns:
  ///   True when packet contains a usable frame.
  virtual bool capture(FramePacket & packet) = 0;

  /// Release device resources.
  virtual void release() = 0;

  /// Return measured or configured source FPS.
  virtual double actual_fps() const = 0;

  /// Return a stable source id such as "basler" or "realsense".
  virtual const std::string & source_id() const = 0;
};

/// Create the configured camera backend.
std::shared_ptr<CameraBackend> create_camera_backend(
  const CameraConfig & config, const rclcpp::Logger & logger);

/// Load camera-related ROS parameters from a node into CameraConfig.
CameraConfig load_camera_config(rclcpp::Node & node);

}  // namespace yolo_cpp

#endif
