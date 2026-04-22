#ifndef YOLOV8_ROS_CPP__CAMERA_BACKEND_HPP_
#define YOLOV8_ROS_CPP__CAMERA_BACKEND_HPP_

#include "yolov8_ros_cpp/frame_packet.hpp"

#include <rclcpp/logger.hpp>

#include <memory>
#include <string>

namespace rclcpp
{
class Node;
}

namespace yolov8_ros_cpp
{

struct CameraConfig
{
  std::string camera_type{"basler"};
  int imgsz_width{640};
  int imgsz_height{640};
  double basler_exposure_us{5000.0};
  double basler_gain_db{5.0};
  int basler_fps{60};
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

class CameraBackend
{
public:
  virtual ~CameraBackend() = default;
  virtual void initialize() = 0;
  virtual bool capture(FramePacket & packet) = 0;
  virtual void release() = 0;
  virtual double actual_fps() const = 0;
  virtual const std::string & source_id() const = 0;
};

std::shared_ptr<CameraBackend> create_camera_backend(
  const CameraConfig & config, const rclcpp::Logger & logger);

CameraConfig load_camera_config(rclcpp::Node & node);

}  // namespace yolov8_ros_cpp

#endif
