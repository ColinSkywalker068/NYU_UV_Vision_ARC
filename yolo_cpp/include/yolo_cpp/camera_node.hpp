#ifndef YOLO_CPP__CAMERA_NODE_HPP_
#define YOLO_CPP__CAMERA_NODE_HPP_

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "yolo_cpp/camera_backend.hpp"
#include "yolo_cpp/visibility_control.hpp"

namespace yolo_cpp
{

/// Publishes raw frames from the configured camera backend as sensor_msgs/Image.
class YOLO_CPP_PUBLIC CameraNode : public rclcpp::Node
{
public:
  /// Create the camera backend, publisher, and capture timer.
  explicit CameraNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

  /// Release the active camera backend.
  ~CameraNode() override;

private:
  /// Capture and publish one frame.
  void timer_callback();

  /// Throttle warnings when the backend does not provide a frame.
  void publish_no_frame_warning();

  /// Convert an internal frame packet to a ROS image message.
  sensor_msgs::msg::Image::UniquePtr to_image_message(const FramePacket& packet);

  std::shared_ptr<CameraBackend> backend_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  CameraConfig camera_config_;
  std::string camera_type_;
  double fps_{60.0};
  mutable rclcpp::Time last_warning_time_;
};

}  // namespace yolo_cpp

#endif
