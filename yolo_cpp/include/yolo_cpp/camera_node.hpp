#ifndef YOLO_CPP__CAMERA_NODE_HPP_
#define YOLO_CPP__CAMERA_NODE_HPP_

#include "yolo_cpp/camera_backend.hpp"
#include "yolo_cpp/visibility_control.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <memory>

namespace yolo_cpp
{

class YOLO_CPP_PUBLIC CameraNode : public rclcpp::Node
{
public:
  explicit CameraNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~CameraNode() override;

private:
  void timer_callback();
  void publish_no_frame_warning();
  sensor_msgs::msg::Image::UniquePtr to_image_message(const FramePacket & packet);

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
