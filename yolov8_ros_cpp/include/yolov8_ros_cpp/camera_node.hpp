#ifndef YOLOV8_ROS_CPP__CAMERA_NODE_HPP_
#define YOLOV8_ROS_CPP__CAMERA_NODE_HPP_

#include "yolov8_ros_cpp/visibility_control.hpp"

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <memory>

namespace yolov8_ros_cpp
{

class CameraBackend;

class YOLOV8_ROS_CPP_PUBLIC CameraNode : public rclcpp::Node
{
public:
  explicit CameraNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~CameraNode() override;

private:
  void timer_callback();
  void publish_no_frame_warning();
  sensor_msgs::msg::Image::UniquePtr to_image_message(const cv::Mat & frame) const;

  std::shared_ptr<CameraBackend> backend_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string camera_type_;
  double fps_{60.0};
  mutable rclcpp::Time last_warning_time_;
};

}  // namespace yolov8_ros_cpp

#endif
