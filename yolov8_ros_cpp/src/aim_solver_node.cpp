#include "yolov8_ros_cpp/aim_solver_node.hpp"

#include "yolov8_ros_cpp/utils.hpp"

#include <rclcpp_components/register_node_macro.hpp>

#include <algorithm>
#include <array>
#include <functional>

namespace yolov8_ros_cpp
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

cv::Rect rect_from_xyxy(const std::array<int32_t, 4> & bbox_xyxy)
{
  const int x1 = bbox_xyxy[0];
  const int y1 = bbox_xyxy[1];
  const int x2 = bbox_xyxy[2];
  const int y2 = bbox_xyxy[3];
  return cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1));
}

}  // namespace

AimSolverNode::AimSolverNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("aim_solver_node", options)
{
  this->declare_parameter<double>("hfov_degrees", 79.2);
  this->declare_parameter<double>("vfov_degrees", 63.78);
  this->declare_parameter<bool>("publish_zero_angles_when_no_target", true);
  this->declare_parameter<int>("default_image_width", default_image_width_);
  this->declare_parameter<int>("default_image_height", default_image_height_);

  hfov_rad_ = static_cast<float>(this->get_parameter("hfov_degrees").as_double() * kPi / 180.0);
  vfov_rad_ = static_cast<float>(this->get_parameter("vfov_degrees").as_double() * kPi / 180.0);
  publish_zero_angles_when_no_target_ =
    this->get_parameter("publish_zero_angles_when_no_target").as_bool();
  default_image_width_ = this->get_parameter("default_image_width").as_int();
  default_image_height_ = this->get_parameter("default_image_height").as_int();

  target_subscription_ = this->create_subscription<cv_auto_aim_interfaces::msg::Target>(
    "tracker/target", 10, std::bind(&AimSolverNode::handle_target, this, std::placeholders::_1));
  bbox_publisher_ =
    this->create_publisher<std_msgs::msg::Int32MultiArray>("bounding_box", 10);
  override_publisher_ = this->create_publisher<std_msgs::msg::Bool>("override", 10);
  angle_publisher_ =
    this->create_publisher<std_msgs::msg::Float32MultiArray>("bounding_box_angles", 10);
  obb_angle_publisher_ =
    this->create_publisher<std_msgs::msg::Float32>("obb_angle", 10);

  RCLCPP_INFO(this->get_logger(), "Initialized AimSolverNode");
}

void AimSolverNode::handle_target(const cv_auto_aim_interfaces::msg::Target::SharedPtr msg)
{
  const int image_width = msg->image_width > 0 ? msg->image_width : default_image_width_;
  const int image_height = msg->image_height > 0 ? msg->image_height : default_image_height_;

  if (!msg->has_target) {
    publish_no_target(image_width, image_height);
    return;
  }

  const auto bbox = rect_from_xyxy(msg->bbox_xyxy);
  bbox_msg_.data = {
    bbox.x,
    bbox.y,
    bbox.x + bbox.width,
    bbox.y + bbox.height};
  override_msg_.data = true;
  bbox_publisher_->publish(bbox_msg_);
  override_publisher_->publish(override_msg_);

  const auto [yaw, pitch] = calculate_angles(
    bbox, image_width, image_height, hfov_rad_, vfov_rad_);
  angle_msg_.data = {yaw, pitch};
  angle_publisher_->publish(angle_msg_);

  obb_angle_msg_.data = msg->has_obb_angle ? msg->obb_angle : 0.0F;
  obb_angle_publisher_->publish(obb_angle_msg_);
}

void AimSolverNode::publish_no_target(int image_width, int image_height)
{
  const int cx = std::max(0, image_width / 2);
  const int cy = std::max(0, image_height / 2);
  bbox_msg_.data = {cx, cy, cx, cy};
  override_msg_.data = false;
  obb_angle_msg_.data = 0.0F;

  bbox_publisher_->publish(bbox_msg_);
  override_publisher_->publish(override_msg_);
  angle_msg_.data = {0.0F, 0.0F};
  angle_publisher_->publish(angle_msg_);
  obb_angle_publisher_->publish(obb_angle_msg_);
}

}  // namespace yolov8_ros_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolov8_ros_cpp::AimSolverNode)
