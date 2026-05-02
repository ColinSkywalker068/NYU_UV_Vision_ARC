/// @file aim_solver_node.cpp
/// @brief Implements target-to-yaw/pitch solving and legacy command publication.

#include "yolo_cpp/aim_solver_node.hpp"

#include "yolo_cpp/utils.hpp"

#include <rclcpp_components/register_node_macro.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>

namespace yolo_cpp
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

/// Convert a fixed xyxy integer box array into an OpenCV rectangle.
cv::Rect rect_from_xyxy(const std::array<int32_t, 4> & bbox_xyxy)
{
  const int x1 = bbox_xyxy[0];
  const int y1 = bbox_xyxy[1];
  const int x2 = bbox_xyxy[2];
  const int y2 = bbox_xyxy[3];
  return cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1));
}

}  // namespace

/// Declare solver parameters and connect target, attitude, and output topics.
AimSolverNode::AimSolverNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("aim_solver_node", options)
{
  this->declare_parameter<double>("hfov_degrees", 79.2);
  this->declare_parameter<double>("vfov_degrees", 63.78);
  this->declare_parameter<bool>("publish_zero_angles_when_no_target", true);
  this->declare_parameter<int>("default_image_width", default_image_width_);
  this->declare_parameter<int>("default_image_height", default_image_height_);
  this->declare_parameter<std::string>("target_topic", target_topic_);
  this->declare_parameter<bool>("use_3d_solution", use_3d_solution_);
  this->declare_parameter<std::string>("output_mode", output_mode_);
  this->declare_parameter<double>("default_bullet_speed_mps", default_bullet_speed_mps_);
  this->declare_parameter<double>("solver_latency_s", solver_latency_s_);
  this->declare_parameter<bool>("compensate_gravity", compensate_gravity_);
  this->declare_parameter<double>("gravity_mps2", gravity_mps2_);

  hfov_rad_ = static_cast<float>(this->get_parameter("hfov_degrees").as_double() * kPi / 180.0);
  vfov_rad_ = static_cast<float>(this->get_parameter("vfov_degrees").as_double() * kPi / 180.0);
  publish_zero_angles_when_no_target_ =
    this->get_parameter("publish_zero_angles_when_no_target").as_bool();
  default_image_width_ = this->get_parameter("default_image_width").as_int();
  default_image_height_ = this->get_parameter("default_image_height").as_int();
  target_topic_ = this->get_parameter("target_topic").as_string();
  use_3d_solution_ = this->get_parameter("use_3d_solution").as_bool();
  output_mode_ = this->get_parameter("output_mode").as_string();
  default_bullet_speed_mps_ =
    static_cast<float>(this->get_parameter("default_bullet_speed_mps").as_double());
  bullet_speed_mps_ = default_bullet_speed_mps_;
  solver_latency_s_ = static_cast<float>(this->get_parameter("solver_latency_s").as_double());
  compensate_gravity_ = this->get_parameter("compensate_gravity").as_bool();
  gravity_mps2_ = static_cast<float>(this->get_parameter("gravity_mps2").as_double());

  target_subscription_ = this->create_subscription<cv_auto_aim_interfaces::msg::Target>(
    target_topic_, 10, std::bind(&AimSolverNode::handle_target, this, std::placeholders::_1));
  attitude_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    "attitude_data", 10, std::bind(&AimSolverNode::handle_attitude, this, std::placeholders::_1));
  attitude_flags_subscription_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
    "attitude_flags", 10,
    std::bind(&AimSolverNode::handle_attitude_flags, this, std::placeholders::_1));
  bbox_publisher_ =
    this->create_publisher<std_msgs::msg::Int32MultiArray>("bounding_box", 10);
  override_publisher_ = this->create_publisher<std_msgs::msg::Bool>("override", 10);
  suggest_fire_publisher_ = this->create_publisher<std_msgs::msg::Bool>("suggest_fire", 10);
  angle_publisher_ =
    this->create_publisher<std_msgs::msg::Float32MultiArray>("bounding_box_angles", 10);
  obb_angle_publisher_ =
    this->create_publisher<std_msgs::msg::Float32>("obb_angle", 10);
  aim_solution_publisher_ =
    this->create_publisher<cv_auto_aim_interfaces::msg::AimSolution>("aim_solution", 10);

  RCLCPP_INFO(this->get_logger(), "Initialized AimSolverNode");
}

void AimSolverNode::handle_target(const cv_auto_aim_interfaces::msg::Target::SharedPtr msg)
{
  // Keep legacy command topics populated even when the richer AimSolution message is used.
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

  float flight_time = 0.0F;
  float range = 0.0F;
  bool used_3d = false;
  auto [yaw, pitch] = calculate_angles(bbox, image_width, image_height, hfov_rad_, vfov_rad_);
  if (use_3d_solution_ && msg->has_pose) {
    auto solved = solve_3d_angles(*msg, &flight_time, &range);
    yaw = solved.first;
    pitch = solved.second;
    used_3d = true;
  }
  if (output_mode_ == "absolute" && have_attitude_) {
    yaw += current_yaw_rad_;
    pitch += current_pitch_rad_;
  }
  angle_msg_.data = {yaw, pitch};
  angle_publisher_->publish(angle_msg_);

  suggest_fire_msg_.data = msg->suggest_fire;
  suggest_fire_publisher_->publish(suggest_fire_msg_);

  obb_angle_msg_.data = msg->has_obb_angle ? msg->obb_angle : 0.0F;
  obb_angle_publisher_->publish(obb_angle_msg_);

  cv_auto_aim_interfaces::msg::AimSolution solution;
  solution.header = msg->header;
  solution.has_solution = true;
  solution.used_3d = used_3d;
  solution.suggest_fire = msg->suggest_fire;
  solution.yaw = yaw;
  solution.pitch = pitch;
  solution.yaw_degrees = static_cast<float>(yaw * 180.0 / kPi);
  solution.pitch_degrees = static_cast<float>(pitch * 180.0 / kPi);
  solution.flight_time = flight_time;
  solution.range = used_3d ? range : msg->range;
  solution.bullet_speed = bullet_speed_mps_;
  aim_solution_publisher_->publish(solution);
}

void AimSolverNode::handle_attitude(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 2) {
    return;
  }
  current_yaw_rad_ = msg->data[0];
  current_pitch_rad_ = msg->data[1];
  have_attitude_ = true;
}

void AimSolverNode::handle_attitude_flags(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 3) {
    return;
  }
  if (msg->data[2] > 0) {
    bullet_speed_mps_ = static_cast<float>(msg->data[2]);
  } else {
    bullet_speed_mps_ = default_bullet_speed_mps_;
  }
}

std::pair<float, float> AimSolverNode::solve_3d_angles(
  const cv_auto_aim_interfaces::msg::Target & target, float * flight_time, float * range) const
{
  // Iterate because target lead depends on flight time, and flight time depends on led range.
  const float speed = std::max(1.0F, bullet_speed_mps_);
  double x = target.pose.position.x;
  double y = target.pose.position.y;
  double z = target.pose.position.z;
  double t = 0.0;

  for (int i = 0; i < 3; ++i) {
    const double distance = std::sqrt((x * x) + (y * y) + (z * z));
    t = distance / speed;
    const double predict_dt = solver_latency_s_ + t;
    if (target.has_velocity) {
      x = target.pose.position.x + (target.velocity.x * predict_dt);
      y = target.pose.position.y + (target.velocity.y * predict_dt);
      z = target.pose.position.z + (target.velocity.z * predict_dt);
    }
  }

  const double horizontal = std::hypot(x, z);
  const double final_range = std::sqrt((x * x) + (y * y) + (z * z));
  const double drop = compensate_gravity_ ? 0.5 * gravity_mps2_ * t * t : 0.0;
  const double yaw = -std::atan2(x, z);
  const double pitch = std::atan2(y - drop, horizontal);

  if (flight_time != nullptr) {
    *flight_time = static_cast<float>(t);
  }
  if (range != nullptr) {
    *range = static_cast<float>(final_range);
  }
  return {static_cast<float>(yaw), static_cast<float>(pitch)};
}

void AimSolverNode::publish_no_target(int image_width, int image_height)
{
  const int cx = std::max(0, image_width / 2);
  const int cy = std::max(0, image_height / 2);
  bbox_msg_.data = {cx, cy, cx, cy};
  override_msg_.data = false;
  suggest_fire_msg_.data = false;
  obb_angle_msg_.data = 0.0F;

  bbox_publisher_->publish(bbox_msg_);
  override_publisher_->publish(override_msg_);
  suggest_fire_publisher_->publish(suggest_fire_msg_);
  angle_msg_.data = {0.0F, 0.0F};
  angle_publisher_->publish(angle_msg_);
  obb_angle_publisher_->publish(obb_angle_msg_);

  cv_auto_aim_interfaces::msg::AimSolution solution;
  solution.has_solution = false;
  solution.used_3d = false;
  solution.suggest_fire = false;
  solution.bullet_speed = bullet_speed_mps_;
  aim_solution_publisher_->publish(solution);
}

}  // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::AimSolverNode)
