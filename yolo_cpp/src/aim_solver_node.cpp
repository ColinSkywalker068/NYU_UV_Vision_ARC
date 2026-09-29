/// @file aim_solver_node.cpp
/// @brief Implements target-to-yaw/pitch solving and legacy command publication.

#include "yolo_cpp/aim_solver_node.hpp"
#include "yolo_cpp/robot_state_estimator.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <rclcpp_components/register_node_macro.hpp>
#include <utility>

#include "yolo_cpp/utils.hpp"

namespace yolo_cpp
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

/// Convert a fixed xyxy integer box array into an OpenCV rectangle.
cv::Rect rect_from_xyxy(const std::array<int32_t, 4>& bbox_xyxy)
{
  const int x1 = bbox_xyxy[0];
  const int y1 = bbox_xyxy[1];
  const int x2 = bbox_xyxy[2];
  const int y2 = bbox_xyxy[3];
  return cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1));
}

}  // namespace

/// Declare solver parameters and connect target, attitude, and output topics.
AimSolverNode::AimSolverNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("aim_solver_node", options)
{
  this->declare_parameter<double>("hfov_degrees", 79.2);
  this->declare_parameter<double>("vfov_degrees", 63.78);
  this->declare_parameter<bool>("publish_zero_angles_when_no_target", true);
  this->declare_parameter<bool>("enable_fire_recommendation", false);
  this->declare_parameter<int>("default_image_width", default_image_width_);
  this->declare_parameter<int>("default_image_height", default_image_height_);
  this->declare_parameter<std::string>("target_topic", target_topic_);
  this->declare_parameter<bool>("use_3d_solution", use_3d_solution_);
  this->declare_parameter<std::string>("output_mode", output_mode_);
  this->declare_parameter<double>("default_bullet_speed_mps", default_bullet_speed_mps_);
  this->declare_parameter<double>("solver_latency_s", solver_latency_s_);
  this->declare_parameter<bool>("compensate_gravity", compensate_gravity_);
  this->declare_parameter<double>("attitude_history_seconds", attitude_history_seconds_);
  this->declare_parameter<double>("attitude_sync_tolerance_ms", attitude_sync_tolerance_ms_);
  this->declare_parameter<double>("gravity_mps2", gravity_mps2_);

  hfov_rad_ = static_cast<float>(this->get_parameter("hfov_degrees").as_double() * kPi / 180.0);
  vfov_rad_ = static_cast<float>(this->get_parameter("vfov_degrees").as_double() * kPi / 180.0);
  publish_zero_angles_when_no_target_ =
    this->get_parameter("publish_zero_angles_when_no_target").as_bool();
  enable_fire_recommendation_ =
    this->get_parameter("enable_fire_recommendation").as_bool();
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
  attitude_history_seconds_ = std::max(
    0.1, this->get_parameter("attitude_history_seconds").as_double());
  attitude_sync_tolerance_ms_ = std::max(
    0.0, this->get_parameter("attitude_sync_tolerance_ms").as_double());

  target_subscription_ = this->create_subscription<cv_auto_aim_interfaces::msg::Target>(
    target_topic_, 10, std::bind(&AimSolverNode::handle_target, this, std::placeholders::_1));
  attitude_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    "attitude_data", 10, std::bind(&AimSolverNode::handle_attitude, this, std::placeholders::_1));
  attitude_flags_subscription_ =
    this->create_subscription<std_msgs::msg::Int32MultiArray>("attitude_flags", 10,
      std::bind(&AimSolverNode::handle_attitude_flags, this, std::placeholders::_1));
  bbox_publisher_ = this->create_publisher<std_msgs::msg::Int32MultiArray>("bounding_box", 10);
  override_publisher_ = this->create_publisher<std_msgs::msg::Bool>("override", 10);
  suggest_fire_publisher_ = this->create_publisher<std_msgs::msg::Bool>("suggest_fire", 10);
  angle_publisher_ =
    this->create_publisher<std_msgs::msg::Float32MultiArray>("bounding_box_angles", 10);
  obb_angle_publisher_ = this->create_publisher<std_msgs::msg::Float32>("obb_angle", 10);
  aim_solution_publisher_ =
    this->create_publisher<cv_auto_aim_interfaces::msg::AimSolution>("aim_solution", 10);

  if (enable_fire_recommendation_) {
    RCLCPP_WARN(
      this->get_logger(),
      "FIRE RECOMMENDATIONS ENABLED by explicit operator configuration");
  } else {
    RCLCPP_INFO(
      this->get_logger(),
      "Fire recommendations disabled. Set enable_fire_recommendation:=true only for competition");
  }
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

  float frame_yaw_rad = 0.0F;
  float frame_pitch_rad = 0.0F;
  if (output_mode_ == "absolute") {
    const rclcpp::Time frame_stamp(msg->header.stamp, this->get_clock()->get_clock_type());
    double nearest_offset_ms = -1.0;
    bool interpolated = false;
    if (!attitude_at(
        frame_stamp, &frame_yaw_rad, &frame_pitch_rad, &nearest_offset_ms, &interpolated))
    {
      const auto rejected = ++attitude_sync_rejected_;
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "AIM_SYNC rejected: nearest_offset_ms=%.2f tolerance_ms=%.2f aligned=%llu rejected=%llu",
        nearest_offset_ms, attitude_sync_tolerance_ms_,
        static_cast<unsigned long long>(attitude_sync_aligned_.load()),
        static_cast<unsigned long long>(rejected));
      publish_no_target(image_width, image_height);
      return;
    }
    const auto aligned = ++attitude_sync_aligned_;
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 2000,
      "AIM_SYNC aligned: mode=%s nearest_offset_ms=%.2f aligned=%llu rejected=%llu",
      interpolated ? "interpolated" : "edge", nearest_offset_ms,
      static_cast<unsigned long long>(aligned),
      static_cast<unsigned long long>(attitude_sync_rejected_.load()));
  }

  const auto bbox = rect_from_xyxy(msg->bbox_xyxy);
  bbox_msg_.data = {bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height};
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
  // Type C logical coordinates are +yaw right, +pitch up. Image solver
  // coordinates have both signs reversed. Convert before adding attitude.
  yaw = -yaw;
  pitch = -pitch;
  if (output_mode_ == "absolute") {
    yaw += frame_yaw_rad;
    pitch += frame_pitch_rad;
    yaw = std::remainder(yaw, static_cast<float>(2.0 * kPi));
  }
  angle_msg_.data = {yaw, pitch};
  angle_publisher_->publish(angle_msg_);

  const bool suggest_fire = enable_fire_recommendation_ && msg->suggest_fire;
  suggest_fire_msg_.data = suggest_fire;
  suggest_fire_publisher_->publish(suggest_fire_msg_);

  obb_angle_msg_.data = msg->has_obb_angle ? msg->obb_angle : 0.0F;
  obb_angle_publisher_->publish(obb_angle_msg_);

  cv_auto_aim_interfaces::msg::AimSolution solution;
  solution.header = msg->header;
  solution.has_solution = true;
  solution.used_3d = used_3d;
  solution.suggest_fire = suggest_fire;
  solution.yaw = yaw;
  solution.pitch = pitch;
  solution.yaw_degrees = static_cast<float>(yaw * 180.0 / kPi);
  solution.pitch_degrees = static_cast<float>(pitch * 180.0 / kPi);
  solution.flight_time = flight_time;
  solution.range = used_3d ? range : msg->range;
  solution.bullet_speed = bullet_speed_mps_;
  aim_solution_publisher_->publish(solution);
}

bool AimSolverNode::attitude_at(
  const rclcpp::Time& frame_stamp, float* yaw, float* pitch,
  double* nearest_offset_ms, bool* interpolated)
{
  std::lock_guard<std::mutex> lock(attitude_mutex_);
  *nearest_offset_ms = -1.0;
  *interpolated = false;
  if (!have_attitude_ || attitude_history_.empty() ||
    std::chrono::steady_clock::now() - attitude_received_ > std::chrono::milliseconds(250))
  {
    return false;
  }

  if (frame_stamp.nanoseconds() == 0) {
    *yaw = current_yaw_rad_;
    *pitch = current_pitch_rad_;
    *nearest_offset_ms = 0.0;
    return true;
  }

  const auto tolerance_ns = static_cast<std::int64_t>(attitude_sync_tolerance_ms_ * 1.0e6);
  const auto use_sample = [&](const AttitudeSample& sample) {
      *yaw = sample.yaw;
      *pitch = sample.pitch;
    };

  if (frame_stamp <= attitude_history_.front().stamp) {
    const auto offset_ns = (attitude_history_.front().stamp - frame_stamp).nanoseconds();
    *nearest_offset_ms = static_cast<double>(offset_ns) / 1.0e6;
    if (offset_ns > tolerance_ns) {
      return false;
    }
    use_sample(attitude_history_.front());
    return true;
  }
  if (frame_stamp >= attitude_history_.back().stamp) {
    const auto offset_ns = (frame_stamp - attitude_history_.back().stamp).nanoseconds();
    *nearest_offset_ms = static_cast<double>(offset_ns) / 1.0e6;
    if (offset_ns > tolerance_ns) {
      return false;
    }
    use_sample(attitude_history_.back());
    return true;
  }

  for (std::size_t i = 1; i < attitude_history_.size(); ++i) {
    const auto& upper = attitude_history_[i];
    if (upper.stamp < frame_stamp) {
      continue;
    }
    const auto& lower = attitude_history_[i - 1];
    const auto lower_offset_ns = (frame_stamp - lower.stamp).nanoseconds();
    const auto upper_offset_ns = (upper.stamp - frame_stamp).nanoseconds();
    *nearest_offset_ms = static_cast<double>(
      std::min(lower_offset_ns, upper_offset_ns)) / 1.0e6;
    *interpolated = true;
    const auto span_ns = (upper.stamp - lower.stamp).nanoseconds();
    if (span_ns <= 0) {
      use_sample(upper);
      return true;
    }
    const auto ratio =
      static_cast<float>(lower_offset_ns) / static_cast<float>(span_ns);
    const auto yaw_delta = std::remainder(
      upper.yaw - lower.yaw, static_cast<float>(2.0 * kPi));
    *yaw = std::remainder(
      lower.yaw + (yaw_delta * ratio), static_cast<float>(2.0 * kPi));
    *pitch = lower.pitch + ((upper.pitch - lower.pitch) * ratio);
    return true;
  }
  return false;
}

void AimSolverNode::handle_attitude(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(attitude_mutex_);
  if (msg->data.size() < 2 || !std::isfinite(msg->data[0]) || !std::isfinite(msg->data[1])) {
    have_attitude_ = false;
    attitude_history_.clear();
    return;
  }
  current_yaw_rad_ = msg->data[0];
  current_pitch_rad_ = msg->data[1];
  have_attitude_ = true;
  attitude_received_ = std::chrono::steady_clock::now();
  const auto stamp = this->get_clock()->now();
  attitude_history_.push_back(AttitudeSample{stamp, current_yaw_rad_, current_pitch_rad_});
  const auto history_ns = static_cast<std::int64_t>(attitude_history_seconds_ * 1.0e9);
  const auto cutoff = stamp - rclcpp::Duration::from_nanoseconds(history_ns);
  while (!attitude_history_.empty() && attitude_history_.front().stamp < cutoff) {
    attitude_history_.pop_front();
  }
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
  const cv_auto_aim_interfaces::msg::Target& target, float* flight_time, float* range) const
{
  // Iterate because target lead depends on flight time, and flight time depends on led range.
  const float speed = std::max(1.0F, bullet_speed_mps_);
  double x = target.pose.position.x;
  double y = target.pose.position.y;
  double z = target.pose.position.z;
  double t = 0.0;

  RobotEstimate robot;
  if (target.has_robot_state) {
    robot.initialized = true;
    robot.center = {target.robot_center.x, target.robot_center.y, target.robot_center.z};
    robot.velocity = {
      target.robot_velocity.x, target.robot_velocity.y, target.robot_velocity.z};
    robot.phase_rad = target.robot_phase;
    robot.angular_velocity_rad_s = target.robot_angular_velocity;
  }

  for (int i = 0; i < 3; ++i) {
    const double distance = std::sqrt((x * x) + (y * y) + (z * z));
    t = distance / speed;
    const double predict_dt = solver_latency_s_ + t;
    if (target.has_robot_state && target.armor_count > 0 && target.armor_radius > 0.0F) {
      const auto armor = select_future_armor(
        robot, predict_dt, target.armor_radius, target.armor_count);
      x = armor.position.x;
      y = armor.position.y;
      z = armor.position.z;
    } else if (target.has_velocity) {
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
  if (publish_zero_angles_when_no_target_) {
    angle_msg_.data = {0.0F, 0.0F};
    angle_publisher_->publish(angle_msg_);
  }
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
