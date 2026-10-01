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
  // Retain the old launch argument; the shared history now retains two seconds.
  this->declare_parameter<double>("attitude_history_seconds", 2.0);
  attitude_max_gap_ms_ = this->declare_parameter<double>("attitude_max_gap_ms", 50.0);
  prediction_config_.actuator_delay_s = this->declare_parameter<double>("actuator_delay_s", 0.0);
  prediction_config_.max_frame_age_s = this->declare_parameter<double>("max_frame_age_s", 0.25);
  prediction_config_.max_measurement_age_s = this->declare_parameter<double>("max_measurement_age_s", 0.15);
  prediction_config_.yaw_speed_rad_s = this->declare_parameter<double>("yaw_max_slew_deg_s", 60.0) * kPi / 180;
  prediction_config_.pitch_speed_rad_s = this->declare_parameter<double>("pitch_max_slew_deg_s", 60.0) * kPi / 180;
  prediction_config_.yaw_accel_rad_s2 = this->declare_parameter<double>("yaw_max_accel_deg_s2", 360.0) * kPi / 180;
  prediction_config_.pitch_accel_rad_s2 = this->declare_parameter<double>("pitch_max_accel_deg_s2", 240.0) * kPi / 180;
  prediction_config_.plate_switch_margin = this->declare_parameter<double>("plate_switch_margin", 0.15);
  this->declare_parameter<double>("attitude_sync_tolerance_ms", attitude_sync_tolerance_ms_);
  this->declare_parameter<double>("gravity_mps2", gravity_mps2_);

  hfov_rad_ = static_cast<float>(this->get_parameter("hfov_degrees").as_double() * kPi / 180.0);
  vfov_rad_ = static_cast<float>(this->get_parameter("vfov_degrees").as_double() * kPi / 180.0);
  publish_zero_angles_when_no_target_ =
    this->get_parameter("publish_zero_angles_when_no_target").as_bool();
  enable_fire_recommendation_ =
    this->get_parameter("enable_fire_recommendation").as_bool();
  if (this->declare_parameter<bool>("calibration_provisional", false)) {
    enable_fire_recommendation_ = false;
    this->set_parameter(rclcpp::Parameter("enable_fire_recommendation", false));
    RCLCPP_WARN(this->get_logger(), "UNCALIBRATED preview: fire recommendations locked off");
  }
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

  attitude_sync_tolerance_ms_ = std::max(
    0.0, this->get_parameter("attitude_sync_tolerance_ms").as_double());

  target_subscription_ = this->create_subscription<cv_auto_aim_interfaces::msg::Target>(
    target_topic_, 10, std::bind(&AimSolverNode::handle_target, this, std::placeholders::_1));
  attitude_subscription_ = this->create_subscription<cv_auto_aim_interfaces::msg::GimbalAttitude>(
    "gimbal/attitude", 100, std::bind(&AimSolverNode::handle_attitude, this, std::placeholders::_1));
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
  const int image_width = msg->image_width > 0 ? msg->image_width : default_image_width_;
  const int image_height = msg->image_height > 0 ? msg->image_height : default_image_height_;
  const auto now_ns = this->now().nanoseconds();
  const auto stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
  std::optional<AttitudeSample> current;
  std::optional<AttitudeSample> capture;
  {
    std::lock_guard<std::mutex> lock(attitude_mutex_);
    current = attitude_history_.latest(now_ns);
    capture = attitude_history_.at(stamp_ns, now_ns,
      attitude_sync_tolerance_ms_ / 1000.0, attitude_max_gap_ms_ / 1000.0);
  }
  const char * rejection = nullptr;
  if (!msg->has_target) {rejection = "no_target";}
  else if (use_3d_solution_ && !msg->has_pose) {rejection = "missing_3d_pose";}
  else if (!current) {rejection = "stale_or_missing_attitude";}
  else if (!capture || !msg->camera_pose_valid) {rejection = "capture_attitude_mismatch";}
  else if (msg->header.frame_id != "gimbal_inertial") {rejection = "wrong_tracking_frame";}
  else if (stamp_ns <= 0 || stamp_ns > now_ns ||
    (now_ns - stamp_ns) * 1e-9 > prediction_config_.max_frame_age_s) {rejection = "invalid_frame_time";}
  else if (msg->measurement_stamp_ns <= 0 || msg->measurement_stamp_ns > stamp_ns ||
    (now_ns - msg->measurement_stamp_ns) * 1e-9 > prediction_config_.max_measurement_age_s)
  {rejection = "stale_or_invalid_measurement";}
  if (rejection) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
      "AIM_SYNC rejected: %s", rejection);
    publish_no_target(image_width, image_height, rejection);
    return;
  }
  const int selected_identity = msg->has_robot_state ? -(msg->robot_track_id + 2) : msg->track_id;
  if (previous_track_id_ != selected_identity) {
    previous_slot_ = -1;
    previous_track_id_ = selected_identity;
  }
  const auto bbox = rect_from_xyxy(msg->bbox_xyxy);
  const auto & q = msg->camera_pose.orientation;
  const auto camera_rotation = quaternion_rotation({q.x, q.y, q.z, q.w});
  // Pinhole angular geometry, rather than a linear degrees-per-pixel approximation.
  const double cx = bbox.x + bbox.width * 0.5;
  const double cy = bbox.y + bbox.height * 0.5;
  const cv::Point3d ray((2 * cx / image_width - 1) * std::tan(hfov_rad_ / 2),
    (2 * cy / image_height - 1) * std::tan(vfov_rad_ / 2), 1.0);
  const auto inertial_ray = rotate_point(camera_rotation, ray);
  double yaw = std::atan2(inertial_ray.x, inertial_ray.z);
  double pitch = std::atan2(-inertial_ray.y, std::hypot(inertial_ray.x, inertial_ray.z));
  AimPredictionResult result;
  const bool used_3d = use_3d_solution_ && msg->has_pose;
  if (used_3d) {
    AimPredictionInput input;
    input.frame_stamp_ns = stamp_ns;
    input.measurement_stamp_ns = msg->measurement_stamp_ns;
    input.now_ns = now_ns;
    input.position = {msg->pose.position.x, msg->pose.position.y, msg->pose.position.z};
    input.velocity = {msg->velocity.x, msg->velocity.y, msg->velocity.z};
    input.has_velocity = msg->has_velocity;
    input.has_robot_state = msg->has_robot_state;
    input.robot.center = {msg->robot_center.x, msg->robot_center.y, msg->robot_center.z};
    input.robot.velocity = {msg->robot_velocity.x, msg->robot_velocity.y, msg->robot_velocity.z};
    input.robot.phase_rad = msg->robot_phase;
    input.robot.alternating_radius_m = msg->alternating_radius_m;
    input.robot.alternating_height_m = msg->alternating_height_m;
    input.robot.angular_velocity_rad_s = msg->robot_angular_velocity;
    input.armor_radius_m = msg->armor_radius;
    input.armor_count = msg->armor_count;
    input.current_attitude = *current;
    input.previous_slot = previous_slot_;
    auto config = prediction_config_;
    config.transport_delay_s = solver_latency_s_;
    config.bullet_speed_mps = bullet_speed_mps_;
    config.compensate_gravity = compensate_gravity_;
    config.gravity_mps2 = gravity_mps2_;
    result = predict_aim(input, config);
    if (!result.valid) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
        "AIM rejected: %s frame_age=%.3f measurement_age=%.3f", result.reason,
        result.frame_age_s, result.measurement_age_s);
      publish_no_target(image_width, image_height, result.reason);
      return;
    }
    yaw = result.yaw;
    pitch = result.pitch;
    previous_slot_ = result.selected_slot;
  }
  if (output_mode_ == "offset") {
    yaw = angle_difference(yaw, current->yaw);
    pitch -= current->pitch;
  }
  if (!std::isfinite(yaw) || !std::isfinite(pitch)) {
    publish_no_target(image_width, image_height);
    return;
  }
  // Fire remains opt-in; coasting and an unreachable command never qualify.
  const bool suggest_fire = enable_fire_recommendation_ && msg->suggest_fire &&
    !msg->is_coasting && used_3d && result.reachable;
  cv_auto_aim_interfaces::msg::AimSolution solution;
  solution.header = msg->header;
  solution.has_solution = true;
  solution.used_3d = used_3d;
  solution.suggest_fire = suggest_fire;
  solution.yaw = yaw;
  solution.pitch = pitch;
  solution.yaw_degrees = yaw * 180 / kPi;
  solution.pitch_degrees = pitch * 180 / kPi;
  solution.flight_time = result.flight_time_s;
  solution.range = result.range_m;
  solution.bullet_speed = bullet_speed_mps_;
  solution.frame_age_s = (now_ns - stamp_ns) * 1e-9;
  solution.measurement_age_s = (now_ns - msg->measurement_stamp_ns) * 1e-9;
  solution.prediction_horizon_s = result.horizon_s;
  solution.minimum_slew_s = result.minimum_slew_s;
  solution.reachable = result.reachable;
  solution.selected_slot = result.selected_slot;
  solution.output_mode = output_mode_;
  solution.status = used_3d ? result.reason : "2d_preview";
  aim_solution_publisher_->publish(solution);

  // Compatibility preview topics; the serial bridge consumes the atomic solution.
  bbox_msg_.data = {bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height};
  override_msg_.data = true;
  suggest_fire_msg_.data = suggest_fire;
  angle_msg_.data = {static_cast<float>(yaw), static_cast<float>(pitch)};
  obb_angle_msg_.data = msg->has_obb_angle ? msg->obb_angle : 0;
  bbox_publisher_->publish(bbox_msg_);
  override_publisher_->publish(override_msg_);
  suggest_fire_publisher_->publish(suggest_fire_msg_);
  angle_publisher_->publish(angle_msg_);
  obb_angle_publisher_->publish(obb_angle_msg_);
}

void AimSolverNode::handle_attitude(const cv_auto_aim_interfaces::msg::GimbalAttitude::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(attitude_mutex_);
  if (!msg->valid) {attitude_history_.clear(); return;}
  attitude_history_.insert({rclcpp::Time(msg->header.stamp).nanoseconds(),
    msg->yaw, msg->pitch, msg->roll});
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

void AimSolverNode::publish_no_target(int image_width, int image_height, const char * reason)
{
  previous_slot_ = -1;
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
  solution.header.stamp = this->now();
  solution.output_mode = output_mode_;
  solution.status = reason;
  solution.has_solution = false;
  solution.used_3d = false;
  solution.suggest_fire = false;
  solution.bullet_speed = bullet_speed_mps_;
  aim_solution_publisher_->publish(solution);
}

}  // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::AimSolverNode)
