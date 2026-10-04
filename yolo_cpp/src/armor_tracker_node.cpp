/// @file armor_tracker_node.cpp
/// @brief Implements ROS bindings around TrackManager and target selection messages.

#include "yolo_cpp/armor_tracker_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <rclcpp_components/register_node_macro.hpp>
#include <utility>
#include <vector>

#include "cv_auto_aim_interfaces/msg/armor.hpp"
#include "cv_auto_aim_interfaces/msg/tracked_armor.hpp"
#include "yolo_cpp/utils.hpp"

namespace yolo_cpp
{

namespace
{

void assign_orientation(geometry_msgs::msg::Quaternion & q, const std::array<double, 4> & values)
{
  q.x = values[0]; q.y = values[1]; q.z = values[2]; q.w = values[3];
}

/// Convert internal TrackState values to cv_auto_aim_interfaces constants.
std::uint8_t to_msg_state(TrackState state)
{
  switch (state) {
    case TrackState::kDetecting:
      return cv_auto_aim_interfaces::msg::TrackedArmor::DETECTING;
    case TrackState::kTracking:
      return cv_auto_aim_interfaces::msg::TrackedArmor::TRACKING;
    case TrackState::kCoasting:
      return cv_auto_aim_interfaces::msg::TrackedArmor::COASTING;
  }
  return cv_auto_aim_interfaces::msg::TrackedArmor::DETECTING;
}

/// Convert a tracker observation back to an Armor message for tracked outputs.
cv_auto_aim_interfaces::msg::Armor make_armor_message(const TrackObservation& observation)
{
  cv_auto_aim_interfaces::msg::Armor armor;
  armor.bbox_xyxy = {observation.bbox.x, observation.bbox.y,
    observation.bbox.x + observation.bbox.width, observation.bbox.y + observation.bbox.height};
  armor.center_x = observation.center.x;
  armor.center_y = observation.center.y;
  armor.confidence = observation.confidence;
  armor.class_id = observation.class_id;
  armor.is_enemy = observation.is_enemy;
  armor.has_obb_angle = observation.obb_angle.has_value();
  armor.obb_angle = observation.obb_angle.value_or(0.0F);
  armor.image_width = observation.image_width;
  armor.image_height = observation.image_height;
  armor.has_corners = observation.has_corners;
  if (observation.has_corners) {
    for (std::size_t i = 0; i < observation.corners_source.size(); ++i) {
      armor.corners[i].x = observation.corners_source[i].x;
      armor.corners[i].y = observation.corners_source[i].y;
      armor.corners[i].z = 0.0F;
    }
  }
  armor.armor_type = static_cast<uint8_t>(observation.armor_type);
  armor.armor_number = observation.armor_number;
  armor.distance_to_image_center = observation.distance_to_image_center;
  armor.corners_are_keypoints = observation.corners_are_keypoints;
  armor.pose_normal_reliable = observation.pose_normal_reliable;
  armor.reprojection_rms_px = observation.reprojection_rms_px;
  armor.has_pose = observation.has_camera_pose;
  if (observation.has_camera_pose) {
    armor.pose_camera.position.x = observation.camera_position.x;
    armor.pose_camera.position.y = observation.camera_position.y;
    armor.pose_camera.position.z = observation.camera_position.z;
    assign_orientation(armor.pose_camera.orientation, observation.camera_orientation);
  }
  return armor;
}

/// Convert an Armor message into the normalized tracker observation type.
TrackObservation make_track_observation(const cv_auto_aim_interfaces::msg::Armor& armor)
{
  const int x1 = armor.bbox_xyxy[0];
  const int y1 = armor.bbox_xyxy[1];
  const int x2 = armor.bbox_xyxy[2];
  const int y2 = armor.bbox_xyxy[3];
  TrackObservation observation;
  observation.bbox = cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1));
  observation.center = cv::Point2f(armor.center_x, armor.center_y);
  observation.confidence = armor.confidence;
  observation.class_id = armor.class_id;
  observation.is_enemy = armor.is_enemy;
  if (armor.has_obb_angle) {
    observation.obb_angle = armor.obb_angle;
  }
  observation.image_width = armor.image_width;
  observation.image_height = armor.image_height;
  observation.has_corners = armor.has_corners;
  if (armor.has_corners) {
    for (std::size_t i = 0; i < observation.corners_source.size(); ++i) {
      observation.corners_source[i] = cv::Point2f(armor.corners[i].x, armor.corners[i].y);
    }
  }
  observation.armor_type = armor.armor_type;
  observation.armor_number = armor.armor_number;
  observation.distance_to_image_center = armor.distance_to_image_center;
  observation.has_pose = armor.has_pose;
  observation.has_camera_pose = armor.has_pose;
  observation.corners_are_keypoints = armor.corners_are_keypoints;
  observation.pose_normal_reliable = armor.pose_normal_reliable;
  observation.reprojection_rms_px = armor.reprojection_rms_px;
  if (armor.has_pose) {
    observation.position = cv::Point3d(
      armor.pose_camera.position.x, armor.pose_camera.position.y, armor.pose_camera.position.z);
    observation.camera_position = observation.position;
    const auto & q = armor.pose_camera.orientation;
    observation.orientation = {q.x, q.y, q.z, q.w};
    observation.camera_orientation = observation.orientation;
  }
  return observation;
}

}  // namespace

/// Declare tracker parameters and wire detector input to tracked output topics.
ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("armor_tracker_node", options)
{
  tracking_frame_ = this->declare_parameter<std::string>("tracking_frame", "gimbal_inertial");
  attitude_edge_tolerance_s_ = this->declare_parameter<double>("attitude_sync_tolerance_ms", 30.0) / 1000.0;
  attitude_max_gap_s_ = this->declare_parameter<double>("attitude_max_gap_ms", 50.0) / 1000.0;
  max_frame_age_s_ = this->declare_parameter<double>("max_frame_age_s", 0.25);
  const auto translation = this->declare_parameter<std::vector<double>>("camera_translation_m", {0.0, 0.0, 0.0});
  const auto rotation = this->declare_parameter<std::vector<double>>("camera_mount_yaw_pitch_roll", {0.0, 0.0, 0.0});
  if (translation.size() != 3 || rotation.size() != 3 ||
    !std::all_of(translation.begin(), translation.end(), [](double v) {return std::isfinite(v);}) ||
    !std::all_of(rotation.begin(), rotation.end(), [](double v) {return std::isfinite(v);})) {
    throw std::runtime_error("Camera extrinsics require three values each");
  }
  camera_mount_translation_ = {translation[0], translation[1], translation[2]};
  camera_mount_rotation_ = camera_to_inertial_rotation({1, rotation[0], rotation[1], rotation[2]});
  this->declare_parameter<bool>("use_robot_id_targeting", false);
  this->declare_parameter<int>("enemy_color_red_value", 0);
  this->declare_parameter<int>("enemy_color_blue_value", 1);
  this->declare_parameter<int>("track_min_hits", 2);
  this->declare_parameter<int>("track_max_coast_frames", 0);
  this->declare_parameter<int>("track_drop_frames", 0);
  this->declare_parameter<double>("association_max_center_distance_ratio", 0.30);
  this->declare_parameter<double>("association_min_iou", 0.0);
  this->declare_parameter<double>("target_switch_min_score_delta", 0.08);
  this->declare_parameter<int>("target_lock_hysteresis_frames", 6);
  this->declare_parameter<int>("stage_log_interval", 300);

  use_robot_id_targeting_ = this->get_parameter("use_robot_id_targeting").as_bool();
  enemy_color_red_value_ = this->get_parameter("enemy_color_red_value").as_int();
  enemy_color_blue_value_ = this->get_parameter("enemy_color_blue_value").as_int();
  stage_log_interval_ =
    std::max(1, static_cast<int>(this->get_parameter("stage_log_interval").as_int()));

  TrackConfig config;
  config.track_min_hits = this->get_parameter("track_min_hits").as_int();
  config.track_max_coast_frames = this->get_parameter("track_max_coast_frames").as_int();
  config.track_drop_frames = this->get_parameter("track_drop_frames").as_int();
  config.track_max_coast_s = this->declare_parameter<double>("track_max_coast_s", 0.12);
  config.track_drop_s = this->declare_parameter<double>("track_drop_s", 0.35);
  config.position_filter_tau_s = this->declare_parameter<double>("position_filter_tau_s", 0.035);
  config.velocity_filter_tau_s = this->declare_parameter<double>("velocity_filter_tau_s", 0.10);
  config.association_max_center_distance_ratio =
    this->get_parameter("association_max_center_distance_ratio").as_double();
  config.association_min_iou = this->get_parameter("association_min_iou").as_double();
  this->declare_parameter<double>("association_max_3d_distance_m", 0.35);
  config.association_max_3d_distance_m =
    this->get_parameter("association_max_3d_distance_m").as_double();
  config.pose_innovation_confirm_m = this->declare_parameter<double>("pose_innovation_confirm_m", 0.20);
  config.pose_max_reprojection_rms_px = this->declare_parameter<double>("pose_max_reprojection_rms_px", 5.0);
  config.target_switch_min_score_delta =
    this->get_parameter("target_switch_min_score_delta").as_double();
  config.target_lock_hysteresis_frames =
    this->get_parameter("target_lock_hysteresis_frames").as_int();
  track_manager_.set_config(config);

  armors_subscription_ =
    this->create_subscription<cv_auto_aim_interfaces::msg::Armors>("detector/armors", 10,
      std::bind(&ArmorTrackerNode::armors_callback, this, std::placeholders::_1));
  attitude_subscription_ = this->create_subscription<cv_auto_aim_interfaces::msg::GimbalAttitude>(
    "gimbal/attitude", 100, std::bind(&ArmorTrackerNode::handle_attitude, this, std::placeholders::_1));
  robot_id_subscription_ = this->create_subscription<std_msgs::msg::Int32>("robot_id", 10,
    std::bind(&ArmorTrackerNode::id_listener_callback, this, std::placeholders::_1));
  attitude_flags_subscription_ =
    this->create_subscription<std_msgs::msg::Int32MultiArray>("attitude_flags", 10,
      std::bind(&ArmorTrackerNode::handle_attitude_flags, this, std::placeholders::_1));
  tracked_armors_publisher_ = this->create_publisher<cv_auto_aim_interfaces::msg::TrackedArmors>(
    "tracker/tracked_armors", 10);
  target_publisher_ =
    this->create_publisher<cv_auto_aim_interfaces::msg::Target>("tracker/target", 10);

  RCLCPP_INFO(this->get_logger(), "Initialized ArmorTrackerNode");
}

void ArmorTrackerNode::armors_callback(const cv_auto_aim_interfaces::msg::Armors::SharedPtr msg)
{
  // Keep association state under a mutex because color-control callbacks can update target class.
  std::lock_guard<std::mutex> lock(state_mutex_);

  rclcpp::Time stamp(msg->header.stamp);
  const auto now = this->get_clock()->now();
  if (stamp.nanoseconds() <= 0 || stamp > now || (now-stamp).seconds() > max_frame_age_s_ ||
      (have_last_frame_time_ && stamp <= last_frame_time_)) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
      "TRACK_TIME rejected zero, future, stale or out-of-order image");
    return;
  }
  const auto attitude = attitude_history_.at(stamp.nanoseconds(), now.nanoseconds(),
    attitude_edge_tolerance_s_, attitude_max_gap_s_);
  if ((!attitude && frame_camera_pose_valid_) ||
      (have_last_frame_time_ && (stamp - last_frame_time_).seconds() > max_frame_age_s_)) {
    track_manager_.reset();
  }
  frame_camera_pose_valid_ = attitude.has_value();
  if (attitude) {
    const auto pivot_rotation = camera_to_inertial_rotation(*attitude);
    frame_rotation_ = pivot_rotation * camera_mount_rotation_;
    frame_camera_position_ = rotate_point(pivot_rotation, camera_mount_translation_);
  }
  double dt = 1.0 / 60.0;
  if (have_last_frame_time_) {
    dt = (stamp - last_frame_time_).seconds();
    if (dt <= 0.0) {
      dt = 0.001;
    }
  }
  last_frame_time_ = stamp;
  have_last_frame_time_ = true;

  std::vector<TrackObservation> observations;
  observations.reserve(msg->armors.size());
  for (const auto& armor : msg->armors) {
    auto observation = make_track_observation(armor);
    observation.measurement_stamp_ns = stamp.nanoseconds();
    if (observation.has_pose && frame_camera_pose_valid_) {
      observation.position = rotate_point(frame_rotation_, observation.camera_position) + frame_camera_position_;
      const auto rotation = frame_rotation_ * quaternion_rotation(observation.camera_orientation);
      observation.orientation = rotation_quaternion(rotation);
      observation.has_pose = finite_point(observation.position) &&
        std::all_of(observation.orientation.begin(), observation.orientation.end(),
          [](double value) {return std::isfinite(value);});
    } else {
      observation.has_pose = false;
    }
    observations.push_back(std::move(observation));
  }

  const auto tracking_start = std::chrono::steady_clock::now();
  const auto result = track_manager_.update(observations, dt);
  const auto tracking_done = std::chrono::steady_clock::now();
  const auto publish_start = std::chrono::steady_clock::now();
  publish_tracked_armors(*msg, result);
  publish_target(*msg, result.selected_target);
  const auto publish_done = std::chrono::steady_clock::now();

  ++frame_count_;
  stage_totals_.tracking_ms +=
    std::chrono::duration<double, std::milli>(tracking_done - tracking_start).count();
  stage_totals_.publish_ms +=
    std::chrono::duration<double, std::milli>(publish_done - publish_start).count();

  if (frame_count_ % stage_log_interval_ == 0) {
    log_stage_timing();
  }
}

void ArmorTrackerNode::handle_attitude(const cv_auto_aim_interfaces::msg::GimbalAttitude::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!msg->valid) {attitude_history_.clear(); return;}
  attitude_history_.insert({rclcpp::Time(msg->header.stamp).nanoseconds(), msg->yaw, msg->pitch, msg->roll});
}

void ArmorTrackerNode::id_listener_callback(const std_msgs::msg::Int32::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!use_robot_id_targeting_ || stm_color_locked_) {
    return;
  }
  if (is_red_team(msg->data)) {
    target_class_ = BLUE_CLASS;
  } else if (is_blue_team(msg->data)) {
    target_class_ = RED_CLASS;
  }
}

void ArmorTrackerNode::handle_attitude_flags(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
{
  if (msg->data.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(state_mutex_);
  const int enemy_color = msg->data[0];
  if (enemy_color == enemy_color_red_value_) {
    target_class_ = RED_CLASS;
    stm_color_locked_ = true;
  } else if (enemy_color == enemy_color_blue_value_) {
    target_class_ = BLUE_CLASS;
    stm_color_locked_ = true;
  }
}

void ArmorTrackerNode::publish_tracked_armors(
  const cv_auto_aim_interfaces::msg::Armors& source_msg, const TrackingFrameResult& result)
{
  cv_auto_aim_interfaces::msg::TrackedArmors msg;
  msg.header = source_msg.header;
  msg.source_id = source_msg.source_id;
  msg.selected_track_id = result.selected_track_id.value_or(-1);
  msg.pose_frame_id = tracking_frame_;
  msg.camera_pose_valid = frame_camera_pose_valid_;
  msg.camera_pose.position.x = frame_camera_position_.x;
  msg.camera_pose.position.y = frame_camera_position_.y;
  msg.camera_pose.position.z = frame_camera_position_.z;
  assign_orientation(msg.camera_pose.orientation, rotation_quaternion(frame_rotation_));
  msg.raw_count = result.raw_count;
  msg.enemy_count = result.enemy_count;
  msg.posed_count = result.posed_count;
  msg.matched_count = result.matched_count;
  msg.quality_rejected_count = result.quality_rejected_count;
  msg.innovation_rejected_count = result.innovation_rejected_count;
  msg.tracks.reserve(result.tracks.size());
  for (const auto& track : result.tracks) {
    cv_auto_aim_interfaces::msg::TrackedArmor tracked;
    tracked.armor = make_armor_message(track.observation);
    tracked.track_id = track.track_id;
    tracked.state = to_msg_state(track.state);
    tracked.age = track.age;
    tracked.hit_streak = track.hit_streak;
    tracked.lost_frames = track.lost_frames;
    tracked.real_measurement_age_s = track.real_measurement_age_s;
    tracked.innovation_m = track.innovation_m;
    tracked.association_reason = track.association_reason;
    tracked.predicted_center_x = track.predicted_center.x;
    tracked.predicted_center_y = track.predicted_center.y;
    tracked.measurement_stamp_ns = track.observation.measurement_stamp_ns;
    tracked.has_pose = track.has_pose && frame_camera_pose_valid_;
    tracked.pose_camera = tracked.armor.pose_camera;
    if (tracked.has_pose) {
      tracked.pose_tracking.position.x = track.observation.position.x;
      tracked.pose_tracking.position.y = track.observation.position.y;
      tracked.pose_tracking.position.z = track.observation.position.z;
      assign_orientation(tracked.pose_tracking.orientation, track.observation.orientation);
      tracked.predicted_position_tracking.x = track.predicted_position.x;
      tracked.predicted_position_tracking.y = track.predicted_position.y;
      tracked.predicted_position_tracking.z = track.predicted_position.z;
      const auto camera_prediction = rotate_point(frame_rotation_.t(), track.predicted_position-frame_camera_position_);
      tracked.predicted_position_camera.x = camera_prediction.x;
      tracked.predicted_position_camera.y = camera_prediction.y;
      tracked.predicted_position_camera.z = camera_prediction.z;
    }
    tracked.has_velocity = track.has_velocity && tracked.has_pose;
    if (tracked.has_velocity) {
      tracked.velocity_tracking.x = track.velocity[0];
      tracked.velocity_tracking.y = track.velocity[1];
      tracked.velocity_tracking.z = track.velocity[2];
      const auto camera_velocity = frame_rotation_.t() * track.velocity;
      tracked.velocity_camera.x = camera_velocity[0];
      tracked.velocity_camera.y = camera_velocity[1];
      tracked.velocity_camera.z = camera_velocity[2];
    }
    msg.tracks.push_back(std::move(tracked));
  }
  tracked_armors_publisher_->publish(std::move(msg));
}

void ArmorTrackerNode::publish_target(
  const cv_auto_aim_interfaces::msg::Armors& source_msg, const std::optional<TrackSnapshot>& target)
{
  cv_auto_aim_interfaces::msg::Target msg;
  msg.header = source_msg.header;
  msg.header.frame_id = tracking_frame_;
  msg.camera_pose_valid = frame_camera_pose_valid_;
  msg.camera_pose.position.x = frame_camera_position_.x;
  msg.camera_pose.position.y = frame_camera_position_.y;
  msg.camera_pose.position.z = frame_camera_position_.z;
  assign_orientation(msg.camera_pose.orientation, rotation_quaternion(frame_rotation_));
  msg.source_id = source_msg.source_id;
  if (!target.has_value()) {
    msg.has_target = false;
    msg.is_coasting = false;
    msg.track_id = -1;
    msg.state = cv_auto_aim_interfaces::msg::Target::DETECTING;
    msg.tracking = false;
    msg.suggest_fire = false;
    msg.has_pose = false;
    msg.has_velocity = false;
    msg.bbox_xyxy = {0, 0, 0, 0};
    if (!source_msg.armors.empty()) {
      msg.image_width = source_msg.armors.front().image_width;
      msg.image_height = source_msg.armors.front().image_height;
    }
    target_publisher_->publish(std::move(msg));
    return;
  }

  const auto& snapshot = *target;
  msg.has_target = true;
  msg.tracking = true;
  msg.suggest_fire = snapshot.state == TrackState::kTracking && snapshot.real_measurement_age_s == 0.0;
  msg.is_coasting = snapshot.state == TrackState::kCoasting;
  msg.track_id = snapshot.track_id;
  msg.bbox_xyxy = {snapshot.observation.bbox.x, snapshot.observation.bbox.y,
    snapshot.observation.bbox.x + snapshot.observation.bbox.width,
    snapshot.observation.bbox.y + snapshot.observation.bbox.height};
  msg.center_x = snapshot.observation.center.x;
  msg.center_y = snapshot.observation.center.y;
  msg.predicted_center_x = snapshot.predicted_center.x;
  msg.predicted_center_y = snapshot.predicted_center.y;
  msg.confidence = snapshot.observation.confidence;
  msg.class_id = snapshot.observation.class_id;
  msg.is_enemy = snapshot.observation.is_enemy;
  msg.has_obb_angle = snapshot.observation.obb_angle.has_value();
  msg.obb_angle = snapshot.observation.obb_angle.value_or(0.0F);
  msg.state = to_msg_state(snapshot.state);
  msg.image_width = snapshot.observation.image_width;
  msg.image_height = snapshot.observation.image_height;
  msg.measurement_stamp_ns = snapshot.observation.measurement_stamp_ns;
  msg.has_pose = snapshot.has_pose && frame_camera_pose_valid_;
  if (msg.has_pose) {
    msg.pose.position.x = snapshot.observation.position.x;
    msg.pose.position.y = snapshot.observation.position.y;
    msg.pose.position.z = snapshot.observation.position.z;
    assign_orientation(msg.pose.orientation, snapshot.observation.orientation);
    msg.range = static_cast<float>(std::sqrt(
      (snapshot.observation.position.x * snapshot.observation.position.x)
      + (snapshot.observation.position.y * snapshot.observation.position.y)
      + (snapshot.observation.position.z * snapshot.observation.position.z)));
  }
  msg.has_velocity = snapshot.has_velocity && msg.has_pose;
  if (snapshot.has_velocity) {
    msg.velocity.x = snapshot.velocity[0];
    msg.velocity.y = snapshot.velocity[1];
    msg.velocity.z = snapshot.velocity[2];
  }
  target_publisher_->publish(std::move(msg));
}

void ArmorTrackerNode::log_stage_timing()
{
  const double n = static_cast<double>(stage_log_interval_);
  double avg_fps = 0.0;
  const auto now = std::chrono::steady_clock::now();
  if (have_last_stage_log_wall_time_) {
    const double elapsed_s = std::chrono::duration<double>(now - last_stage_log_wall_time_).count();
    if (elapsed_s > 0.0) {
      avg_fps = n / elapsed_s;
    }
  }
  last_stage_log_wall_time_ = now;
  have_last_stage_log_wall_time_ = true;
  RCLCPP_INFO(this->get_logger(),
    "Tracker timing mean over %d frames: fps=%.1f tracking=%.2fms publish=%.2fms",
    stage_log_interval_, avg_fps, stage_totals_.tracking_ms / n, stage_totals_.publish_ms / n);
  stage_totals_ = StageTotals{};
}

}  // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::ArmorTrackerNode)
