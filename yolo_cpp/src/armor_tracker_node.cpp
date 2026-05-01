#include "yolo_cpp/armor_tracker_node.hpp"

#include "cv_auto_aim_interfaces/msg/armor.hpp"
#include "cv_auto_aim_interfaces/msg/tracked_armor.hpp"
#include "yolo_cpp/utils.hpp"

#include <rclcpp_components/register_node_macro.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace yolo_cpp
{

namespace
{

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

cv_auto_aim_interfaces::msg::Armor make_armor_message(const TrackObservation & observation)
{
  cv_auto_aim_interfaces::msg::Armor armor;
  armor.bbox_xyxy = {
    observation.bbox.x,
    observation.bbox.y,
    observation.bbox.x + observation.bbox.width,
    observation.bbox.y + observation.bbox.height};
  armor.center_x = observation.center.x;
  armor.center_y = observation.center.y;
  armor.confidence = observation.confidence;
  armor.class_id = observation.class_id;
  armor.is_enemy = observation.is_enemy;
  armor.has_obb_angle = observation.obb_angle.has_value();
  armor.obb_angle = observation.obb_angle.value_or(0.0F);
  armor.image_width = observation.image_width;
  armor.image_height = observation.image_height;
  return armor;
}

TrackObservation make_track_observation(const cv_auto_aim_interfaces::msg::Armor & armor)
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
  return observation;
}

}  // namespace

ArmorTrackerNode::ArmorTrackerNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("armor_tracker_node", options)
{
  this->declare_parameter<bool>("use_robot_id_targeting", false);
  this->declare_parameter<int>("enemy_color_red_value", 0);
  this->declare_parameter<int>("enemy_color_blue_value", 1);
  this->declare_parameter<int>("track_min_hits", 2);
  this->declare_parameter<int>("track_max_coast_frames", 3);
  this->declare_parameter<int>("track_drop_frames", 8);
  this->declare_parameter<double>("association_max_center_distance_ratio", 0.20);
  this->declare_parameter<double>("association_min_iou", 0.10);
  this->declare_parameter<double>("target_switch_min_score_delta", 0.15);
  this->declare_parameter<int>("target_lock_hysteresis_frames", 2);
  this->declare_parameter<int>("stage_log_interval", 300);

  use_robot_id_targeting_ = this->get_parameter("use_robot_id_targeting").as_bool();
  enemy_color_red_value_ = this->get_parameter("enemy_color_red_value").as_int();
  enemy_color_blue_value_ = this->get_parameter("enemy_color_blue_value").as_int();
  stage_log_interval_ = std::max(
    1, static_cast<int>(this->get_parameter("stage_log_interval").as_int()));

  TrackConfig config;
  config.track_min_hits = this->get_parameter("track_min_hits").as_int();
  config.track_max_coast_frames = this->get_parameter("track_max_coast_frames").as_int();
  config.track_drop_frames = this->get_parameter("track_drop_frames").as_int();
  config.association_max_center_distance_ratio =
    this->get_parameter("association_max_center_distance_ratio").as_double();
  config.association_min_iou = this->get_parameter("association_min_iou").as_double();
  config.target_switch_min_score_delta =
    this->get_parameter("target_switch_min_score_delta").as_double();
  config.target_lock_hysteresis_frames =
    this->get_parameter("target_lock_hysteresis_frames").as_int();
  track_manager_.set_config(config);

  armors_subscription_ = this->create_subscription<cv_auto_aim_interfaces::msg::Armors>(
    "detector/armors", 10,
    std::bind(&ArmorTrackerNode::armors_callback, this, std::placeholders::_1));
  robot_id_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
    "robot_id", 10, std::bind(&ArmorTrackerNode::id_listener_callback, this, std::placeholders::_1));
  attitude_flags_subscription_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
    "attitude_flags", 10,
    std::bind(&ArmorTrackerNode::handle_attitude_flags, this, std::placeholders::_1));
  tracked_armors_publisher_ = this->create_publisher<cv_auto_aim_interfaces::msg::TrackedArmors>(
    "tracker/tracked_armors", 10);
  target_publisher_ =
    this->create_publisher<cv_auto_aim_interfaces::msg::Target>("tracker/target", 10);

  RCLCPP_INFO(this->get_logger(), "Initialized ArmorTrackerNode");
}

void ArmorTrackerNode::armors_callback(const cv_auto_aim_interfaces::msg::Armors::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);

  rclcpp::Time stamp(msg->header.stamp);
  if (stamp.nanoseconds() == 0) {
    stamp = this->get_clock()->now();
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
  for (const auto & armor : msg->armors) {
    observations.push_back(make_track_observation(armor));
  }

  const auto tracking_start = std::chrono::steady_clock::now();
  const auto result = track_manager_.update(observations, dt);
  const auto tracking_done = std::chrono::steady_clock::now();
  const auto publish_start = std::chrono::steady_clock::now();
  publish_tracked_armors(*msg, result);
  publish_target(*msg, result.selected_target);
  const auto publish_done = std::chrono::steady_clock::now();

  ++frame_count_;
  stage_totals_.tracking_ms += std::chrono::duration<double, std::milli>(
    tracking_done - tracking_start).count();
  stage_totals_.publish_ms += std::chrono::duration<double, std::milli>(
    publish_done - publish_start).count();

  if (frame_count_ % stage_log_interval_ == 0) {
    log_stage_timing();
  }
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
  const cv_auto_aim_interfaces::msg::Armors & source_msg,
  const TrackingFrameResult & result)
{
  cv_auto_aim_interfaces::msg::TrackedArmors msg;
  msg.header = source_msg.header;
  msg.source_id = source_msg.source_id;
  msg.selected_track_id = result.selected_track_id.value_or(-1);
  msg.tracks.reserve(result.tracks.size());
  for (const auto & track : result.tracks) {
    cv_auto_aim_interfaces::msg::TrackedArmor tracked;
    tracked.armor = make_armor_message(track.observation);
    tracked.track_id = track.track_id;
    tracked.state = to_msg_state(track.state);
    tracked.age = track.age;
    tracked.hit_streak = track.hit_streak;
    tracked.lost_frames = track.lost_frames;
    tracked.predicted_center_x = track.predicted_center.x;
    tracked.predicted_center_y = track.predicted_center.y;
    msg.tracks.push_back(std::move(tracked));
  }
  tracked_armors_publisher_->publish(std::move(msg));
}

void ArmorTrackerNode::publish_target(
  const cv_auto_aim_interfaces::msg::Armors & source_msg,
  const std::optional<TrackSnapshot> & target)
{
  cv_auto_aim_interfaces::msg::Target msg;
  msg.header = source_msg.header;
  msg.source_id = source_msg.source_id;
  if (!target.has_value()) {
    msg.has_target = false;
    msg.is_coasting = false;
    msg.track_id = -1;
    msg.state = cv_auto_aim_interfaces::msg::Target::DETECTING;
    msg.bbox_xyxy = {0, 0, 0, 0};
    if (!source_msg.armors.empty()) {
      msg.image_width = source_msg.armors.front().image_width;
      msg.image_height = source_msg.armors.front().image_height;
    }
    target_publisher_->publish(std::move(msg));
    return;
  }

  const auto & snapshot = *target;
  msg.has_target = true;
  msg.is_coasting = snapshot.state == TrackState::kCoasting;
  msg.track_id = snapshot.track_id;
  msg.bbox_xyxy = {
    snapshot.observation.bbox.x,
    snapshot.observation.bbox.y,
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
  RCLCPP_INFO(
    this->get_logger(),
    "Tracker timing mean over %d frames: fps=%.1f tracking=%.2fms publish=%.2fms",
    stage_log_interval_,
    avg_fps,
    stage_totals_.tracking_ms / n,
    stage_totals_.publish_ms / n);
  stage_totals_ = StageTotals{};
}

}  // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::ArmorTrackerNode)
