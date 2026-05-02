#include "yolo_cpp/armor_processor_node.hpp"

#include <geometry_msgs/msg/point_stamped.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <algorithm>
#include <cmath>
#include <functional>

namespace yolo_cpp
{

namespace
{

constexpr double kPi = 3.14159265358979323846;

double point_range(const geometry_msgs::msg::Point & point)
{
  return std::sqrt((point.x * point.x) + (point.y * point.y) + (point.z * point.z));
}

double normalize_angle(double value)
{
  while (value > kPi) {
    value -= 2.0 * kPi;
  }
  while (value < -kPi) {
    value += 2.0 * kPi;
  }
  return value;
}

}  // namespace

ArmorProcessorNode::ArmorProcessorNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("armor_processor_node", options)
{
  this->declare_parameter<std::string>("target_frame", target_frame_);
  this->declare_parameter<int>("processor_lost_threshold", processor_lost_threshold_);
  this->declare_parameter<bool>("spin_observer.allow", allow_spin_observer_);
  this->declare_parameter<double>("spin_observer.max_jump_angle", spin_max_jump_angle_);
  this->declare_parameter<double>("spin_observer.max_jump_period", spin_max_jump_period_);
  this->declare_parameter<double>(
    "spin_observer.allow_following_range", spin_allow_following_range_);
  this->declare_parameter<double>("filter.position_alpha", position_alpha_);
  this->declare_parameter<double>("filter.velocity_alpha", velocity_alpha_);

  target_frame_ = this->get_parameter("target_frame").as_string();
  processor_lost_threshold_ = this->get_parameter("processor_lost_threshold").as_int();
  allow_spin_observer_ = this->get_parameter("spin_observer.allow").as_bool();
  spin_max_jump_angle_ = this->get_parameter("spin_observer.max_jump_angle").as_double();
  spin_max_jump_period_ = this->get_parameter("spin_observer.max_jump_period").as_double();
  spin_allow_following_range_ =
    this->get_parameter("spin_observer.allow_following_range").as_double();
  position_alpha_ = std::clamp(this->get_parameter("filter.position_alpha").as_double(), 0.0, 1.0);
  velocity_alpha_ = std::clamp(this->get_parameter("filter.velocity_alpha").as_double(), 0.0, 1.0);

  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  tracks_subscription_ = this->create_subscription<cv_auto_aim_interfaces::msg::TrackedArmors>(
    "tracker/tracked_armors", rclcpp::SensorDataQoS(),
    std::bind(&ArmorProcessorNode::handle_tracks, this, std::placeholders::_1));
  target_publisher_ =
    this->create_publisher<cv_auto_aim_interfaces::msg::Target>("processor/target", 10);
  spin_info_publisher_ =
    this->create_publisher<cv_auto_aim_interfaces::msg::SpinInfo>("debug/spin_info", 10);
  marker_publisher_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("processor/marker", 10);

  spin_.last_jump_time = this->get_clock()->now();
  RCLCPP_INFO(this->get_logger(), "Initialized ArmorProcessorNode target_frame=%s", target_frame_.c_str());
}

void ArmorProcessorNode::handle_tracks(
  const cv_auto_aim_interfaces::msg::TrackedArmors::SharedPtr msg)
{
  rclcpp::Time stamp(msg->header.stamp);
  if (stamp.nanoseconds() == 0) {
    stamp = this->get_clock()->now();
  }

  const cv_auto_aim_interfaces::msg::TrackedArmor * selected = nullptr;
  if (msg->selected_track_id >= 0) {
    for (const auto & track : msg->tracks) {
      if (track.track_id == msg->selected_track_id) {
        selected = &track;
        break;
      }
    }
  }

  cv_auto_aim_interfaces::msg::Target target;
  target.header.stamp = stamp;
  target.header.frame_id = target_frame_;
  target.source_id = msg->source_id;
  target.track_id = selected != nullptr ? selected->track_id : -1;
  target.state = selected != nullptr ?
    selected->state :
    cv_auto_aim_interfaces::msg::Target::DETECTING;
  target.suggest_fire = false;

  bool valid_measurement = false;
  geometry_msgs::msg::Point measurement;
  if (selected != nullptr && selected->has_pose) {
    valid_measurement = transform_point(msg->header, selected->pose_camera.position, measurement);
  }

  if (valid_measurement) {
    update_filter(selected->track_id, stamp, measurement);
  } else {
    predict_without_measurement(stamp);
  }

  const bool stable_track = selected != nullptr &&
    (selected->state == cv_auto_aim_interfaces::msg::TrackedArmor::TRACKING ||
    selected->state == cv_auto_aim_interfaces::msg::TrackedArmor::COASTING);
  const bool can_publish_filtered =
    filter_.initialized && filter_.lost_frames <= processor_lost_threshold_;

  target.has_target = stable_track;
  target.tracking = target.has_target;
  target.suggest_fire = target.has_target;
  target.is_coasting = selected != nullptr &&
    selected->state == cv_auto_aim_interfaces::msg::TrackedArmor::COASTING;

  if (selected != nullptr) {
    target.bbox_xyxy = selected->armor.bbox_xyxy;
    target.center_x = selected->armor.center_x;
    target.center_y = selected->armor.center_y;
    target.predicted_center_x = selected->predicted_center_x;
    target.predicted_center_y = selected->predicted_center_y;
    target.confidence = selected->armor.confidence;
    target.class_id = selected->armor.class_id;
    target.is_enemy = selected->armor.is_enemy;
    target.has_obb_angle = selected->armor.has_obb_angle;
    target.obb_angle = selected->armor.obb_angle;
    target.image_width = selected->armor.image_width;
    target.image_height = selected->armor.image_height;
  }

  if (target.has_target && can_publish_filtered) {
    target.has_pose = true;
    target.pose.position = filter_.position;
    target.pose.orientation.w = 1.0;
    target.has_velocity = true;
    target.velocity = filter_.velocity;
    target.range = static_cast<float>(point_range(filter_.position));
  }

  apply_spin_observer(target);
  if (target.has_pose) {
    target.range = static_cast<float>(point_range(target.pose.position));
  }
  target_publisher_->publish(target);
  publish_markers(target);
}

bool ArmorProcessorNode::transform_point(
  const std_msgs::msg::Header & header, const geometry_msgs::msg::Point & point,
  geometry_msgs::msg::Point & transformed)
{
  geometry_msgs::msg::PointStamped source;
  source.header = header;
  source.point = point;
  try {
    transformed = tf2_buffer_->transform(source, target_frame_).point;
    return true;
  } catch (const tf2::TransformException & exc) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000, "Target transform unavailable: %s", exc.what());
    return false;
  }
}

void ArmorProcessorNode::update_filter(
  int track_id, const rclcpp::Time & stamp, const geometry_msgs::msg::Point & measurement)
{
  double dt = filter_.initialized ? (stamp - filter_.stamp).seconds() : 1.0 / 60.0;
  if (dt <= 0.0) {
    dt = 1e-3;
  }

  if (!filter_.initialized || filter_.track_id != track_id) {
    filter_.track_id = track_id;
    filter_.initialized = true;
    filter_.stamp = stamp;
    filter_.position = measurement;
    filter_.velocity = geometry_msgs::msg::Vector3();
    filter_.lost_frames = 0;
    return;
  }

  geometry_msgs::msg::Point predicted;
  predicted.x = filter_.position.x + (filter_.velocity.x * dt);
  predicted.y = filter_.position.y + (filter_.velocity.y * dt);
  predicted.z = filter_.position.z + (filter_.velocity.z * dt);
  geometry_msgs::msg::Vector3 measured_velocity;
  measured_velocity.x = (measurement.x - filter_.position.x) / dt;
  measured_velocity.y = (measurement.y - filter_.position.y) / dt;
  measured_velocity.z = (measurement.z - filter_.position.z) / dt;

  filter_.position.x = ((1.0 - position_alpha_) * predicted.x) + (position_alpha_ * measurement.x);
  filter_.position.y = ((1.0 - position_alpha_) * predicted.y) + (position_alpha_ * measurement.y);
  filter_.position.z = ((1.0 - position_alpha_) * predicted.z) + (position_alpha_ * measurement.z);
  filter_.velocity.x =
    ((1.0 - velocity_alpha_) * filter_.velocity.x) + (velocity_alpha_ * measured_velocity.x);
  filter_.velocity.y =
    ((1.0 - velocity_alpha_) * filter_.velocity.y) + (velocity_alpha_ * measured_velocity.y);
  filter_.velocity.z =
    ((1.0 - velocity_alpha_) * filter_.velocity.z) + (velocity_alpha_ * measured_velocity.z);
  filter_.stamp = stamp;
  filter_.lost_frames = 0;
}

void ArmorProcessorNode::predict_without_measurement(const rclcpp::Time & stamp)
{
  if (!filter_.initialized) {
    return;
  }
  double dt = (stamp - filter_.stamp).seconds();
  if (dt <= 0.0) {
    dt = 1e-3;
  }
  filter_.position.x += filter_.velocity.x * dt;
  filter_.position.y += filter_.velocity.y * dt;
  filter_.position.z += filter_.velocity.z * dt;
  filter_.stamp = stamp;
  ++filter_.lost_frames;
}

void ArmorProcessorNode::apply_spin_observer(cv_auto_aim_interfaces::msg::Target & target)
{
  cv_auto_aim_interfaces::msg::SpinInfo info;
  info.header = target.header;

  if (!allow_spin_observer_) {
    info.suggest_fire = target.suggest_fire;
    spin_info_publisher_->publish(info);
    return;
  }

  const rclcpp::Time current_time(target.header.stamp);
  const double time_after_jump = (current_time - spin_.last_jump_time).seconds();
  if (time_after_jump > spin_max_jump_period_) {
    spin_.target_spinning = false;
    spin_.jump_count = 0;
  }

  double yaw_diff = 0.0;
  if (target.has_target && target.has_pose) {
    const double current_yaw = std::atan2(target.pose.position.x, target.pose.position.z);
    yaw_diff = normalize_angle(current_yaw - spin_.last_yaw);
    if (std::abs(yaw_diff) > spin_max_jump_angle_) {
      ++spin_.jump_count;
      if (spin_.jump_count > 1 && std::signbit(yaw_diff) == std::signbit(spin_.last_jump_yaw_diff)) {
        spin_.target_spinning = true;
        spin_.jump_period = std::max(1e-3, time_after_jump);
      }
      spin_.last_jump_time = current_time;
      spin_.last_jump_position = target.pose.position;
      spin_.last_jump_yaw_diff = yaw_diff;
    }
    spin_.last_yaw = current_yaw;
  }

  if (spin_.target_spinning && spin_.jump_period > 0.0) {
    if ((time_after_jump / spin_.jump_period) < spin_allow_following_range_) {
      target.suggest_fire = target.has_target;
    } else {
      target.pose.position = spin_.last_jump_position;
      target.velocity = geometry_msgs::msg::Vector3();
      target.suggest_fire = false;
    }
  } else {
    target.suggest_fire = target.has_target;
  }

  info.target_spinning = spin_.target_spinning;
  info.suggest_fire = target.suggest_fire;
  info.jump_count = spin_.jump_count;
  info.yaw_diff = yaw_diff;
  info.jump_period = spin_.jump_period;
  info.time_after_jumping = time_after_jump;
  spin_info_publisher_->publish(info);
}

void ArmorProcessorNode::publish_markers(const cv_auto_aim_interfaces::msg::Target & target)
{
  visualization_msgs::msg::Marker position;
  visualization_msgs::msg::Marker velocity;
  position.header = target.header;
  velocity.header = target.header;
  position.ns = "target_position";
  velocity.ns = "target_velocity";
  position.id = 0;
  velocity.id = 1;
  position.type = visualization_msgs::msg::Marker::SPHERE;
  velocity.type = visualization_msgs::msg::Marker::ARROW;
  position.scale.x = 0.08;
  position.scale.y = 0.08;
  position.scale.z = 0.08;
  velocity.scale.x = 0.02;
  velocity.scale.y = 0.04;
  position.color.a = 1.0;
  velocity.color.a = 1.0;
  position.color.g = target.suggest_fire ? 1.0 : 0.0;
  position.color.r = target.suggest_fire ? 0.0 : 1.0;
  velocity.color.b = 1.0;

  if (target.has_target && target.has_pose) {
    position.action = visualization_msgs::msg::Marker::ADD;
    velocity.action = visualization_msgs::msg::Marker::ADD;
    position.pose.position = target.pose.position;
    position.pose.orientation.w = 1.0;
    velocity.points.push_back(target.pose.position);
    geometry_msgs::msg::Point end = target.pose.position;
    end.x += target.velocity.x;
    end.y += target.velocity.y;
    end.z += target.velocity.z;
    velocity.points.push_back(end);
  } else {
    position.action = visualization_msgs::msg::Marker::DELETE;
    velocity.action = visualization_msgs::msg::Marker::DELETE;
  }

  visualization_msgs::msg::MarkerArray markers;
  markers.markers.push_back(position);
  markers.markers.push_back(velocity);
  marker_publisher_->publish(markers);
}

}  // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::ArmorProcessorNode)
