#include "yolo_cpp/rm_handoff_tracker_node.hpp"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/create_timer_ros.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <utility>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp_components/register_node_macro.hpp"

namespace yolo_cpp {

namespace {

constexpr double kDefaultDt = 1.0 / 60.0;

} // namespace

RmHandoffTrackerNode::RmHandoffTrackerNode(const rclcpp::NodeOptions &options)
    : rclcpp::Node("rm_handoff_tracker", options) {
  max_armor_distance_ = this->declare_parameter("max_armor_distance", 10.0);
  target_frame_ = this->declare_parameter("target_frame", "gimbal_odom");

  const double max_match_distance =
      this->declare_parameter("tracker.max_match_distance", 0.15);
  const double max_match_yaw_diff =
      this->declare_parameter("tracker.max_match_yaw_diff", 1.0);
  tracker_ = std::make_unique<RmHandoffTracker>(max_match_distance,
                                                max_match_yaw_diff);
  tracker_->tracking_thres =
      this->declare_parameter("tracker.tracking_thres", 5);
  lost_time_thres_ = this->declare_parameter("tracker.lost_time_thres", 0.3);

  s2qxyz_ = this->declare_parameter("ekf.sigma2_q_xyz", 20.0);
  s2qyaw_ = this->declare_parameter("ekf.sigma2_q_yaw", 100.0);
  s2qr_ = this->declare_parameter("ekf.sigma2_q_r", 800.0);
  r_xyz_factor_ = this->declare_parameter("ekf.r_xyz_factor", 0.05);
  r_yaw_ = this->declare_parameter("ekf.r_yaw", 0.02);
  configure_tracker_ekf(kDefaultDt);

  tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
      this->get_node_base_interface(), this->get_node_timers_interface());
  tf2_buffer_->setCreateTimerInterface(timer_interface);
  tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

  reset_service_ = this->create_service<std_srvs::srv::Trigger>(
      "/tracker/reset",
      [this](const std_srvs::srv::Trigger::Request::SharedPtr,
             std_srvs::srv::Trigger::Response::SharedPtr response) {
        tracker_->tracker_state = RmHandoffTracker::kLost;
        response->success = true;
        response->message = "RM handoff tracker reset";
        RCLCPP_INFO(this->get_logger(), "RM handoff tracker reset");
      });

  armors_subscription_ =
      this->create_subscription<auto_aim_interfaces::msg::Armors>(
          "/detector/armors", rclcpp::SensorDataQoS(),
          std::bind(&RmHandoffTrackerNode::armors_callback, this,
                    std::placeholders::_1));
  target_publisher_ = this->create_publisher<auto_aim_interfaces::msg::Target>(
      "/tracker/target", rclcpp::SensorDataQoS());
  info_publisher_ =
      this->create_publisher<auto_aim_interfaces::msg::TrackerInfo>(
          "/tracker/info", 10);
  marker_publisher_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/tracker/marker", 10);

  position_marker_.ns = "position";
  position_marker_.type = visualization_msgs::msg::Marker::SPHERE;
  position_marker_.scale.x = position_marker_.scale.y =
      position_marker_.scale.z = 0.1;
  position_marker_.color.a = 1.0;
  position_marker_.color.g = 1.0;

  linear_v_marker_.ns = "linear_v";
  linear_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  linear_v_marker_.scale.x = 0.03;
  linear_v_marker_.scale.y = 0.05;
  linear_v_marker_.color.a = 1.0;
  linear_v_marker_.color.r = 1.0;
  linear_v_marker_.color.g = 1.0;

  angular_v_marker_.ns = "angular_v";
  angular_v_marker_.type = visualization_msgs::msg::Marker::ARROW;
  angular_v_marker_.scale.x = 0.03;
  angular_v_marker_.scale.y = 0.05;
  angular_v_marker_.color.a = 1.0;
  angular_v_marker_.color.b = 1.0;
  angular_v_marker_.color.g = 1.0;

  armor_marker_.ns = "armors";
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.03;
  armor_marker_.scale.z = 0.125;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.r = 1.0;

  RCLCPP_INFO(this->get_logger(), "Initialized RM handoff tracker");
}

void RmHandoffTrackerNode::armors_callback(
    const auto_aim_interfaces::msg::Armors::SharedPtr msg) {
  auto transformed_msg = *msg;
  for (auto &armor : transformed_msg.armors) {
    geometry_msgs::msg::PoseStamped source_pose;
    source_pose.header = transformed_msg.header;
    source_pose.pose = armor.pose;
    try {
      armor.pose = tf2_buffer_->transform(source_pose, target_frame_).pose;
    } catch (const tf2::TransformException &exc) {
      RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "Armor transform unavailable: %s", exc.what());
      return;
    }
  }

  transformed_msg.armors.erase(
      std::remove_if(
          transformed_msg.armors.begin(), transformed_msg.armors.end(),
          [this](const auto_aim_interfaces::msg::Armor &armor) {
            return std::abs(armor.pose.position.z) > 1.2 ||
                   std::hypot(armor.pose.position.x, armor.pose.position.y) >
                       max_armor_distance_;
          }),
      transformed_msg.armors.end());

  rclcpp::Time stamp(transformed_msg.header.stamp);
  if (stamp.nanoseconds() == 0) {
    stamp = this->get_clock()->now();
  }

  auto_aim_interfaces::msg::Target target_msg;
  target_msg.header.stamp = stamp;
  target_msg.header.frame_id = target_frame_;

  if (tracker_->tracker_state == RmHandoffTracker::kLost) {
    tracker_->init(transformed_msg);
    target_msg.tracking = false;
  } else {
    double dt = last_dt_;
    if (have_last_time_) {
      dt = (stamp - last_time_).seconds();
      if (dt <= 0.0) {
        dt = 0.001;
      }
    }
    last_dt_ = dt;
    tracker_->lost_thres = std::max(1, static_cast<int>(lost_time_thres_ / dt));
    tracker_->set_dt(dt);
    tracker_->update(transformed_msg);

    auto_aim_interfaces::msg::TrackerInfo info_msg;
    info_msg.position_diff = tracker_->info_position_diff;
    info_msg.yaw_diff = tracker_->info_yaw_diff;
    info_msg.position.x = tracker_->measurement(0);
    info_msg.position.y = tracker_->measurement(1);
    info_msg.position.z = tracker_->measurement(2);
    info_msg.yaw = tracker_->measurement(3);
    info_publisher_->publish(info_msg);

    publish_target(stamp, target_msg);
  }

  last_time_ = stamp;
  have_last_time_ = true;
  target_publisher_->publish(target_msg);
  publish_markers(target_msg);
}

void RmHandoffTrackerNode::publish_target(
    const rclcpp::Time &stamp, auto_aim_interfaces::msg::Target &target_msg) {
  target_msg.header.stamp = stamp;
  target_msg.header.frame_id = target_frame_;

  if (tracker_->tracker_state == RmHandoffTracker::kDetecting) {
    target_msg.tracking = false;
    return;
  }

  if (tracker_->tracker_state != RmHandoffTracker::kTracking &&
      tracker_->tracker_state != RmHandoffTracker::kTempLost) {
    target_msg.tracking = false;
    return;
  }

  const auto &state = tracker_->target_state;
  target_msg.tracking = true;
  target_msg.id = tracker_->tracked_id;
  target_msg.armors_num = static_cast<int>(tracker_->tracked_armors_num);
  target_msg.position.x = state(0);
  target_msg.velocity.x = state(1);
  target_msg.position.y = state(2);
  target_msg.velocity.y = state(3);
  target_msg.position.z = state(4);
  target_msg.velocity.z = state(5);
  target_msg.yaw = state(6);
  target_msg.v_yaw = state(7);
  target_msg.radius_1 = state(8);
  target_msg.radius_2 = tracker_->another_r;
  target_msg.dz = tracker_->dz;
}

void RmHandoffTrackerNode::publish_markers(
    const auto_aim_interfaces::msg::Target &target_msg) {
  position_marker_.header = target_msg.header;
  linear_v_marker_.header = target_msg.header;
  angular_v_marker_.header = target_msg.header;
  armor_marker_.header = target_msg.header;

  visualization_msgs::msg::MarkerArray marker_array;
  if (target_msg.tracking) {
    const double yaw = target_msg.yaw;
    const double r1 = target_msg.radius_1;
    const double r2 = target_msg.radius_2;
    const double xc = target_msg.position.x;
    const double yc = target_msg.position.y;
    const double za = target_msg.position.z;
    const double vx = target_msg.velocity.x;
    const double vy = target_msg.velocity.y;
    const double vz = target_msg.velocity.z;
    const double dz = target_msg.dz;

    position_marker_.action = visualization_msgs::msg::Marker::ADD;
    position_marker_.pose.position.x = xc;
    position_marker_.pose.position.y = yc;
    position_marker_.pose.position.z = za + dz / 2.0;

    linear_v_marker_.action = visualization_msgs::msg::Marker::ADD;
    linear_v_marker_.points.clear();
    linear_v_marker_.points.emplace_back(position_marker_.pose.position);
    geometry_msgs::msg::Point arrow_end = position_marker_.pose.position;
    arrow_end.x += vx;
    arrow_end.y += vy;
    arrow_end.z += vz;
    linear_v_marker_.points.emplace_back(arrow_end);

    angular_v_marker_.action = visualization_msgs::msg::Marker::ADD;
    angular_v_marker_.points.clear();
    angular_v_marker_.points.emplace_back(position_marker_.pose.position);
    arrow_end = position_marker_.pose.position;
    arrow_end.z += target_msg.v_yaw / M_PI;
    angular_v_marker_.points.emplace_back(arrow_end);

    armor_marker_.action = visualization_msgs::msg::Marker::ADD;
    armor_marker_.scale.y =
        RmHandoffTracker::armor_type(tracker_->tracked_armor) == "small" ? 0.135
                                                                         : 0.23;
    bool is_current_pair = true;
    geometry_msgs::msg::Point armor_position;
    for (int i = 0; i < target_msg.armors_num; ++i) {
      const double armor_yaw = yaw + i * (2.0 * M_PI / target_msg.armors_num);
      double radius = r1;
      if (target_msg.armors_num == 4) {
        radius = is_current_pair ? r1 : r2;
        armor_position.z = za + (is_current_pair ? 0.0 : dz);
        is_current_pair = !is_current_pair;
      } else {
        armor_position.z = za;
      }
      armor_position.x = xc - radius * std::cos(armor_yaw);
      armor_position.y = yc - radius * std::sin(armor_yaw);

      armor_marker_.id = i;
      armor_marker_.pose.position = armor_position;
      tf2::Quaternion q;
      q.setRPY(0.0, target_msg.id == "outpost" ? -0.26 : 0.26, armor_yaw);
      armor_marker_.pose.orientation = tf2::toMsg(q);
      marker_array.markers.emplace_back(armor_marker_);
    }
  } else {
    position_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
    linear_v_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
    angular_v_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
    armor_marker_.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.emplace_back(armor_marker_);
  }

  marker_array.markers.emplace_back(position_marker_);
  marker_array.markers.emplace_back(linear_v_marker_);
  marker_array.markers.emplace_back(angular_v_marker_);
  marker_publisher_->publish(marker_array);
}

void RmHandoffTrackerNode::configure_tracker_ekf(double dt) {
  tracker_->configure_ekf(dt, s2qxyz_, s2qyaw_, s2qr_, r_xyz_factor_, r_yaw_);
}

} // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::RmHandoffTrackerNode)
