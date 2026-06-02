#ifndef YOLO_CPP__ARMOR_PROCESSOR_NODE_HPP_
#define YOLO_CPP__ARMOR_PROCESSOR_NODE_HPP_

#include "cv_auto_aim_interfaces/msg/spin_info.hpp"
#include "cv_auto_aim_interfaces/msg/target.hpp"
#include "cv_auto_aim_interfaces/msg/tracked_armors.hpp"
#include "yolo_cpp/visibility_control.hpp"

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include <optional>
#include <string>

namespace yolo_cpp
{

/// Filters selected tracks in 3D space and observes spin behavior before solving aim.
class YOLO_CPP_PUBLIC ArmorProcessorNode : public rclcpp::Node
{
public:
  /// Create the processor, TF listener, and publishers.
  explicit ArmorProcessorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ArmorProcessorNode() override = default;

private:
  struct FilterState
  {
    int track_id{-1};
    bool initialized{false};
    rclcpp::Time stamp;
    geometry_msgs::msg::Point position;
    geometry_msgs::msg::Vector3 velocity;
    int lost_frames{0};
  };

  struct SpinState
  {
    bool target_spinning{false};
    int jump_count{0};
    double last_yaw{0.0};
    double last_jump_yaw_diff{0.0};
    double jump_period{0.0};
    rclcpp::Time last_jump_time;
    geometry_msgs::msg::Point last_jump_position;
  };

  /// Select and filter the best tracked armor from one tracker frame.
  void handle_tracks(const cv_auto_aim_interfaces::msg::TrackedArmors::SharedPtr msg);

  /// Transform a camera-frame point into the configured target frame.
  bool transform_point(
    const std_msgs::msg::Header & header, const geometry_msgs::msg::Point & point,
    geometry_msgs::msg::Point & transformed);
  /// Update the position/velocity filter with a measured target point.
  void update_filter(
    int track_id,
    const rclcpp::Time & stamp,
    const geometry_msgs::msg::Point & measurement);

  /// Advance filter state when the selected target is coasting.
  void predict_without_measurement(const rclcpp::Time & stamp);

  /// Annotate the target with spin-following and fire-suggestion state.
  void apply_spin_observer(cv_auto_aim_interfaces::msg::Target & target);

  /// Publish visualization markers for the selected processed target.
  void publish_markers(const cv_auto_aim_interfaces::msg::Target & target);

  std::string target_frame_{"shooter_link"};
  int processor_lost_threshold_{5};
  bool allow_spin_observer_{true};
  double spin_max_jump_angle_{0.2};
  double spin_max_jump_period_{0.8};
  double spin_allow_following_range_{0.3};
  double position_alpha_{0.7};
  double velocity_alpha_{0.5};

  rclcpp::Subscription<cv_auto_aim_interfaces::msg::TrackedArmors>::SharedPtr tracks_subscription_;
  rclcpp::Publisher<cv_auto_aim_interfaces::msg::Target>::SharedPtr target_publisher_;
  rclcpp::Publisher<cv_auto_aim_interfaces::msg::SpinInfo>::SharedPtr spin_info_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;

  FilterState filter_;
  SpinState spin_;
};

}  // namespace yolo_cpp

#endif
