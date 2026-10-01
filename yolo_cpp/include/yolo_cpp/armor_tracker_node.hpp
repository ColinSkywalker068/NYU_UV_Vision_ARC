#ifndef YOLO_CPP__ARMOR_TRACKER_NODE_HPP_
#define YOLO_CPP__ARMOR_TRACKER_NODE_HPP_

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include "cv_auto_aim_interfaces/msg/armors.hpp"
#include "cv_auto_aim_interfaces/msg/target.hpp"
#include "cv_auto_aim_interfaces/msg/tracked_armors.hpp"
#include "cv_auto_aim_interfaces/msg/gimbal_attitude.hpp"
#include "yolo_cpp/aim_geometry.hpp"
#include "yolo_cpp/tracker.hpp"
#include "yolo_cpp/utils.hpp"
#include "yolo_cpp/visibility_control.hpp"

namespace yolo_cpp
{

/// Converts detector armor candidates into stable tracks and a selected target.
class YOLO_CPP_PUBLIC ArmorTrackerNode : public rclcpp::Node
{
public:
  /// Create subscriptions, publishers, and configure the track manager.
  explicit ArmorTrackerNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~ArmorTrackerNode() override = default;

private:
  struct StageTotals
  {
    double tracking_ms{0.0};
    double publish_ms{0.0};
  };

  /// Process one detector output frame.
  void armors_callback(const cv_auto_aim_interfaces::msg::Armors::SharedPtr msg);

  /// Update enemy-color targeting from robot id.
  void id_listener_callback(const std_msgs::msg::Int32::SharedPtr msg);

  /// Update controller-sourced color/targeting flags.
  void handle_attitude_flags(const std_msgs::msg::Int32MultiArray::SharedPtr msg);
  void handle_attitude(const cv_auto_aim_interfaces::msg::GimbalAttitude::SharedPtr msg);

  /// Publish every active track for downstream processor/debug consumers.
  void publish_tracked_armors(
    const cv_auto_aim_interfaces::msg::Armors& source_msg, const TrackingFrameResult& result);
  /// Publish the currently selected 2D target, or an empty target message.
  void publish_target(const cv_auto_aim_interfaces::msg::Armors& source_msg,
    const std::optional<TrackSnapshot>& target);
  /// Periodically log tracking and publish timings.
  void log_stage_timing();

  rclcpp::Subscription<cv_auto_aim_interfaces::msg::Armors>::SharedPtr armors_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr robot_id_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr attitude_flags_subscription_;
  rclcpp::Subscription<cv_auto_aim_interfaces::msg::GimbalAttitude>::SharedPtr attitude_subscription_;
  rclcpp::Publisher<cv_auto_aim_interfaces::msg::TrackedArmors>::SharedPtr
    tracked_armors_publisher_;
  rclcpp::Publisher<cv_auto_aim_interfaces::msg::Target>::SharedPtr target_publisher_;

  TrackManager track_manager_;
  rclcpp::Time last_frame_time_;
  bool have_last_frame_time_{false};
  std::chrono::steady_clock::time_point last_stage_log_wall_time_{};
  bool have_last_stage_log_wall_time_{false};
  std::mutex state_mutex_;
  AttitudeHistory attitude_history_;
  std::string tracking_frame_{"gimbal_inertial"};
  double attitude_edge_tolerance_s_{0.03};
  double attitude_max_gap_s_{0.05};
  double max_frame_age_s_{0.25};
  cv::Matx33d camera_mount_rotation_{cv::Matx33d::eye()};
  cv::Point3d camera_mount_translation_;
  cv::Matx33d frame_rotation_{cv::Matx33d::eye()};
  cv::Point3d frame_camera_position_;
  bool frame_camera_pose_valid_{false};

  bool use_robot_id_targeting_{false};
  bool stm_color_locked_{false};
  int enemy_color_red_value_{0};
  int enemy_color_blue_value_{1};
  int target_class_{BLUE_CLASS};
  int stage_log_interval_{300};
  int frame_count_{0};
  StageTotals stage_totals_;
};

}  // namespace yolo_cpp

#endif
