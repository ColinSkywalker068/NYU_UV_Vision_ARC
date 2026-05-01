#ifndef YOLOV8_ROS_CPP__ARMOR_TRACKER_NODE_HPP_
#define YOLOV8_ROS_CPP__ARMOR_TRACKER_NODE_HPP_

#include "cv_auto_aim_interfaces/msg/armors.hpp"
#include "cv_auto_aim_interfaces/msg/target.hpp"
#include "cv_auto_aim_interfaces/msg/tracked_armors.hpp"
#include "yolov8_ros_cpp/tracker.hpp"
#include "yolov8_ros_cpp/visibility_control.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>

namespace yolov8_ros_cpp
{

class YOLOV8_ROS_CPP_PUBLIC ArmorTrackerNode : public rclcpp::Node
{
public:
  explicit ArmorTrackerNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~ArmorTrackerNode() override = default;

private:
  struct StageTotals
  {
    double tracking_ms{0.0};
    double publish_ms{0.0};
  };

  void armors_callback(const cv_auto_aim_interfaces::msg::Armors::SharedPtr msg);
  void id_listener_callback(const std_msgs::msg::Int32::SharedPtr msg);
  void handle_attitude_flags(const std_msgs::msg::Int32MultiArray::SharedPtr msg);
  void publish_tracked_armors(
    const cv_auto_aim_interfaces::msg::Armors & source_msg,
    const TrackingFrameResult & result);
  void publish_target(
    const cv_auto_aim_interfaces::msg::Armors & source_msg,
    const std::optional<TrackSnapshot> & target);
  void log_stage_timing();

  rclcpp::Subscription<cv_auto_aim_interfaces::msg::Armors>::SharedPtr armors_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr robot_id_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr attitude_flags_subscription_;
  rclcpp::Publisher<cv_auto_aim_interfaces::msg::TrackedArmors>::SharedPtr tracked_armors_publisher_;
  rclcpp::Publisher<cv_auto_aim_interfaces::msg::Target>::SharedPtr target_publisher_;

  TrackManager track_manager_;
  rclcpp::Time last_frame_time_;
  bool have_last_frame_time_{false};
  std::chrono::steady_clock::time_point last_stage_log_wall_time_{};
  bool have_last_stage_log_wall_time_{false};
  std::mutex state_mutex_;

  bool use_robot_id_targeting_{false};
  bool stm_color_locked_{false};
  int enemy_color_red_value_{0};
  int enemy_color_blue_value_{1};
  int target_class_{BLUE_CLASS};
  int stage_log_interval_{300};
  int frame_count_{0};
  StageTotals stage_totals_;
};

}  // namespace yolov8_ros_cpp

#endif
