#ifndef YOLO_CPP__AIM_SOLVER_NODE_HPP_
#define YOLO_CPP__AIM_SOLVER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <string>
#include <utility>

#include "cv_auto_aim_interfaces/msg/aim_solution.hpp"
#include "cv_auto_aim_interfaces/msg/target.hpp"
#include "cv_auto_aim_interfaces/msg/gimbal_attitude.hpp"
#include "yolo_cpp/aim_prediction.hpp"
#include "yolo_cpp/visibility_control.hpp"

namespace yolo_cpp
{

/// Converts selected targets into yaw/pitch commands and fire suggestions.
class YOLO_CPP_PUBLIC AimSolverNode : public rclcpp::Node
{
public:
  /// Create solver subscriptions and command publishers.
  explicit AimSolverNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~AimSolverNode() override = default;

private:
  /// Apply live tuning of the prediction parameters (ros2 param set); rejects start-up-only ones.
  rcl_interfaces::msg::SetParametersResult on_parameters(const std::vector<rclcpp::Parameter>& parameters);

  /// Solve and publish aim output for one target message.
  void handle_target(const cv_auto_aim_interfaces::msg::Target::SharedPtr msg);

  /// Update measured robot attitude and bullet speed.
  void handle_attitude(const cv_auto_aim_interfaces::msg::GimbalAttitude::SharedPtr msg);

  /// Update controller-provided flags used by solver behavior.
  void handle_attitude_flags(const std_msgs::msg::Int32MultiArray::SharedPtr msg);

  void publish_no_target(int image_width, int image_height, const char * reason = "no_target_or_unsynchronized");

  rclcpp::Subscription<cv_auto_aim_interfaces::msg::Target>::SharedPtr target_subscription_;
  rclcpp::Subscription<cv_auto_aim_interfaces::msg::GimbalAttitude>::SharedPtr attitude_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr attitude_flags_subscription_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr bbox_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr override_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr suggest_fire_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr angle_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr obb_angle_publisher_;
  rclcpp::Publisher<cv_auto_aim_interfaces::msg::AimSolution>::SharedPtr aim_solution_publisher_;

  std_msgs::msg::Int32MultiArray bbox_msg_;
  std_msgs::msg::Bool override_msg_;
  std_msgs::msg::Bool suggest_fire_msg_;
  std_msgs::msg::Float32MultiArray angle_msg_;
  std_msgs::msg::Float32 obb_angle_msg_;

  bool publish_zero_angles_when_no_target_{true};
  bool enable_fire_recommendation_{false};
  bool use_3d_solution_{true};
  bool compensate_gravity_{true};
  std::string target_topic_{"tracker/target"};
  std::string output_mode_{"offset"};
  int default_image_width_{640};
  int default_image_height_{480};
  float default_bullet_speed_mps_{15.0F};
  float bullet_speed_mps_{15.0F};
  float solver_latency_s_{0.0F};
  float gravity_mps2_{9.80665F};
  AttitudeHistory attitude_history_;
  std::mutex attitude_mutex_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_handle_;
  double attitude_sync_tolerance_ms_{30.0};
  double attitude_max_gap_ms_{50.0};
  AimPredictionConfig prediction_config_;
  int previous_slot_{-1};
  int previous_track_id_{-1};
  float hfov_rad_{0.0F};
  float vfov_rad_{0.0F};
};

}  // namespace yolo_cpp

#endif
