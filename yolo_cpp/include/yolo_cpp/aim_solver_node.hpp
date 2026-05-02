#ifndef YOLO_CPP__AIM_SOLVER_NODE_HPP_
#define YOLO_CPP__AIM_SOLVER_NODE_HPP_

#include "cv_auto_aim_interfaces/msg/aim_solution.hpp"
#include "cv_auto_aim_interfaces/msg/target.hpp"
#include "yolo_cpp/visibility_control.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <string>
#include <utility>

namespace yolo_cpp
{

class YOLO_CPP_PUBLIC AimSolverNode : public rclcpp::Node
{
public:
  explicit AimSolverNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~AimSolverNode() override = default;

private:
  void handle_target(const cv_auto_aim_interfaces::msg::Target::SharedPtr msg);
  void handle_attitude(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void handle_attitude_flags(const std_msgs::msg::Int32MultiArray::SharedPtr msg);
  void publish_no_target(int image_width, int image_height);
  std::pair<float, float> solve_3d_angles(
    const cv_auto_aim_interfaces::msg::Target & target, float * flight_time, float * range) const;

  rclcpp::Subscription<cv_auto_aim_interfaces::msg::Target>::SharedPtr target_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr attitude_subscription_;
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
  float current_yaw_rad_{0.0F};
  float current_pitch_rad_{0.0F};
  bool have_attitude_{false};
  float hfov_rad_{0.0F};
  float vfov_rad_{0.0F};
};

}  // namespace yolo_cpp

#endif
