#ifndef YOLO_CPP__AIM_SOLVER_NODE_HPP_
#define YOLO_CPP__AIM_SOLVER_NODE_HPP_

#include "cv_auto_aim_interfaces/msg/target.hpp"
#include "yolo_cpp/visibility_control.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

namespace yolo_cpp
{

class YOLO_CPP_PUBLIC AimSolverNode : public rclcpp::Node
{
public:
  explicit AimSolverNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~AimSolverNode() override = default;

private:
  void handle_target(const cv_auto_aim_interfaces::msg::Target::SharedPtr msg);
  void publish_no_target(int image_width, int image_height);

  rclcpp::Subscription<cv_auto_aim_interfaces::msg::Target>::SharedPtr target_subscription_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr bbox_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr override_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr angle_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr obb_angle_publisher_;

  std_msgs::msg::Int32MultiArray bbox_msg_;
  std_msgs::msg::Bool override_msg_;
  std_msgs::msg::Float32MultiArray angle_msg_;
  std_msgs::msg::Float32 obb_angle_msg_;

  bool publish_zero_angles_when_no_target_{true};
  int default_image_width_{640};
  int default_image_height_{480};
  float hfov_rad_{0.0F};
  float vfov_rad_{0.0F};
};

}  // namespace yolo_cpp

#endif
