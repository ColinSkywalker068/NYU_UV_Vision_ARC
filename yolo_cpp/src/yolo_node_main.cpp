/// @file yolo_node_main.cpp
/// @brief Compatibility executable that currently spins the combined PerceptionNode.

#include "yolo_cpp/perception_node.hpp"

#include <rclcpp/rclcpp.hpp>

/// Initialize ROS, spin the compatibility perception entry point, and shut ROS down.
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<yolo_cpp::PerceptionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
