/// @file perception_node_main.cpp
/// @brief Standalone executable entry point for the legacy PerceptionNode.

#include "yolo_cpp/perception_node.hpp"

#include <rclcpp/rclcpp.hpp>

/// Initialize ROS, spin the combined perception node, and shut ROS down.
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  {
    auto node = std::make_shared<yolo_cpp::PerceptionNode>();
    rclcpp::spin(node);
    node.reset();
  }
  rclcpp::shutdown();
  return 0;
}
