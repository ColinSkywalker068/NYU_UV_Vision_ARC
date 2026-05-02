/// @file armor_tracker_node_main.cpp
/// @brief Standalone executable entry point for ArmorTrackerNode.

#include "yolo_cpp/armor_tracker_node.hpp"

#include <rclcpp/rclcpp.hpp>

/// Initialize ROS, spin the armor tracker, and shut ROS down.
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolo_cpp::ArmorTrackerNode>());
  rclcpp::shutdown();
  return 0;
}
