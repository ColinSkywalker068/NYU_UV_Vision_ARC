/// @file armor_processor_node_main.cpp
/// @brief Standalone executable entry point for ArmorProcessorNode.

#include <rclcpp/rclcpp.hpp>

#include "yolo_cpp/armor_processor_node.hpp"

/// Initialize ROS, spin the armor processor, and shut ROS down.
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<yolo_cpp::ArmorProcessorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
