/// @file armor_detector_node_main.cpp
/// @brief Standalone executable entry point for ArmorDetectorNode.

#include "yolo_cpp/armor_detector_node.hpp"

#include <rclcpp/rclcpp.hpp>

/// Initialize ROS, spin the armor detector, and shut ROS down.
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  {
    auto node = std::make_shared<yolo_cpp::ArmorDetectorNode>();
    rclcpp::spin(node);
    node.reset();
  }
  rclcpp::shutdown();
  return 0;
}
