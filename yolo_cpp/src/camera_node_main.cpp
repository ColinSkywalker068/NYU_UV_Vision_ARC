/// @file camera_node_main.cpp
/// @brief Standalone executable entry point for CameraNode.

#include <rclcpp/rclcpp.hpp>

#include "yolo_cpp/camera_node.hpp"

/// Initialize ROS, spin the camera publisher, and shut ROS down.
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  {
    auto node = std::make_shared<yolo_cpp::CameraNode>();
    rclcpp::spin(node);
    node.reset();
  }
  rclcpp::shutdown();
  return 0;
}
