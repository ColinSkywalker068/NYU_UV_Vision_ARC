/// @file stm32_serial_bridge_main.cpp
/// @brief Standalone executable entry point for STM32SerialBridgeNode.

#include <rclcpp/rclcpp.hpp>

#include "yolo_cpp/stm32_serial_bridge_node.hpp"

/// Initialize ROS, spin the serial bridge, and shut ROS down.
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<yolo_cpp::STM32SerialBridgeNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
