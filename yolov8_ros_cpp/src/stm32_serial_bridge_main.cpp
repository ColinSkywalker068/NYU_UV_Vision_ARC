#include "yolov8_ros_cpp/stm32_serial_bridge_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<yolov8_ros_cpp::STM32SerialBridgeNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
