#include "yolo_cpp/armor_processor_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<yolo_cpp::ArmorProcessorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
