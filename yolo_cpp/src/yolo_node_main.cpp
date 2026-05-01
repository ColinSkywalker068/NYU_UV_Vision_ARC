#include "yolo_cpp/perception_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<yolo_cpp::PerceptionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
