#include "yolov8_ros_cpp/perception_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<yolov8_ros_cpp::PerceptionNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
