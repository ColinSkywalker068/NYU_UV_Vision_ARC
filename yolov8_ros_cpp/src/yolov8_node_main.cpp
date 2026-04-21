#include "yolov8_ros_cpp/yolov8_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<yolov8_ros_cpp::YOLOv8Node>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
