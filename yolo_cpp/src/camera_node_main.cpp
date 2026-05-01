#include "yolo_cpp/camera_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<yolo_cpp::CameraNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
