#include "yolo_cpp/armor_detector_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolo_cpp::ArmorDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
