#include "yolo_cpp/armor_tracker_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolo_cpp::ArmorTrackerNode>());
  rclcpp::shutdown();
  return 0;
}
