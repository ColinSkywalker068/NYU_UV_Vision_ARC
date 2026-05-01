#include "yolov8_ros_cpp/armor_tracker_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolov8_ros_cpp::ArmorTrackerNode>());
  rclcpp::shutdown();
  return 0;
}
