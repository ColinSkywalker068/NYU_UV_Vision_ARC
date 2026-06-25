#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "yolo_cpp/rm_handoff_tracker_node.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolo_cpp::RmHandoffTrackerNode>());
  rclcpp::shutdown();
  return 0;
}
