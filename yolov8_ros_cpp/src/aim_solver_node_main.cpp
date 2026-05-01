#include "yolov8_ros_cpp/aim_solver_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolov8_ros_cpp::AimSolverNode>());
  rclcpp::shutdown();
  return 0;
}
