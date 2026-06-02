/// @file aim_solver_node_main.cpp
/// @brief Standalone executable entry point for AimSolverNode.

#include <rclcpp/rclcpp.hpp>

#include "yolo_cpp/aim_solver_node.hpp"

/// Initialize ROS, spin the aim solver, and shut ROS down.
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolo_cpp::AimSolverNode>());
  rclcpp::shutdown();
  return 0;
}
