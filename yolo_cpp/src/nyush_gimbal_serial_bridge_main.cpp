#include <rclcpp/rclcpp.hpp>

namespace yolo_cpp::nyush_gimbal {

int run_nyush_gimbal_serial_bridge(int argc, char **argv);

} // namespace yolo_cpp::nyush_gimbal

int main(int argc, char **argv) {
  return yolo_cpp::nyush_gimbal::run_nyush_gimbal_serial_bridge(argc, argv);
}
