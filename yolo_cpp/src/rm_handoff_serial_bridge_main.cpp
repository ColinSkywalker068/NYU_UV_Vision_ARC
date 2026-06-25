#include <rclcpp/rclcpp.hpp>

namespace yolo_cpp::rm_handoff {

int run_rm_handoff_serial_bridge(int argc, char **argv);

} // namespace yolo_cpp::rm_handoff

int main(int argc, char **argv) {
  return yolo_cpp::rm_handoff::run_rm_handoff_serial_bridge(argc, argv);
}
