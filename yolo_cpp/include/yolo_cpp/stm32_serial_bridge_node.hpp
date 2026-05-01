#ifndef YOLO_CPP__STM32_SERIAL_BRIDGE_NODE_HPP_
#define YOLO_CPP__STM32_SERIAL_BRIDGE_NODE_HPP_

#include "yolo_cpp/visibility_control.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <memory>
#include <string>
#include <vector>
#include <fstream>

namespace yolo_cpp
{

class SerialTransport;

class YOLO_CPP_PUBLIC STM32SerialBridgeNode : public rclcpp::Node
{
public:
  explicit STM32SerialBridgeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~STM32SerialBridgeNode() override;

private:
  void handle_override(const std_msgs::msg::Bool::SharedPtr msg);
  void handle_angles(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
  void handle_rx_poll();
  void maybe_send();
  std::string resolve_runtime_log_dir() const;
  void write_runtime_log_line(const std::string & line);
  void log_tx_event(float yaw, float pitch, bool success);
  void log_rx_event(
    float yaw, float pitch, float roll, int enemy_color, int work_mode, int bullet_speed,
    int flags_register, int command_id);

  std::unique_ptr<SerialTransport> transport_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr override_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr angle_subscription_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr attitude_publisher_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr attitude_flags_publisher_;
  rclcpp::TimerBase::SharedPtr rx_timer_;

  bool transmit_enable_{true};
  bool serial_verbose_{false};
  bool has_target_{false};
  std::string runtime_log_dir_;
  std::string runtime_log_path_;
  int fire_mode_{0};
  int target_type_{0};
  int target_state_{0};
  float latest_yaw_{0.0F};
  float latest_pitch_{0.0F};
  std::ofstream runtime_log_stream_;
};

}  // namespace yolo_cpp

#endif
