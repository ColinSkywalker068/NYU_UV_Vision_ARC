#ifndef YOLO_CPP__STM32_SERIAL_BRIDGE_NODE_HPP_
#define YOLO_CPP__STM32_SERIAL_BRIDGE_NODE_HPP_

#include <fstream>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <string>
#include <vector>

#include "yolo_cpp/visibility_control.hpp"

namespace yolo_cpp
{

class SerialTransport;

/// Bridges ROS aim commands to the STM32 SP/ST USB CDC protocol.
class YOLO_CPP_PUBLIC STM32SerialBridgeNode : public rclcpp::Node
{
public:
  /// Open serial transport, subscriptions, publishers, and RX polling timer.
  explicit STM32SerialBridgeNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

  /// Flush logs and close serial transport.
  ~STM32SerialBridgeNode() override;

private:
  /// Update whether the vision stack currently owns aim control.
  void handle_override(const std_msgs::msg::Bool::SharedPtr msg);

  /// Update fire suggestion state from the aim solver.
  void handle_suggest_fire(const std_msgs::msg::Bool::SharedPtr msg);

  /// Cache latest yaw/pitch command and attempt transmission.
  void handle_angles(const std_msgs::msg::Float32MultiArray::SharedPtr msg);

  /// Poll serial RX for attitude frames from the STM32.
  void handle_rx_poll();

  /// Send the latest vision command when transmission is enabled.
  void maybe_send();

  /// Resolve the runtime log directory for serial telemetry.
  std::string resolve_runtime_log_dir() const;

  /// Append one line to the serial runtime log.
  void write_runtime_log_line(const std::string& line);

  /// Log one transmitted command event.
  void log_tx_event(float yaw, float pitch, bool success, int fire_mode);

  /// Log one received attitude/status frame.
  void log_rx_event(float yaw, float pitch, float roll, int enemy_color, int work_mode,
    int bullet_speed, int flags_register, int command_id);

  std::unique_ptr<SerialTransport> transport_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr override_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr suggest_fire_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr angle_subscription_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr attitude_publisher_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr robot_id_publisher_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr attitude_flags_publisher_;
  rclcpp::TimerBase::SharedPtr rx_timer_;

  bool transmit_enable_{true};
  bool serial_verbose_{false};
  bool has_target_{false};
  bool suggest_fire_{false};
  bool transmit_angles_in_degrees_{false};
  std::string runtime_log_dir_;
  std::string runtime_log_path_;
  int fire_mode_{0};
  int hold_fire_mode_{0};
  int target_type_{0};
  int target_state_{0};
  float latest_yaw_{0.0F};
  float latest_pitch_{0.0F};
  std::ofstream runtime_log_stream_;
};

}  // namespace yolo_cpp

#endif
