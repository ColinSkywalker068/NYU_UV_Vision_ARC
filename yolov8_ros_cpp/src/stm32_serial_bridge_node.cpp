#include "yolov8_ros_cpp/stm32_serial_bridge_node.hpp"

#include "yolov8_ros_cpp/seasky_protocol.hpp"

#include <fcntl.h>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace yolov8_ros_cpp
{

speed_t baud_to_constant(int baudrate)
{
  switch (baudrate) {
    case 9600:
      return B9600;
    case 19200:
      return B19200;
    case 38400:
      return B38400;
    case 57600:
      return B57600;
    case 115200:
    default:
      return B115200;
  }
}

class SerialTransport
{
public:
  SerialTransport(rclcpp::Logger logger, std::string port, int baudrate, bool verbose)
  : logger_(std::move(logger)),
    port_(std::move(port)),
    baudrate_(baudrate),
    verbose_(verbose)
  {
  }

  ~SerialTransport()
  {
    close();
  }

  bool connect()
  {
    fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
      RCLCPP_WARN(
        logger_, "Failed to open serial port %s: %s. Bridge will stay alive without TX.",
        port_.c_str(), std::strerror(errno));
      return false;
    }

    termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
      RCLCPP_WARN(logger_, "tcgetattr failed for %s: %s", port_.c_str(), std::strerror(errno));
      close();
      return false;
    }

    cfsetospeed(&tty, baud_to_constant(baudrate_));
    cfsetispeed(&tty, baud_to_constant(baudrate_));
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag = IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
      RCLCPP_WARN(logger_, "tcsetattr failed for %s: %s", port_.c_str(), std::strerror(errno));
      close();
      return false;
    }

    connected_ = true;
    RCLCPP_INFO(logger_, "STM32 serial connected: %s @ %d baud", port_.c_str(), baudrate_);
    return true;
  }

  bool send_vision_data(
    float pitch, float yaw, int fire_mode, int target_state, int target_type)
  {
    if (!connected_ || fd_ < 0) {
      return false;
    }

    const uint16_t flags_register = build_vision_flags(fire_mode, target_state, target_type);
    const auto packet = build_seasky_frame(
      SEASKY_COMMAND_VISION_RECV, flags_register, std::vector<float>{pitch, yaw});
    const auto bytes_written =
      ::write(fd_, reinterpret_cast<const char *>(packet.data()), packet.size());
    if (bytes_written < 0) {
      connected_ = false;
      RCLCPP_WARN(logger_, "Serial write error: %s", std::strerror(errno));
      return false;
    }
    ++message_count_;

    if (verbose_ || target_state == TARGET_STATE_CONVERGING) {
      RCLCPP_INFO(
        logger_, "[Seasky TX #%d] pitch=%.6f yaw=%.6f flags=0x%04X bytes=%ld",
        message_count_, static_cast<double>(pitch), static_cast<double>(yaw), flags_register,
        static_cast<long>(bytes_written));
    }
    return true;
  }

  std::vector<AttitudeFrame> read_attitude_frames()
  {
    if (!connected_ || fd_ < 0) {
      return {};
    }

    int available_bytes = 0;
    if (ioctl(fd_, FIONREAD, &available_bytes) != 0 || available_bytes <= 0) {
      return {};
    }

    const std::size_t old_size = rx_buffer_.size();
    rx_buffer_.resize(old_size + static_cast<std::size_t>(available_bytes));
    const auto bytes_read = ::read(fd_, rx_buffer_.data() + old_size, static_cast<std::size_t>(available_bytes));
    if (bytes_read <= 0) {
      rx_buffer_.resize(old_size);
      return {};
    }
    rx_buffer_.resize(old_size + static_cast<std::size_t>(bytes_read));
    return parse_attitude_frames(rx_buffer_);
  }

  void close()
  {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    connected_ = false;
  }

private:
  rclcpp::Logger logger_;
  std::string port_;
  int baudrate_{115200};
  bool verbose_{false};
  int fd_{-1};
  bool connected_{false};
  int message_count_{0};
  std::vector<uint8_t> rx_buffer_;
};

namespace
{

}  // namespace

STM32SerialBridgeNode::STM32SerialBridgeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("stm32_serial_bridge_node", options)
{
  this->declare_parameter<bool>("transmit_enable", true);
  this->declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
  this->declare_parameter<int>("serial_baudrate", 115200);
  this->declare_parameter<bool>("serial_verbose", false);
  this->declare_parameter<int>("fire_mode", DEFAULT_FIRE_MODE);
  this->declare_parameter<int>("target_type", DEFAULT_TARGET_TYPE);

  transmit_enable_ = this->get_parameter("transmit_enable").as_bool();
  serial_verbose_ = this->get_parameter("serial_verbose").as_bool();
  fire_mode_ = this->get_parameter("fire_mode").as_int();
  target_type_ = this->get_parameter("target_type").as_int();
  target_state_ = TARGET_STATE_NO_TARGET;

  if (transmit_enable_) {
    transport_ = std::make_unique<SerialTransport>(
      this->get_logger(),
      this->get_parameter("serial_port").as_string(),
      this->get_parameter("serial_baudrate").as_int(),
      serial_verbose_);
    transport_->connect();
  } else {
    RCLCPP_INFO(this->get_logger(), "Transmit disabled. Set transmit_enable:=true to enable TX.");
  }

  override_subscription_ = this->create_subscription<std_msgs::msg::Bool>(
    "override", 10, std::bind(&STM32SerialBridgeNode::handle_override, this, std::placeholders::_1));
  angle_subscription_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
    "bounding_box_angles", 10,
    std::bind(&STM32SerialBridgeNode::handle_angles, this, std::placeholders::_1));

  attitude_publisher_ =
    this->create_publisher<std_msgs::msg::Float32MultiArray>("attitude_data", 10);
  attitude_flags_publisher_ =
    this->create_publisher<std_msgs::msg::Int32MultiArray>("attitude_flags", 10);

  rx_timer_ = this->create_wall_timer(
    std::chrono::duration<double>(DEFAULT_RX_POLL_PERIOD_SECONDS),
    std::bind(&STM32SerialBridgeNode::handle_rx_poll, this));
}

STM32SerialBridgeNode::~STM32SerialBridgeNode()
{
  if (transport_) {
    transport_->close();
  }
}

void STM32SerialBridgeNode::handle_override(const std_msgs::msg::Bool::SharedPtr msg)
{
  const bool prev_has_target = has_target_;
  has_target_ = msg->data;
  target_state_ = has_target_ ? TARGET_STATE_CONVERGING : TARGET_STATE_NO_TARGET;
  if (has_target_ && !prev_has_target) {
    maybe_send();
  }
}

void STM32SerialBridgeNode::handle_angles(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 2) {
    RCLCPP_WARN(this->get_logger(), "Received malformed angle message");
    return;
  }
  latest_yaw_ = msg->data[0];
  latest_pitch_ = msg->data[1];
  maybe_send();
}

void STM32SerialBridgeNode::handle_rx_poll()
{
  if (!transport_) {
    return;
  }
  const auto frames = transport_->read_attitude_frames();
  for (const auto & frame : frames) {
    std_msgs::msg::Float32MultiArray attitude_msg;
    attitude_msg.data = {frame.yaw, frame.pitch, frame.roll};
    attitude_publisher_->publish(attitude_msg);

    std_msgs::msg::Int32MultiArray flags_msg;
    flags_msg.data = {
      frame.enemy_color, frame.work_mode, frame.bullet_speed, frame.flags_register, frame.command_id};
    attitude_flags_publisher_->publish(flags_msg);
  }
}

void STM32SerialBridgeNode::maybe_send()
{
  if (!transmit_enable_ || !transport_) {
    return;
  }
  transport_->send_vision_data(
    latest_pitch_, latest_yaw_, fire_mode_, target_state_, target_type_);
}

}  // namespace yolov8_ros_cpp
