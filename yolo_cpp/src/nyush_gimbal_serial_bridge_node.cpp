#include <fcntl.h>
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

#include <cv_auto_aim_interfaces/msg/aim_solution.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include "yolo_cpp/nyush_gimbal_protocol.hpp"

namespace yolo_cpp::nyush_gimbal {
namespace {

using namespace std::chrono_literals;

speed_t baud_to_constant(int baudrate) {
  switch (baudrate) {
  case 9600:
    return B9600;
  case 19200:
    return B19200;
  case 38400:
    return B38400;
  case 57600:
    return B57600;
#ifdef B921600
  case 921600:
    return B921600;
#endif
  case 115200:
  default:
    return B115200;
  }
}

class RawSerial {
public:
  RawSerial(rclcpp::Logger logger, std::string port, int baudrate)
      : logger_(std::move(logger)), port_(std::move(port)),
        baudrate_(baudrate) {}

  ~RawSerial() { close(); }

  bool connect() {
    if (connected()) {
      return true;
    }

    fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
      RCLCPP_WARN(logger_, "Failed to open NYUSH gimbal port %s: %s",
                  port_.c_str(), std::strerror(errno));
      return false;
    }

    termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
      RCLCPP_WARN(logger_, "tcgetattr failed for %s: %s", port_.c_str(),
                  std::strerror(errno));
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
      RCLCPP_WARN(logger_, "tcsetattr failed for %s: %s", port_.c_str(),
                  std::strerror(errno));
      close();
      return false;
    }

    RCLCPP_INFO(logger_, "NYUSH gimbal serial connected: %s @ %d baud",
                port_.c_str(), baudrate_);
    return true;
  }

  bool connected() const { return fd_ >= 0; }

  bool send(const VisionToGimbal &packet) {
    if (!connected()) {
      return false;
    }
    const auto bytes = to_bytes(packet);
    const auto bytes_written = ::write(
        fd_, reinterpret_cast<const char *>(bytes.data()), bytes.size());
    if (bytes_written < 0 ||
        static_cast<std::size_t>(bytes_written) != bytes.size()) {
      RCLCPP_WARN(logger_, "NYUSH gimbal write error: %s",
                  std::strerror(errno));
      close();
      return false;
    }
    return true;
  }

  std::vector<AttitudePacket> read_packets() {
    if (!connected()) {
      return {};
    }

    int available_bytes = 0;
    if (ioctl(fd_, FIONREAD, &available_bytes) != 0 || available_bytes <= 0) {
      return {};
    }

    const std::size_t old_size = rx_buffer_.size();
    rx_buffer_.resize(old_size + static_cast<std::size_t>(available_bytes));
    const auto bytes_read = ::read(fd_, rx_buffer_.data() + old_size,
                                   static_cast<std::size_t>(available_bytes));
    if (bytes_read <= 0) {
      rx_buffer_.resize(old_size);
      return {};
    }
    rx_buffer_.resize(old_size + static_cast<std::size_t>(bytes_read));
    return parse_gimbal_packets(rx_buffer_);
  }

  void close() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

private:
  rclcpp::Logger logger_;
  std::string port_;
  int baudrate_{115200};
  int fd_{-1};
  std::vector<uint8_t> rx_buffer_;
};

} // namespace

class NYUSHGimbalSerialBridge : public rclcpp::Node {
public:
  explicit NYUSHGimbalSerialBridge(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : rclcpp::Node("nyush_gimbal_serial_bridge", options) {
    declare_parameter<bool>("transmit_enable", true);
    declare_parameter<std::string>("serial_port",
                                   "/tmp/nyush-rm-sentry-vision");
    declare_parameter<int>("serial_baudrate", 115200);
    declare_parameter<bool>("serial_verbose", false);
    declare_parameter<int>("enemy_color", 1);
    declare_parameter<int>("rx_yaw_sign", -1);
    declare_parameter<int>("rx_pitch_sign", -1);
    declare_parameter<int>("rx_roll_sign", 1);
    declare_parameter<int>("tx_yaw_sign", -1);
    declare_parameter<int>("tx_pitch_sign", -1);

    transmit_enable_ = get_parameter("transmit_enable").as_bool();
    serial_verbose_ = get_parameter("serial_verbose").as_bool();
    enemy_color_ = get_parameter("enemy_color").as_int();
    rx_yaw_sign_ = get_parameter("rx_yaw_sign").as_int();
    rx_pitch_sign_ = get_parameter("rx_pitch_sign").as_int();
    rx_roll_sign_ = get_parameter("rx_roll_sign").as_int();
    tx_yaw_sign_ = get_parameter("tx_yaw_sign").as_int();
    tx_pitch_sign_ = get_parameter("tx_pitch_sign").as_int();

    if (transmit_enable_) {
      serial_ = std::make_unique<RawSerial>(
          get_logger(), get_parameter("serial_port").as_string(),
          get_parameter("serial_baudrate").as_int());
      serial_->connect();
    }

    aim_solution_subscription_ =
        create_subscription<cv_auto_aim_interfaces::msg::AimSolution>(
            "aim_solution", 10,
            std::bind(&NYUSHGimbalSerialBridge::handle_aim_solution, this,
                      std::placeholders::_1));
    attitude_publisher_ =
        create_publisher<std_msgs::msg::Float32MultiArray>("attitude_data", 10);
    attitude_flags_publisher_ =
        create_publisher<std_msgs::msg::Int32MultiArray>("attitude_flags", 10);

    rx_timer_ = create_wall_timer(
        std::chrono::duration<double>(DEFAULT_RX_POLL_PERIOD_SECONDS),
        std::bind(&NYUSHGimbalSerialBridge::poll_serial, this));
    reconnect_timer_ = create_wall_timer(
        1s, std::bind(&NYUSHGimbalSerialBridge::ensure_connected, this));

    RCLCPP_INFO(get_logger(),
                "NYUSH gimbal bridge ready. transmit_enable=%s port=%s baud=%d",
                transmit_enable_ ? "true" : "false",
                get_parameter("serial_port").as_string().c_str(),
                get_parameter("serial_baudrate").as_int());
  }

private:
  static int checked_sign(int sign) { return sign < 0 ? -1 : 1; }

  void ensure_connected() {
    if (transmit_enable_ && serial_ && !serial_->connected()) {
      serial_->connect();
    }
  }

  void handle_aim_solution(
      const cv_auto_aim_interfaces::msg::AimSolution::SharedPtr msg) {
    if (!transmit_enable_ || !serial_) {
      return;
    }

    const bool control = msg->has_solution;
    const bool fire = msg->has_solution && msg->suggest_fire;
    const float yaw = static_cast<float>(checked_sign(tx_yaw_sign_)) * msg->yaw;
    const float pitch =
        static_cast<float>(checked_sign(tx_pitch_sign_)) * msg->pitch;
    const auto packet = build_vision_packet(control, fire, yaw, pitch);
    const bool ok = serial_->send(packet);

    if (serial_verbose_ || fire || !ok) {
      RCLCPP_INFO(get_logger(),
                  "[NYUSH Gimbal TX] ok=%s mode=%u yaw=%.6f pitch=%.6f fire=%s",
                  ok ? "true" : "false", packet.mode, static_cast<double>(yaw),
                  static_cast<double>(pitch), fire ? "true" : "false");
    }
  }

  void poll_serial() {
    if (!serial_) {
      return;
    }

    for (const auto &packet : serial_->read_packets()) {
      std_msgs::msg::Float32MultiArray attitude;
      attitude.data = {
          static_cast<float>(checked_sign(rx_yaw_sign_)) * packet.yaw,
          static_cast<float>(checked_sign(rx_pitch_sign_)) * packet.pitch,
          static_cast<float>(checked_sign(rx_roll_sign_)) * packet.roll,
      };
      attitude_publisher_->publish(attitude);

      std_msgs::msg::Int32MultiArray flags;
      flags.data = {enemy_color_, static_cast<int>(packet.mode),
                    static_cast<int>(packet.bullet_speed),
                    static_cast<int>(packet.bullet_count), 0};
      attitude_flags_publisher_->publish(flags);

      if (serial_verbose_) {
        RCLCPP_INFO(get_logger(),
                    "[NYUSH Gimbal RX] yaw=%.6f pitch=%.6f roll=%.6f mode=%u "
                    "bullet_speed=%.2f "
                    "bullet_count=%u",
                    static_cast<double>(attitude.data[0]),
                    static_cast<double>(attitude.data[1]),
                    static_cast<double>(attitude.data[2]), packet.mode,
                    static_cast<double>(packet.bullet_speed),
                    packet.bullet_count);
      }
    }
  }

  std::unique_ptr<RawSerial> serial_;
  rclcpp::Subscription<cv_auto_aim_interfaces::msg::AimSolution>::SharedPtr
      aim_solution_subscription_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr
      attitude_publisher_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr
      attitude_flags_publisher_;
  rclcpp::TimerBase::SharedPtr rx_timer_;
  rclcpp::TimerBase::SharedPtr reconnect_timer_;

  bool transmit_enable_{true};
  bool serial_verbose_{false};
  int enemy_color_{1};
  int rx_yaw_sign_{-1};
  int rx_pitch_sign_{-1};
  int rx_roll_sign_{1};
  int tx_yaw_sign_{-1};
  int tx_pitch_sign_{-1};
};

int run_nyush_gimbal_serial_bridge(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<NYUSHGimbalSerialBridge>());
  rclcpp::shutdown();
  return 0;
}

} // namespace yolo_cpp::nyush_gimbal
