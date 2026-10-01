/// @file stm32_serial_bridge_node.cpp
/// @brief Implements ROS-to-SP/ST USB CDC transport for the STM32 controller.

#include "yolo_cpp/stm32_serial_bridge_node.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <optional>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace yolo_cpp
{

namespace
{

constexpr uint8_t SP_HEADER_0 = static_cast<uint8_t>('S');
constexpr uint8_t SP_HEADER_1 = static_cast<uint8_t>('P');
constexpr uint8_t ST_HEADER_1 = static_cast<uint8_t>('T');
constexpr std::size_t SP_RX_SIZE = 29U;  // VisionToGimbal: Jetson -> MCU.
constexpr std::size_t SP_TX_SIZE = 43U;  // GimbalToVision: MCU -> Jetson.
constexpr std::size_t ST_TX_SIZE = 39U;  // SentryTelemetry: MCU -> Jetson.

constexpr int FIRE_MODE_NO_FIRE = 0;
constexpr int FIRE_MODE_AUTO_FIRE = 1;
constexpr int FIRE_MODE_AUTO_AIM = 2;
constexpr int DEFAULT_FIRE_MODE = FIRE_MODE_AUTO_AIM;
constexpr int DEFAULT_TARGET_TYPE = 0;
constexpr int TARGET_STATE_NO_TARGET = 0;
constexpr int TARGET_STATE_CONVERGING = 1;
constexpr int TARGET_STATE_READY_TO_FIRE = 2;
constexpr double DEFAULT_RX_POLL_PERIOD_SECONDS = 0.01;
constexpr int ENEMY_COLOR_RED = 0;
constexpr int ENEMY_COLOR_BLUE = 1;
constexpr int WORK_MODE_AIM = 0;

constexpr uint16_t CRC16_TABLE[256] = {
  0x0000, 0x1189, 0x2312, 0x329B, 0x4624, 0x57AD, 0x6536, 0x74BF, 0x8C48, 0x9DC1, 0xAF5A, 0xBED3,
  0xCA6C, 0xDBE5, 0xE97E, 0xF8F7, 0x1081, 0x0108, 0x3393, 0x221A, 0x56A5, 0x472C, 0x75B7, 0x643E,
  0x9CC9, 0x8D40, 0xBFDB, 0xAE52, 0xDAED, 0xCB64, 0xF9FF, 0xE876, 0x2102, 0x308B, 0x0210, 0x1399,
  0x6726, 0x76AF, 0x4434, 0x55BD, 0xAD4A, 0xBCC3, 0x8E58, 0x9FD1, 0xEB6E, 0xFAE7, 0xC87C, 0xD9F5,
  0x3183, 0x200A, 0x1291, 0x0318, 0x77A7, 0x662E, 0x54B5, 0x453C, 0xBDCB, 0xAC42, 0x9ED9, 0x8F50,
  0xFBEF, 0xEA66, 0xD8FD, 0xC974, 0x4204, 0x538D, 0x6116, 0x709F, 0x0420, 0x15A9, 0x2732, 0x36BB,
  0xCE4C, 0xDFC5, 0xED5E, 0xFCD7, 0x8868, 0x99E1, 0xAB7A, 0xBAF3, 0x5285, 0x430C, 0x7197, 0x601E,
  0x14A1, 0x0528, 0x37B3, 0x263A, 0xDECD, 0xCF44, 0xFDDF, 0xEC56, 0x98E9, 0x8960, 0xBBFB, 0xAA72,
  0x6306, 0x728F, 0x4014, 0x519D, 0x2522, 0x34AB, 0x0630, 0x17B9, 0xEF4E, 0xFEC7, 0xCC5C, 0xDDD5,
  0xA96A, 0xB8E3, 0x8A78, 0x9BF1, 0x7387, 0x620E, 0x5095, 0x411C, 0x35A3, 0x242A, 0x16B1, 0x0738,
  0xFFCF, 0xEE46, 0xDCDD, 0xCD54, 0xB9EB, 0xA862, 0x9AF9, 0x8B70, 0x8408, 0x9581, 0xA71A, 0xB693,
  0xC22C, 0xD3A5, 0xE13E, 0xF0B7, 0x0840, 0x19C9, 0x2B52, 0x3ADB, 0x4E64, 0x5FED, 0x6D76, 0x7CFF,
  0x9489, 0x8500, 0xB79B, 0xA612, 0xD2AD, 0xC324, 0xF1BF, 0xE036, 0x18C1, 0x0948, 0x3BD3, 0x2A5A,
  0x5EE5, 0x4F6C, 0x7DF7, 0x6C7E, 0xA50A, 0xB483, 0x8618, 0x9791, 0xE32E, 0xF2A7, 0xC03C, 0xD1B5,
  0x2942, 0x38CB, 0x0A50, 0x1BD9, 0x6F66, 0x7EEF, 0x4C74, 0x5DFD, 0xB58B, 0xA402, 0x9699, 0x8710,
  0xF3AF, 0xE226, 0xD0BD, 0xC134, 0x39C3, 0x284A, 0x1AD1, 0x0B58, 0x7FE7, 0x6E6E, 0x5CF5, 0x4D7C,
  0xC60C, 0xD785, 0xE51E, 0xF497, 0x8028, 0x91A1, 0xA33A, 0xB2B3, 0x4A44, 0x5BCD, 0x6956, 0x78DF,
  0x0C60, 0x1DE9, 0x2F72, 0x3EFB, 0xD68D, 0xC704, 0xF59F, 0xE416, 0x90A9, 0x8120, 0xB3BB, 0xA232,
  0x5AC5, 0x4B4C, 0x79D7, 0x685E, 0x1CE1, 0x0D68, 0x3FF3, 0x2E7A, 0xE70E, 0xF687, 0xC41C, 0xD595,
  0xA12A, 0xB0A3, 0x8238, 0x93B1, 0x6B46, 0x7ACF, 0x4854, 0x59DD, 0x2D62, 0x3CEB, 0x0E70, 0x1FF9,
  0xF78F, 0xE606, 0xD49D, 0xC514, 0xB1AB, 0xA022, 0x92B9, 0x8330, 0x7BC7, 0x6A4E, 0x58D5, 0x495C,
  0x3DE3, 0x2C6A, 0x1EF1, 0x0F78};

uint16_t compute_sp_crc16(const uint8_t* data, std::size_t len)
{
  uint16_t crc = 0xFFFF;
  for (std::size_t i = 0; i < len; ++i) {
    crc = static_cast<uint16_t>((crc >> 8) ^ CRC16_TABLE[(crc ^ data[i]) & 0x00FFU]);
  }
  return crc;
}

uint16_t read_u16_le(const std::vector<uint8_t>& data, std::size_t offset)
{
  return static_cast<uint16_t>(data[offset])
         | static_cast<uint16_t>(static_cast<uint16_t>(data[offset + 1]) << 8U);
}

void append_u16_le(std::vector<uint8_t>& data, uint16_t value)
{
  data.push_back(static_cast<uint8_t>(value & 0xFFU));
  data.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
}

void append_float_le(std::vector<uint8_t>& data, float value)
{
  const auto* raw = reinterpret_cast<const uint8_t*>(&value);
  data.insert(data.end(), raw, raw + sizeof(float));
}

float read_float_le(const std::vector<uint8_t>& data, std::size_t offset)
{
  float value = 0.0F;
  std::memcpy(&value, data.data() + offset, sizeof(float));
  return value;
}

uint8_t fire_mode_to_sp_mode(int fire_mode)
{
  if (fire_mode == FIRE_MODE_AUTO_FIRE) {
    return 2U;
  }
  if (fire_mode == FIRE_MODE_AUTO_AIM) {
    return 1U;
  }
  return 0U;
}

uint16_t build_vision_flags(int fire_mode, int target_state, int target_type)
{
  return static_cast<uint16_t>(
    (fire_mode & 0x0003) | ((target_state & 0x0003) << 2) | ((target_type & 0x000F) << 4));
}

struct AttitudeFrame
{
  float yaw{0.0F};
  float pitch{0.0F};
  float roll{0.0F};
  float yaw_rate{0.0F};
  float pitch_rate{0.0F};
  int enemy_color{-1};
  int work_mode{WORK_MODE_AIM};
  int bullet_speed{0};
  int flags_register{0};
  int command_id{0};
  int robot_id{0};
};

std::vector<uint8_t> build_sp_command_frame(
  uint8_t mode, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  std::vector<uint8_t> packet;
  packet.reserve(SP_RX_SIZE);
  packet.push_back(SP_HEADER_0);
  packet.push_back(SP_HEADER_1);
  packet.push_back(mode);
  append_float_le(packet, yaw);
  append_float_le(packet, yaw_vel);
  append_float_le(packet, yaw_acc);
  append_float_le(packet, pitch);
  append_float_le(packet, pitch_vel);
  append_float_le(packet, pitch_acc);
  append_u16_le(packet, compute_sp_crc16(packet.data(), packet.size()));
  return packet;
}

bool crc_ok(const std::vector<uint8_t>& frame)
{
  if (frame.size() < 2U) {
    return false;
  }
  return compute_sp_crc16(frame.data(), frame.size() - 2U) == read_u16_le(frame, frame.size() - 2U);
}

std::optional<AttitudeFrame> parse_sp_frame(const std::vector<uint8_t>& frame)
{
  if (frame.size() != SP_TX_SIZE || frame[0] != SP_HEADER_0 || frame[1] != SP_HEADER_1
      || !crc_ok(frame)) {
    return std::nullopt;
  }

  AttitudeFrame parsed;
  parsed.command_id = static_cast<int>((SP_HEADER_0 << 8U) | SP_HEADER_1);
  parsed.work_mode = std::max(0, static_cast<int>(frame[2]) - 1);
  parsed.yaw = read_float_le(frame, 19);
  parsed.pitch = read_float_le(frame, 27);
  parsed.yaw_rate = read_float_le(frame, 23);
  parsed.pitch_rate = read_float_le(frame, 31);
  const float speed = read_float_le(frame, 35);
  if (!std::isfinite(parsed.yaw) || !std::isfinite(parsed.pitch) ||
      !std::isfinite(parsed.yaw_rate) || !std::isfinite(parsed.pitch_rate) ||
      !std::isfinite(speed) || speed < 0.0F || speed > 100.0F) {
    return std::nullopt;
  }
  // SP already includes the controller's wxyz quaternion. Older test/firmware
  // streams may leave it zero; yaw and pitch remain authoritative in that case.
  const double w = read_float_le(frame, 3), x = read_float_le(frame, 7);
  const double y = read_float_le(frame, 11), z = read_float_le(frame, 15);
  const double qnorm2 = w*w + x*x + y*y + z*z;
  if (!std::isfinite(qnorm2)) {return std::nullopt;}
  if (qnorm2 > 1e-12) {
    if (qnorm2 < 0.25 || qnorm2 > 4.0) {return std::nullopt;}
    parsed.roll = static_cast<float>(std::atan2(2.0*(w*x+y*z), qnorm2-2.0*(x*x+y*y)));
  }
  parsed.bullet_speed = static_cast<int>(speed);
  parsed.flags_register = (parsed.work_mode & 0x0003) | ((parsed.bullet_speed & 0x00FF) << 8);
  return parsed;
}

std::optional<AttitudeFrame> parse_st_frame(const std::vector<uint8_t>& frame)
{
  if (frame.size() != ST_TX_SIZE || frame[0] != SP_HEADER_0 || frame[1] != ST_HEADER_1
      || !crc_ok(frame)) {
    return std::nullopt;
  }

  AttitudeFrame parsed;
  parsed.command_id = static_cast<int>((SP_HEADER_0 << 8U) | ST_HEADER_1);
  parsed.robot_id = static_cast<int>(frame[30]);
  // ST offset 33 is shooter heat, not bullet speed. Speed arrives in SP.
  parsed.bullet_speed = 0;
  const int team_color = static_cast<int>(frame[35]);  // 0=red, 1=blue.
  parsed.enemy_color = team_color == 0 ? ENEMY_COLOR_BLUE : ENEMY_COLOR_RED;
  parsed.flags_register = parsed.enemy_color | ((parsed.bullet_speed & 0x00FF) << 8);
  return parsed;
}

std::vector<AttitudeFrame> parse_sp_st_frames(std::vector<uint8_t>& rx_buffer)
{
  std::vector<AttitudeFrame> parsed_frames;

  while (true) {
    auto header = std::find(rx_buffer.begin(), rx_buffer.end(), SP_HEADER_0);
    if (header == rx_buffer.end()) {
      rx_buffer.clear();
      break;
    }
    if (header != rx_buffer.begin()) {
      rx_buffer.erase(rx_buffer.begin(), header);
    }
    if (rx_buffer.size() < 2U) {
      break;
    }

    std::size_t frame_size = 0U;
    if (rx_buffer[1] == SP_HEADER_1) {
      frame_size = SP_TX_SIZE;
    } else if (rx_buffer[1] == ST_HEADER_1) {
      frame_size = ST_TX_SIZE;
    } else {
      rx_buffer.erase(rx_buffer.begin());
      continue;
    }

    if (rx_buffer.size() < frame_size) {
      break;
    }

    std::vector<uint8_t> frame(rx_buffer.begin(), rx_buffer.begin() + frame_size);
    std::optional<AttitudeFrame> parsed =
      frame_size == SP_TX_SIZE ? parse_sp_frame(frame) : parse_st_frame(frame);
    if (parsed.has_value()) {
      parsed_frames.push_back(*parsed);
      rx_buffer.erase(rx_buffer.begin(), rx_buffer.begin() + frame_size);
    } else {
      rx_buffer.erase(rx_buffer.begin());
    }
  }

  return parsed_frames;
}

/// Convert an integer baud rate parameter into a termios constant.
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

}  // namespace

/// Thin non-blocking POSIX serial wrapper for SP/ST TX/RX frames.
class SerialTransport
{
public:
  SerialTransport(rclcpp::Logger logger, std::string port, int baudrate, bool verbose)
      : logger_(std::move(logger)), port_(std::move(port)), baudrate_(baudrate), verbose_(verbose)
  {
  }

  ~SerialTransport()
  {
    close();
  }

  bool connect()
  {
    close();
    next_connect_attempt_ = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    // The bridge stays alive without a serial device so vision can be tested on dev machines.
    fd_ = ::open(port_.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
      RCLCPP_WARN(logger_, "Failed to open serial port %s: %s. Bridge will stay alive without TX.",
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

    // Discard bytes queued before this connection; new telemetry must be fresh.
    if (tcflush(fd_, TCIFLUSH) != 0) {
      RCLCPP_WARN(logger_, "tcflush failed for %s: %s", port_.c_str(), std::strerror(errno));
      close();
      return false;
    }

    connected_ = true;
    last_valid_rx_ = std::chrono::steady_clock::now();
    next_rx_warning_ = last_valid_rx_ + std::chrono::seconds(2);
    session_rx_bytes_ = 0;
    RCLCPP_INFO(logger_, "STM32 serial connected: %s @ %d baud", port_.c_str(), baudrate_);
    return true;
  }

  bool send_vision_data(float pitch, float yaw, int fire_mode, int target_state, int target_type)
  {
    (void)target_type;
    if (!connected_ || fd_ < 0) {
      return false;
    }

    const uint8_t mode =
      target_state == TARGET_STATE_NO_TARGET ? 0U : fire_mode_to_sp_mode(fire_mode);
    const auto packet = build_sp_command_frame(mode, yaw, 0.0F, 0.0F, pitch, 0.0F, 0.0F);
    const auto bytes_written =
      ::write(fd_, reinterpret_cast<const char*>(packet.data()), packet.size());
    if (bytes_written < 0) {
      const int error = errno;
      if (error != EAGAIN && error != EWOULDBLOCK && error != EINTR) {
        disconnect("write", error);
      }
      return false;
    }
    if (static_cast<std::size_t>(bytes_written) != packet.size()) {
      RCLCPP_WARN(logger_, "Incomplete serial write: %ld/%zu bytes; dropping command",
        static_cast<long>(bytes_written), packet.size());
      return false;
    }
    ++message_count_;

    if (verbose_ || target_state == TARGET_STATE_CONVERGING) {
      RCLCPP_INFO(logger_, "[SP TX #%d] mode=%u pitch=%.6f yaw=%.6f bytes=%ld",
        message_count_, static_cast<unsigned>(mode), static_cast<double>(pitch),
        static_cast<double>(yaw), static_cast<long>(bytes_written));
    }
    return true;
  }

  std::vector<AttitudeFrame> read_attitude_frames()
  {
    if (!connected_ || fd_ < 0) {
      if (std::chrono::steady_clock::now() < next_connect_attempt_ || !connect()) {
        return {};
      }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_rx_warning_) {
      const double idle_seconds = std::chrono::duration<double>(now - last_valid_rx_).count();
      RCLCPP_WARN(logger_, "Serial open but no valid SP/ST telemetry for %.1f seconds "
        "(%zu bytes this session); check controller stream/protocol", idle_seconds, session_rx_bytes_);
      next_rx_warning_ = now + std::chrono::seconds(5);
    }

    // A removed USB device can keep a valid-looking fd with no available bytes.
    // Check hangup/error events before FIONREAD, then reopen the stable by-id path.
    pollfd port_state{fd_, POLLIN, 0};
    const int poll_result = ::poll(&port_state, 1, 0);
    if (poll_result < 0) {
      if (errno != EINTR) {
        disconnect("poll", errno);
      }
      return {};
    }
    if (port_state.revents & (POLLHUP | POLLERR | POLLNVAL)) {
      disconnect("USB hangup", 0);
      return {};
    }
    if (!(port_state.revents & POLLIN)) {
      return {};
    }

    int available_bytes = 0;
    if (ioctl(fd_, FIONREAD, &available_bytes) != 0) {
      if (errno != EINTR) {
        disconnect("FIONREAD", errno);
      }
      return {};
    }
    if (available_bytes <= 0) {
      return {};
    }

    const std::size_t old_size = rx_buffer_.size();
    rx_buffer_.resize(old_size + static_cast<std::size_t>(available_bytes));
    const auto bytes_read =
      ::read(fd_, rx_buffer_.data() + old_size, static_cast<std::size_t>(available_bytes));
    const int read_error = errno;
    if (bytes_read <= 0) {
      rx_buffer_.resize(old_size);
      if (bytes_read == 0) {
        disconnect("read EOF", 0);
      } else if (read_error != EAGAIN && read_error != EWOULDBLOCK && read_error != EINTR) {
        disconnect("read", read_error);
      }
      return {};
    }
    rx_buffer_.resize(old_size + static_cast<std::size_t>(bytes_read));
    session_rx_bytes_ += static_cast<std::size_t>(bytes_read);
    auto frames = parse_sp_st_frames(rx_buffer_);
    if (!frames.empty()) {
      last_valid_rx_ = std::chrono::steady_clock::now();
      next_rx_warning_ = last_valid_rx_ + std::chrono::seconds(2);
    }
    return frames;
  }

  void close()
  {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    connected_ = false;
    // Never combine an incomplete frame from one USB session with the next.
    rx_buffer_.clear();
  }

private:
  void disconnect(const char* operation, int error)
  {
    RCLCPP_WARN(logger_, "Serial disconnected (%s%s%s); retrying %s in 1 second",
      operation, error ? ": " : "", error ? std::strerror(error) : "", port_.c_str());
    close();
    next_connect_attempt_ = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  }

  rclcpp::Logger logger_;
  std::string port_;
  int baudrate_{115200};
  bool verbose_{false};
  int fd_{-1};
  bool connected_{false};
  int message_count_{0};
  std::vector<uint8_t> rx_buffer_;
  std::chrono::steady_clock::time_point next_connect_attempt_{};
  std::chrono::steady_clock::time_point last_valid_rx_{};
  std::chrono::steady_clock::time_point next_rx_warning_{};
  std::size_t session_rx_bytes_{0};
};

constexpr char kVisionLogDirName[] = "vision_logs";
constexpr float kRadToDeg = 57.29577951308232F;

std::string timestamp_now_iso8601_ms()
{
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  const auto milliseconds =
    std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds).count();
  const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &now_c);
#else
  localtime_r(&now_c, &local_tm);
#endif

  std::ostringstream out;
  out << std::put_time(&local_tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0')
      << milliseconds;
  return out.str();
}

std::string timestamp_now_compact()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &now_c);
#else
  localtime_r(&now_c, &local_tm);
#endif

  std::ostringstream out;
  out << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
  return out.str();
}

STM32SerialBridgeNode::STM32SerialBridgeNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("stm32_serial_bridge_node", options)
{
  this->declare_parameter<bool>("transmit_enable", false);
  attitude_receive_delay_s_ = this->declare_parameter<double>("attitude_receive_delay_s", 0.0);
  if (!std::isfinite(attitude_receive_delay_s_) || attitude_receive_delay_s_ < 0.0 ||
      attitude_receive_delay_s_ > 0.25) {
    throw std::runtime_error("attitude_receive_delay_s must be in [0, 0.25]");
  }
  const bool receive_enable = this->declare_parameter<bool>("receive_enable", false);
  const auto command_mode = this->declare_parameter<std::string>("command_output_mode", "absolute");
  this->declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
  this->declare_parameter<int>("serial_baudrate", 115200);
  this->declare_parameter<bool>("serial_verbose", false);
  this->declare_parameter<std::string>("runtime_log_dir", "");
  this->declare_parameter<int>("fire_mode", DEFAULT_FIRE_MODE);
  this->declare_parameter<int>("hold_fire_mode", FIRE_MODE_NO_FIRE);
  this->declare_parameter<int>("target_type", DEFAULT_TARGET_TYPE);
  this->declare_parameter<bool>("transmit_angles_in_degrees", false);

  transmit_enable_ = this->get_parameter("transmit_enable").as_bool();
  if (this->declare_parameter<bool>("calibration_provisional", false)) {
    // Provisional intrinsics are for observation only, even if TX is requested.
    transmit_enable_ = false;
    this->set_parameter(rclcpp::Parameter("transmit_enable", false));
    RCLCPP_WARN(this->get_logger(), "UNCALIBRATED preview: controller TX locked off");
  }
  serial_verbose_ = this->get_parameter("serial_verbose").as_bool();
  runtime_log_dir_ = this->get_parameter("runtime_log_dir").as_string();
  fire_mode_ = this->get_parameter("fire_mode").as_int();
  hold_fire_mode_ = this->get_parameter("hold_fire_mode").as_int();
  target_type_ = this->get_parameter("target_type").as_int();
  transmit_angles_in_degrees_ = this->get_parameter("transmit_angles_in_degrees").as_bool();
  if (transmit_angles_in_degrees_ || (transmit_enable_ && command_mode != "absolute")) {
    throw std::runtime_error("SP requires absolute angles in radians; offset preview cannot transmit");
  }
  target_state_ = TARGET_STATE_NO_TARGET;

  runtime_log_path_ = (std::filesystem::path(resolve_runtime_log_dir())
                       / ("vision_serial_runtime_" + timestamp_now_compact() + ".log"))
                        .string();
  runtime_log_stream_.open(runtime_log_path_, std::ios::app);
  if (!runtime_log_stream_.is_open()) {
    RCLCPP_WARN(
      this->get_logger(), "Failed to open serial runtime log file: %s", runtime_log_path_.c_str());
  } else {
    RCLCPP_INFO(this->get_logger(), "Serial runtime logs: %s", runtime_log_path_.c_str());
  }

  if (transmit_enable_ || receive_enable) {
    transport_ = std::make_unique<SerialTransport>(this->get_logger(),
      this->get_parameter("serial_port").as_string(),
      this->get_parameter("serial_baudrate").as_int(), serial_verbose_);
    transport_->connect();
  } else {
    RCLCPP_INFO(this->get_logger(), "Transmit disabled. Set transmit_enable:=true to enable TX.");
  }

  const bool legacy_commands = this->declare_parameter<bool>("use_legacy_command_topics", false);
  if (legacy_commands) {
  override_subscription_ = this->create_subscription<std_msgs::msg::Bool>("override", 10,
    std::bind(&STM32SerialBridgeNode::handle_override, this, std::placeholders::_1));
  suggest_fire_subscription_ = this->create_subscription<std_msgs::msg::Bool>("suggest_fire", 10,
    std::bind(&STM32SerialBridgeNode::handle_suggest_fire, this, std::placeholders::_1));
  angle_subscription_ =
    this->create_subscription<std_msgs::msg::Float32MultiArray>("bounding_box_angles", 10,
      std::bind(&STM32SerialBridgeNode::handle_angles, this, std::placeholders::_1));

  } else {
    solution_subscription_ = this->create_subscription<cv_auto_aim_interfaces::msg::AimSolution>(
      "aim_solution", 10, std::bind(&STM32SerialBridgeNode::handle_solution, this, std::placeholders::_1));
  }
  attitude_publisher_ =
    this->create_publisher<std_msgs::msg::Float32MultiArray>("attitude_data", 10);
  stamped_attitude_publisher_ =
    this->create_publisher<cv_auto_aim_interfaces::msg::GimbalAttitude>("gimbal/attitude", 100);
  robot_id_publisher_ = this->create_publisher<std_msgs::msg::Int32>("robot_id", 10);
  attitude_flags_publisher_ =
    this->create_publisher<std_msgs::msg::Int32MultiArray>("attitude_flags", 10);

  rx_timer_ = this->create_wall_timer(std::chrono::duration<double>(DEFAULT_RX_POLL_PERIOD_SECONDS),
    std::bind(&STM32SerialBridgeNode::handle_rx_poll, this));
  RCLCPP_INFO(this->get_logger(), "STM32 serial bridge ready. transmit_enable=%s, port=%s, baud=%ld",
    transmit_enable_ ? "true" : "false", this->get_parameter("serial_port").as_string().c_str(),
    this->get_parameter("serial_baudrate").as_int());
}

STM32SerialBridgeNode::~STM32SerialBridgeNode()
{
  if (transport_) {
    transport_->close();
  }
  if (runtime_log_stream_.is_open()) {
    runtime_log_stream_.close();
  }
}

void STM32SerialBridgeNode::handle_override(const std_msgs::msg::Bool::SharedPtr msg)
{
  const bool prev_has_target = has_target_;
  has_target_ = msg->data;
  override_time_ = std::chrono::steady_clock::now();
  target_state_ = has_target_
                    ? (suggest_fire_ ? TARGET_STATE_READY_TO_FIRE : TARGET_STATE_CONVERGING)
                    : TARGET_STATE_NO_TARGET;
  if (!has_target_ && prev_has_target) {
    have_command_ = false;
    maybe_send();
  }
}

void STM32SerialBridgeNode::handle_suggest_fire(const std_msgs::msg::Bool::SharedPtr msg)
{
  suggest_fire_ = msg->data;
  target_state_ = has_target_
                    ? (suggest_fire_ ? TARGET_STATE_READY_TO_FIRE : TARGET_STATE_CONVERGING)
                    : TARGET_STATE_NO_TARGET;
  // Transmit only on a new angle sample, not on an independent fire flag.
}

void STM32SerialBridgeNode::handle_angles(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() != 2 || !std::isfinite(msg->data[0]) || !std::isfinite(msg->data[1]) ||
      std::abs(msg->data[0]) > 3.15F || std::abs(msg->data[1]) > 3.15F) {
    have_command_ = false;
    has_target_ = false;
    maybe_send();
    RCLCPP_WARN(this->get_logger(), "Received malformed angle message");
    return;
  }
  latest_yaw_ = msg->data[0];
  latest_pitch_ = msg->data[1];
  command_time_ = std::chrono::steady_clock::now();
  have_command_ = true;
  maybe_send();
}

void STM32SerialBridgeNode::handle_solution(
  const cv_auto_aim_interfaces::msg::AimSolution::SharedPtr msg)
{
  const auto stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
  const auto age_ns = this->now().nanoseconds() - stamp_ns;
  const bool valid = msg->has_solution && msg->output_mode == "absolute" &&
    msg->header.frame_id == "gimbal_inertial" &&
    stamp_ns > last_solution_stamp_ns_ && stamp_ns > 0 && age_ns >= 0 && age_ns <= 250000000 &&
    std::isfinite(msg->yaw) && std::isfinite(msg->pitch);
  // A replayed or malformed solution must not revive a previous command.
  has_target_ = valid;
  have_command_ = valid;
  suggest_fire_ = valid && msg->suggest_fire;
  if (valid) {
    last_solution_stamp_ns_ = stamp_ns;
    latest_yaw_ = msg->yaw;
    latest_pitch_ = msg->pitch;
    command_time_ = override_time_ = std::chrono::steady_clock::now();
  }
  maybe_send();
}

void STM32SerialBridgeNode::handle_rx_poll()
{
  if (!transport_) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  if (has_target_ && ((have_command_ && now - command_time_ > std::chrono::milliseconds(250)) ||
      now - override_time_ > std::chrono::milliseconds(250))) {
    has_target_ = false;
    have_command_ = false;
    maybe_send();
  }
  const auto frames = transport_->read_attitude_frames();
  const auto received = this->get_clock()->now();
  const AttitudeFrame * newest_sp = nullptr;
  for (const auto & frame : frames) {
    if (frame.command_id == static_cast<int>((SP_HEADER_0 << 8U) | SP_HEADER_1)) {
      newest_sp = &frame;
    }
  }
  // The wire has no device clock. Discard older samples in a read batch instead
  // of assigning multiple historical poses the same receive timestamp.
  for (const auto& frame : frames) {
    std_msgs::msg::Float32MultiArray attitude_msg;
    attitude_msg.data = {frame.yaw, frame.pitch, frame.roll};
    // ST carries status, not attitude. Never replace real SP angles with zeros.
    if (&frame == newest_sp &&
        std::isfinite(frame.yaw) && std::isfinite(frame.pitch)) {
      attitude_publisher_->publish(attitude_msg);
      cv_auto_aim_interfaces::msg::GimbalAttitude sample;
      sample.header.stamp = received - rclcpp::Duration::from_seconds(attitude_receive_delay_s_);
      sample.header.frame_id = "gimbal_inertial";
      sample.receive_stamp_ns = received.nanoseconds();
      sample.source_timestamped = false;
      sample.valid = true;
      sample.yaw = frame.yaw;
      sample.pitch = frame.pitch;
      sample.roll = frame.roll;
      sample.yaw_rate = frame.yaw_rate;
      sample.pitch_rate = frame.pitch_rate;
      stamped_attitude_publisher_->publish(sample);
    }

    if (frame.robot_id > 0) {
      std_msgs::msg::Int32 robot_id_msg;
      robot_id_msg.data = frame.robot_id;
      robot_id_publisher_->publish(robot_id_msg);
    }

    if (frame.enemy_color >= 0) {
      std_msgs::msg::Int32MultiArray flags_msg;
      flags_msg.data = {frame.enemy_color, frame.work_mode, frame.bullet_speed,
        frame.flags_register, frame.command_id};
      attitude_flags_publisher_->publish(flags_msg);
    }
    log_rx_event(frame.yaw, frame.pitch, frame.roll, frame.enemy_color, frame.work_mode,
      frame.bullet_speed, frame.flags_register, frame.command_id);
  }
}

void STM32SerialBridgeNode::maybe_send()
{
  if (!transmit_enable_ || !transport_) {
    return;
  }
  const float yaw_tx = transmit_angles_in_degrees_ ? latest_yaw_ * kRadToDeg : latest_yaw_;
  const float pitch_tx = transmit_angles_in_degrees_ ? latest_pitch_ * kRadToDeg : latest_pitch_;
  const auto now = std::chrono::steady_clock::now();
  const bool active = has_target_ && have_command_ &&
      now - command_time_ < std::chrono::milliseconds(250) &&
      now - override_time_ < std::chrono::milliseconds(250);
  target_state_ = active
                    ? (suggest_fire_ ? TARGET_STATE_READY_TO_FIRE : TARGET_STATE_CONVERGING)
                    : TARGET_STATE_NO_TARGET;
  const int configured_mode = (fire_mode_ == FIRE_MODE_AUTO_FIRE && !suggest_fire_) ?
    FIRE_MODE_AUTO_AIM : fire_mode_;
  const int fire_mode = active ? configured_mode : FIRE_MODE_NO_FIRE;
  const bool success =
    transport_->send_vision_data(pitch_tx, yaw_tx, fire_mode, target_state_, target_type_);
  log_tx_event(yaw_tx, pitch_tx, success, fire_mode);
}

std::string STM32SerialBridgeNode::resolve_runtime_log_dir() const
{
  if (!runtime_log_dir_.empty()) {
    std::filesystem::create_directories(runtime_log_dir_);
    return runtime_log_dir_;
  }
  if (const char* env_dir = std::getenv("VISION_LOG_DIR")) {
    const std::string value(env_dir);
    if (!value.empty()) {
      std::filesystem::create_directories(value);
      return value;
    }
  }
  const auto log_dir = std::filesystem::current_path() / kVisionLogDirName;
  std::filesystem::create_directories(log_dir);
  return log_dir.string();
}

void STM32SerialBridgeNode::write_runtime_log_line(const std::string& line)
{
  if (!runtime_log_stream_.is_open()) {
    return;
  }
  runtime_log_stream_ << "ts=" << timestamp_now_iso8601_ms() << ' ' << line << '\n';
  runtime_log_stream_.flush();
}

void STM32SerialBridgeNode::log_tx_event(float yaw, float pitch, bool success, int fire_mode)
{
  const uint16_t flags_register = build_vision_flags(fire_mode, target_state_, target_type_);
  std::ostringstream line;
  line << std::fixed << std::setprecision(6) << "kind=serial_tx ok=" << (success ? 1 : 0)
       << " angle_unit=" << (transmit_angles_in_degrees_ ? "degrees" : "radians") << " yaw=" << yaw
       << " pitch=" << pitch << " fire_mode=" << fire_mode << " target_state=" << target_state_
       << " target_type=" << target_type_ << " flags=0x" << std::uppercase << std::hex
       << std::setw(4) << std::setfill('0') << flags_register;
  write_runtime_log_line(line.str());
}

void STM32SerialBridgeNode::log_rx_event(float yaw, float pitch, float roll, int enemy_color,
  int work_mode, int bullet_speed, int flags_register, int command_id)
{
  std::ostringstream line;
  line << std::fixed << std::setprecision(6) << "kind=serial_rx"
       << " yaw=" << yaw << " pitch=" << pitch << " roll=" << roll << " enemy_color=" << enemy_color
       << " work_mode=" << work_mode << " bullet_speed=" << bullet_speed << " flags=0x"
       << std::uppercase << std::hex << std::setw(4) << std::setfill('0') << flags_register
       << " cmd=0x" << std::setw(4) << command_id;
  write_runtime_log_line(line.str());
}

}  // namespace yolo_cpp
