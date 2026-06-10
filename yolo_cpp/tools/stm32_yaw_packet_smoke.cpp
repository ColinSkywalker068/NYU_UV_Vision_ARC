/// @file stm32_yaw_packet_smoke.cpp
/// @brief Manual hardware smoke test that sends one yaw-only Seasky vision packet.

#include "yolo_cpp/seasky_protocol.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

struct Options
{
  std::string port{"/dev/ttyUSB0"};
  int baudrate{115200};
  float yaw_rad{0.05F};
  int count{20};
  int interval_ms{50};
  int fire_mode{yolo_cpp::DEFAULT_FIRE_MODE};
  int target_state{yolo_cpp::TARGET_STATE_CONVERGING};
  int target_type{yolo_cpp::DEFAULT_TARGET_TYPE};
  bool allow_large_yaw{false};
};

void print_usage(const char * program)
{
  std::cout
    << "Usage: " << program << " [options]\n"
    << "\n"
    << "Sends pitch=0 and yaw_rad as a Seasky cmd 0x0001 vision packet.\n"
    << "Use this only with the robot in a safe test stance.\n"
    << "\n"
    << "Options:\n"
    << "  --port PATH           Serial device path (default: /dev/ttyUSB0)\n"
    << "  --baud RATE           Baud rate (default: 115200)\n"
    << "  --yaw-rad VALUE       Yaw error in radians; positive turns one direction (default: 0.05)\n"
    << "  --count N             Number of packets to send (default: 20)\n"
    << "  --interval-ms N       Delay between packets when count > 1 (default: 50)\n"
    << "  --target-state N      1=converging, 2=ready-to-fire (default: 1)\n"
    << "  --target-type N       Controls target type enum value (default: 0 unknown)\n"
    << "  --allow-large-yaw     Allow |yaw| > 0.35 rad\n"
    << "  --help                Show this help\n";
}

int parse_int(const std::string & value, const std::string & name)
{
  try {
    std::size_t parsed = 0;
    const int result = std::stoi(value, &parsed, 0);
    if (parsed != value.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return result;
  } catch (const std::exception & error) {
    throw std::runtime_error("Invalid " + name + " value '" + value + "': " + error.what());
  }
}

float parse_float(const std::string & value, const std::string & name)
{
  try {
    std::size_t parsed = 0;
    const float result = std::stof(value, &parsed);
    if (parsed != value.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return result;
  } catch (const std::exception & error) {
    throw std::runtime_error("Invalid " + name + " value '" + value + "': " + error.what());
  }
}

Options parse_args(int argc, char ** argv)
{
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto require_value = [&](const std::string & name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(name + " requires a value");
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--port") {
      options.port = require_value(arg);
    } else if (arg == "--baud") {
      options.baudrate = parse_int(require_value(arg), arg);
    } else if (arg == "--yaw-rad") {
      options.yaw_rad = parse_float(require_value(arg), arg);
    } else if (arg == "--count") {
      options.count = parse_int(require_value(arg), arg);
    } else if (arg == "--interval-ms") {
      options.interval_ms = parse_int(require_value(arg), arg);
    } else if (arg == "--target-state") {
      options.target_state = parse_int(require_value(arg), arg);
    } else if (arg == "--target-type") {
      options.target_type = parse_int(require_value(arg), arg);
    } else if (arg == "--allow-large-yaw") {
      options.allow_large_yaw = true;
    } else {
      throw std::runtime_error("Unknown argument: " + arg);
    }
  }

  if (options.count < 1) {
    throw std::runtime_error("--count must be at least 1");
  }
  if (options.interval_ms < 0) {
    throw std::runtime_error("--interval-ms must be non-negative");
  }
  if (!options.allow_large_yaw && std::fabs(options.yaw_rad) > 0.35F) {
    throw std::runtime_error("Refusing yaw magnitude > 0.35 rad without --allow-large-yaw");
  }
  return options;
}

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
      return B115200;
#ifdef B230400
    case 230400:
      return B230400;
#endif
#ifdef B460800
    case 460800:
      return B460800;
#endif
#ifdef B921600
    case 921600:
      return B921600;
#endif
    default:
      throw std::runtime_error("Unsupported baud rate: " + std::to_string(baudrate));
  }
}

class SerialPort
{
public:
  SerialPort(const std::string & port, int baudrate)
  {
    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY);
    if (fd_ < 0) {
      throw std::runtime_error("Failed to open " + port + ": " + std::strerror(errno));
    }

    termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
      throw std::runtime_error("tcgetattr failed: " + std::string(std::strerror(errno)));
    }

    cfsetospeed(&tty, baud_to_constant(baudrate));
    cfsetispeed(&tty, baud_to_constant(baudrate));
    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_iflag = IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 5;
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
#ifdef CRTSCTS
    tty.c_cflag &= ~CRTSCTS;
#endif

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
      throw std::runtime_error("tcsetattr failed: " + std::string(std::strerror(errno)));
    }
  }

  ~SerialPort()
  {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  SerialPort(const SerialPort &) = delete;
  SerialPort & operator=(const SerialPort &) = delete;

  void write_packet(const std::vector<uint8_t> & packet)
  {
    std::size_t written = 0;
    while (written < packet.size()) {
      const auto result = ::write(fd_, packet.data() + written, packet.size() - written);
      if (result < 0) {
        throw std::runtime_error("Serial write failed: " + std::string(std::strerror(errno)));
      }
      written += static_cast<std::size_t>(result);
    }
    tcdrain(fd_);
  }

private:
  int fd_{-1};
};

std::string hex_dump(const std::vector<uint8_t> & packet)
{
  std::ostringstream out;
  for (std::size_t i = 0; i < packet.size(); ++i) {
    if (i != 0) {
      out << ' ';
    }
    out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(packet[i]);
  }
  return out.str();
}

}  // namespace

int main(int argc, char ** argv)
{
  try {
    const Options options = parse_args(argc, argv);
    const uint16_t flags =
      yolo_cpp::build_vision_flags(options.fire_mode, options.target_state, options.target_type);
    const auto packet = yolo_cpp::build_seasky_frame(
      yolo_cpp::SEASKY_COMMAND_VISION_RECV, flags, std::vector<float>{0.0F, options.yaw_rad});

    std::cout << "Sending yaw-only STM packet: port=" << options.port
              << " baud=" << options.baudrate << " pitch_rad=0 yaw_rad=" << options.yaw_rad
              << " count=" << options.count << " flags=0x" << std::hex << std::setw(4)
              << std::setfill('0') << flags << std::dec << "\n"
              << "packet[" << packet.size() << "]=" << hex_dump(packet) << "\n";

    SerialPort serial(options.port, options.baudrate);
    for (int i = 0; i < options.count; ++i) {
      serial.write_packet(packet);
      if (i + 1 < options.count) {
        std::this_thread::sleep_for(std::chrono::milliseconds(options.interval_ms));
      }
    }
    std::cout << "Done.\n";
    return 0;
  } catch (const std::exception & error) {
    std::cerr << "stm32_yaw_packet_smoke: " << error.what() << "\n";
    std::cerr << "Run with --help for usage.\n";
    return 1;
  }
}
