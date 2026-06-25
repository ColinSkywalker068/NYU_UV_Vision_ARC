#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <auto_aim_interfaces/msg/target.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rm_decision_interfaces/msg/all_robot_hp.hpp>
#include <rm_decision_interfaces/msg/game_status.hpp>
#include <rm_decision_interfaces/msg/robot_control.hpp>
#include <rm_decision_interfaces/msg/robot_status.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>

#include "yolo_cpp/rm_handoff_crc.hpp"
#include "yolo_cpp/rm_handoff_packet.hpp"

namespace yolo_cpp::rm_handoff {
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
  case 115200:
  default:
    return B115200;
  }
}

bool is_handoff_header(uint8_t value) {
  return value == 0x5A || value == 0x5B || value == 0x5C || value == 0x5D;
}

std::size_t packet_size_for_header(uint8_t header) {
  switch (header) {
  case 0x5A:
    return sizeof(ReceivePacketVision);
  case 0x5B:
    return sizeof(ReceivePacketAllRobotHP);
  case 0x5C:
    return sizeof(ReceivePacketGameStatus);
  case 0x5D:
    return sizeof(ReceivePacketRobotStatus);
  default:
    return 0;
  }
}

template <typename PacketT> bool packet_crc_ok(const PacketT &packet) {
  return crc16::verify_crc16_check_sum(
      reinterpret_cast<const uint8_t *>(&packet), sizeof(packet));
}

} // namespace

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
      RCLCPP_WARN(logger_, "Failed to open handoff serial port %s: %s",
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

    RCLCPP_INFO(logger_, "RM handoff serial connected: %s @ %d baud",
                port_.c_str(), baudrate_);
    return true;
  }

  bool connected() const { return fd_ >= 0; }

  bool send(const uint8_t *data, std::size_t size) {
    if (!connected()) {
      return false;
    }
    const auto bytes_written =
        ::write(fd_, reinterpret_cast<const char *>(data), size);
    if (bytes_written < 0 || static_cast<std::size_t>(bytes_written) != size) {
      RCLCPP_WARN(logger_, "RM handoff serial write error: %s",
                  std::strerror(errno));
      close();
      return false;
    }
    return true;
  }

  std::vector<uint8_t> read_available() {
    if (!connected()) {
      return {};
    }

    int available_bytes = 0;
    if (ioctl(fd_, FIONREAD, &available_bytes) != 0 || available_bytes <= 0) {
      return {};
    }

    std::vector<uint8_t> data(static_cast<std::size_t>(available_bytes));
    const auto bytes_read = ::read(fd_, data.data(), data.size());
    if (bytes_read <= 0) {
      return {};
    }
    data.resize(static_cast<std::size_t>(bytes_read));
    return data;
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
};

class RMHandoffSerialBridge : public rclcpp::Node {
public:
  explicit RMHandoffSerialBridge(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
      : rclcpp::Node("rm_handoff_serial_bridge", options) {
    declare_parameter<std::string>("device_name", "/dev/ttyUSB0");
    declare_parameter<int>("baud_rate", 115200);
    declare_parameter<double>("timestamp_offset", 0.0);
    declare_parameter<std::string>("gimbal_parent_frame", "gimbal_odom");
    declare_parameter<std::string>("gimbal_child_frame", "gimbal_link");
    declare_parameter<std::string>("detect_color_parameter", "detect_color");
    declare_parameter<std::vector<std::string>>(
        "detect_color_nodes",
        std::vector<std::string>{"armor_detector", "rm_rune_detector"});

    serial_ = std::make_unique<RawSerial>(
        get_logger(), get_parameter("device_name").as_string(),
        get_parameter("baud_rate").as_int());
    serial_->connect();

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    reset_tracker_client_ =
        create_client<std_srvs::srv::Trigger>("/tracker/reset");
    for (const auto &node_name :
         get_parameter("detect_color_nodes").as_string_array()) {
      detect_color_clients_.push_back(
          std::make_shared<rclcpp::AsyncParametersClient>(this, node_name));
    }

    latency_pub_ = create_publisher<std_msgs::msg::Float64>("/latency", 10);
    aiming_point_pub_ =
        create_publisher<visualization_msgs::msg::Marker>("/aiming_point", 10);
    game_status_pub_ =
        create_publisher<rm_decision_interfaces::msg::GameStatus>(
            "/game_status", 1);
    robot_status_pub_ =
        create_publisher<rm_decision_interfaces::msg::RobotStatus>(
            "/robot_status", 10);
    all_robot_hp_pub_ =
        create_publisher<rm_decision_interfaces::msg::AllRobotHP>(
            "/all_robot_hp", 3);

    target_sub_ = create_subscription<auto_aim_interfaces::msg::Target>(
        "/tracker/target", rclcpp::SensorDataQoS(),
        std::bind(&RMHandoffSerialBridge::send_vision_packet, this,
                  std::placeholders::_1));
    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel_chassis", rclcpp::QoS(rclcpp::KeepLast(1)),
        std::bind(&RMHandoffSerialBridge::send_twist_packet, this,
                  std::placeholders::_1));
    robot_control_sub_ =
        create_subscription<rm_decision_interfaces::msg::RobotControl>(
            "/robot_control", rclcpp::QoS(rclcpp::KeepLast(1)),
            std::bind(&RMHandoffSerialBridge::send_robot_control_packet, this,
                      std::placeholders::_1));

    aiming_point_.header.frame_id =
        get_parameter("gimbal_parent_frame").as_string();
    aiming_point_.ns = "aiming_point";
    aiming_point_.type = visualization_msgs::msg::Marker::SPHERE;
    aiming_point_.action = visualization_msgs::msg::Marker::ADD;
    aiming_point_.scale.x = 0.12;
    aiming_point_.scale.y = 0.12;
    aiming_point_.scale.z = 0.12;
    aiming_point_.color.r = 1.0;
    aiming_point_.color.g = 1.0;
    aiming_point_.color.b = 1.0;
    aiming_point_.color.a = 1.0;
    aiming_point_.lifetime = rclcpp::Duration::from_seconds(0.1);

    rx_timer_ = create_wall_timer(
        2ms, std::bind(&RMHandoffSerialBridge::poll_serial, this));
    reconnect_timer_ = create_wall_timer(
        1s, std::bind(&RMHandoffSerialBridge::ensure_connected, this));
    RCLCPP_INFO(get_logger(), "RM handoff serial bridge ready");
  }

private:
  void ensure_connected() {
    if (serial_ && !serial_->connected()) {
      serial_->connect();
    }
  }

  void poll_serial() {
    if (!serial_ || !serial_->connected()) {
      return;
    }

    const auto bytes = serial_->read_available();
    rx_buffer_.insert(rx_buffer_.end(), bytes.begin(), bytes.end());

    while (!rx_buffer_.empty()) {
      const auto header_it =
          std::find_if(rx_buffer_.begin(), rx_buffer_.end(), is_handoff_header);
      if (header_it == rx_buffer_.end()) {
        rx_buffer_.clear();
        return;
      }
      rx_buffer_.erase(rx_buffer_.begin(), header_it);

      const std::size_t packet_size =
          packet_size_for_header(rx_buffer_.front());
      if (packet_size == 0) {
        rx_buffer_.erase(rx_buffer_.begin());
        continue;
      }
      if (rx_buffer_.size() < packet_size) {
        return;
      }

      const std::vector<uint8_t> packet_bytes(rx_buffer_.begin(),
                                              rx_buffer_.begin() + packet_size);
      if (!dispatch_packet(packet_bytes)) {
        rx_buffer_.erase(rx_buffer_.begin());
      } else {
        rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + packet_size);
      }
    }
  }

  bool dispatch_packet(const std::vector<uint8_t> &packet_bytes) {
    switch (packet_bytes.front()) {
    case 0x5A:
      return receive_vision(packet_bytes);
    case 0x5B:
      return receive_all_robot_hp(packet_bytes);
    case 0x5C:
      return receive_game_status(packet_bytes);
    case 0x5D:
      return receive_robot_status(packet_bytes);
    default:
      return false;
    }
  }

  bool receive_vision(const std::vector<uint8_t> &packet_bytes) {
    const auto packet = from_vector<ReceivePacketVision>(packet_bytes);
    if (!packet_crc_ok(packet)) {
      RCLCPP_ERROR(get_logger(), "CRC error in ReceivePacketVision");
      return false;
    }

    set_detect_color(packet.detect_color);
    if (packet.reset_tracker) {
      reset_tracker();
    }
    publish_gimbal_transform(packet);
    publish_aiming_point(packet);
    return true;
  }

  bool receive_all_robot_hp(const std::vector<uint8_t> &packet_bytes) {
    const auto packet = from_vector<ReceivePacketAllRobotHP>(packet_bytes);
    if (!packet_crc_ok(packet)) {
      RCLCPP_ERROR(get_logger(), "CRC error in ReceivePacketAllRobotHP");
      return false;
    }

    rm_decision_interfaces::msg::AllRobotHP msg;
    msg.red_1_robot_hp = packet.red_1_robot_hp;
    msg.red_2_robot_hp = packet.red_2_robot_hp;
    msg.red_3_robot_hp = packet.red_3_robot_hp;
    msg.red_4_robot_hp = packet.red_4_robot_hp;
    msg.red_5_robot_hp = packet.red_5_robot_hp;
    msg.red_7_robot_hp = packet.red_7_robot_hp;
    msg.red_outpost_hp = packet.red_outpost_hp;
    msg.red_base_hp = packet.red_base_hp;
    msg.blue_1_robot_hp = packet.blue_1_robot_hp;
    msg.blue_2_robot_hp = packet.blue_2_robot_hp;
    msg.blue_3_robot_hp = packet.blue_3_robot_hp;
    msg.blue_4_robot_hp = packet.blue_4_robot_hp;
    msg.blue_5_robot_hp = packet.blue_5_robot_hp;
    msg.blue_7_robot_hp = packet.blue_7_robot_hp;
    msg.blue_outpost_hp = packet.blue_outpost_hp;
    msg.blue_base_hp = packet.blue_base_hp;
    all_robot_hp_pub_->publish(msg);
    return true;
  }

  bool receive_game_status(const std::vector<uint8_t> &packet_bytes) {
    const auto packet = from_vector<ReceivePacketGameStatus>(packet_bytes);
    if (!packet_crc_ok(packet)) {
      RCLCPP_ERROR(get_logger(), "CRC error in ReceivePacketGameStatus");
      return false;
    }

    rm_decision_interfaces::msg::GameStatus msg;
    msg.game_progress = packet.game_progress;
    msg.stage_remain_time = packet.stage_remain_time;
    game_status_pub_->publish(msg);
    return true;
  }

  bool receive_robot_status(const std::vector<uint8_t> &packet_bytes) {
    const auto packet = from_vector<ReceivePacketRobotStatus>(packet_bytes);
    if (!packet_crc_ok(packet)) {
      RCLCPP_ERROR(get_logger(), "CRC error in ReceivePacketRobotStatus");
      return false;
    }

    rm_decision_interfaces::msg::RobotStatus msg;
    msg.robot_id = packet.robot_id;
    msg.current_hp = packet.current_hp;
    msg.shooter_heat = packet.shooter_heat;
    msg.team_color = packet.team_color;
    msg.is_attacked = packet.is_attacked;
    robot_status_pub_->publish(msg);
    return true;
  }

  void set_detect_color(uint8_t detect_color) {
    if (detect_color_set_ && detect_color == previous_detect_color_) {
      return;
    }
    previous_detect_color_ = detect_color;
    detect_color_set_ = true;

    const auto parameter_name =
        get_parameter("detect_color_parameter").as_string();
    for (auto &client : detect_color_clients_) {
      if (!client->service_is_ready()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "Detect color parameter service not ready");
        continue;
      }
      client->set_parameters(
          {rclcpp::Parameter(parameter_name, static_cast<int>(detect_color))});
    }
  }

  void reset_tracker() {
    if (!reset_tracker_client_->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "Tracker reset service not ready");
      return;
    }
    reset_tracker_client_->async_send_request(
        std::make_shared<std_srvs::srv::Trigger::Request>());
    RCLCPP_INFO(get_logger(), "Reset tracker requested from handoff packet");
  }

  void publish_gimbal_transform(const ReceivePacketVision &packet) {
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp =
        now() + rclcpp::Duration::from_seconds(
                    get_parameter("timestamp_offset").as_double());
    transform.header.frame_id =
        get_parameter("gimbal_parent_frame").as_string();
    transform.child_frame_id = get_parameter("gimbal_child_frame").as_string();
    tf2::Quaternion rotation;
    rotation.setRPY(packet.roll, packet.pitch, packet.yaw);
    transform.transform.rotation = tf2::toMsg(rotation);
    tf_broadcaster_->sendTransform(transform);
  }

  void publish_aiming_point(const ReceivePacketVision &packet) {
    if (std::abs(packet.aim_x) <= 0.01F) {
      return;
    }
    aiming_point_.header.stamp = now();
    aiming_point_.pose.position.x = packet.aim_x;
    aiming_point_.pose.position.y = packet.aim_y;
    aiming_point_.pose.position.z = packet.aim_z;
    aiming_point_pub_->publish(aiming_point_);
  }

  void
  send_vision_packet(const auto_aim_interfaces::msg::Target::SharedPtr msg) {
    static const std::map<std::string, uint8_t> kIdMap{
        {"", 0},  {"outpost", 0}, {"1", 1},     {"2", 2},   {"3", 3},
        {"4", 4}, {"5", 5},       {"guard", 6}, {"base", 7}};

    const auto id_it = kIdMap.find(msg->id);
    if (id_it == kIdMap.end()) {
      RCLCPP_ERROR(get_logger(), "Unknown RM handoff target id '%s'",
                   msg->id.c_str());
      return;
    }

    SendPacketVision packet;
    packet.tracking = msg->tracking;
    packet.id = id_it->second;
    packet.armors_num = static_cast<uint8_t>(msg->armors_num);
    packet.x = static_cast<float>(msg->position.x);
    packet.y = static_cast<float>(msg->position.y);
    packet.z = static_cast<float>(msg->position.z);
    packet.yaw = static_cast<float>(msg->yaw);
    packet.vx = static_cast<float>(msg->velocity.x);
    packet.vy = static_cast<float>(msg->velocity.y);
    packet.vz = static_cast<float>(msg->velocity.z);
    packet.v_yaw = static_cast<float>(msg->v_yaw);
    packet.r1 = static_cast<float>(msg->radius_1);
    packet.r2 = static_cast<float>(msg->radius_2);
    packet.dz = static_cast<float>(msg->dz);

    crc16::append_crc16_check_sum(reinterpret_cast<uint8_t *>(&packet),
                                  sizeof(packet));
    send_packet(packet);

    std_msgs::msg::Float64 latency;
    latency.data = (now() - msg->header.stamp).seconds() * 1000.0;
    latency_pub_->publish(latency);
  }

  void send_twist_packet(const geometry_msgs::msg::Twist::SharedPtr msg) {
    SendPacketTwist packet;
    packet.vx = static_cast<float>(msg->linear.x);
    packet.vy = static_cast<float>(msg->linear.y);
    packet.wz = static_cast<float>(msg->angular.z);
    crc8::append_crc8_check_sum(reinterpret_cast<uint8_t *>(&packet),
                                sizeof(packet));
    send_packet(packet);
  }

  void send_robot_control_packet(
      const rm_decision_interfaces::msg::RobotControl::SharedPtr msg) {
    SendPacketRobotControl packet;
    packet.stop_gimbal_scan = msg->stop_gimbal_scan;
    packet.chassis_spin_vel = msg->chassis_spin_vel;
    crc16::append_crc16_check_sum(reinterpret_cast<uint8_t *>(&packet),
                                  sizeof(packet));
    send_packet(packet);
  }

  template <typename PacketT> void send_packet(const PacketT &packet) {
    if (!serial_ || !serial_->connected()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "RM handoff serial is not connected");
      return;
    }
    const auto data = to_vector(packet);
    serial_->send(data.data(), data.size());
  }

  std::unique_ptr<RawSerial> serial_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::vector<std::shared_ptr<rclcpp::AsyncParametersClient>>
      detect_color_clients_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_tracker_client_;

  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr latency_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      aiming_point_pub_;
  rclcpp::Publisher<rm_decision_interfaces::msg::GameStatus>::SharedPtr
      game_status_pub_;
  rclcpp::Publisher<rm_decision_interfaces::msg::RobotStatus>::SharedPtr
      robot_status_pub_;
  rclcpp::Publisher<rm_decision_interfaces::msg::AllRobotHP>::SharedPtr
      all_robot_hp_pub_;
  rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr target_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<rm_decision_interfaces::msg::RobotControl>::SharedPtr
      robot_control_sub_;
  rclcpp::TimerBase::SharedPtr rx_timer_;
  rclcpp::TimerBase::SharedPtr reconnect_timer_;

  visualization_msgs::msg::Marker aiming_point_;
  std::vector<uint8_t> rx_buffer_;
  uint8_t previous_detect_color_{0};
  bool detect_color_set_{false};
};

int run_rm_handoff_serial_bridge(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<yolo_cpp::rm_handoff::RMHandoffSerialBridge>());
  rclcpp::shutdown();
  return 0;
}

} // namespace yolo_cpp::rm_handoff
