#ifndef YOLO_CPP__RM_HANDOFF_PACKET_HPP_
#define YOLO_CPP__RM_HANDOFF_PACKET_HPP_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace yolo_cpp::rm_handoff {

struct ReceivePacketVision {
  uint8_t header = 0x5A;
  uint8_t detect_color : 1; // 0-red 1-blue
  bool reset_tracker : 1;
  uint8_t reserved : 6;
  float roll;
  float pitch;
  float yaw;
  float aim_x;
  float aim_y;
  float aim_z;
  uint16_t checksum = 0;
} __attribute__((packed));

struct SendPacketVision {
  uint8_t header = 0xA5;
  bool tracking : 1;
  uint8_t id : 3;         // 0-outpost 6-guard 7-base
  uint8_t armors_num : 3; // 2-balance 3-outpost 4-normal
  uint8_t reserved : 1;
  float x;
  float y;
  float z;
  float yaw;
  float vx;
  float vy;
  float vz;
  float v_yaw;
  float r1;
  float r2;
  float dz;
  uint16_t checksum = 0;
} __attribute__((packed));

struct SendPacketTwist {
  uint8_t header1 = 0xA5;
  uint8_t header2 = 0x5A;
  float vx; // linear.x
  float vy; // linear.y
  float wz; // angular.z
  uint8_t crc8 = 0;
} __attribute__((packed));

struct SendPacketRobotControl {
  uint8_t header = 0xA3;
  bool stop_gimbal_scan;
  float chassis_spin_vel;
  uint16_t checksum = 0;
} __attribute__((packed));

struct ReceivePacketAllRobotHP {
  uint8_t header = 0x5B;
  uint16_t red_1_robot_hp;
  uint16_t red_2_robot_hp;
  uint16_t red_3_robot_hp;
  uint16_t red_4_robot_hp;
  uint16_t red_5_robot_hp;
  uint16_t red_7_robot_hp;
  uint16_t red_outpost_hp;
  uint16_t red_base_hp;
  uint16_t blue_1_robot_hp;
  uint16_t blue_2_robot_hp;
  uint16_t blue_3_robot_hp;
  uint16_t blue_4_robot_hp;
  uint16_t blue_5_robot_hp;
  uint16_t blue_7_robot_hp;
  uint16_t blue_outpost_hp;
  uint16_t blue_base_hp;
  uint16_t checksum = 0;
} __attribute__((packed));

struct ReceivePacketGameStatus {
  uint8_t header = 0x5C;
  uint8_t game_progress;
  uint16_t stage_remain_time;
  uint16_t checksum = 0;
} __attribute__((packed));

struct ReceivePacketRobotStatus {
  uint8_t header = 0x5D;
  uint8_t robot_id;
  uint16_t current_hp;
  uint16_t shooter_heat;
  bool team_color;
  bool is_attacked;
  uint16_t checksum = 0;
} __attribute__((packed));

template <typename T> inline T from_vector(const std::vector<uint8_t> &data) {
  T packet;
  std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t *>(&packet));
  return packet;
}

template <typename T> inline std::vector<uint8_t> to_vector(const T &data) {
  std::vector<uint8_t> packet(sizeof(T));
  std::copy(reinterpret_cast<const uint8_t *>(&data),
            reinterpret_cast<const uint8_t *>(&data) + sizeof(T),
            packet.begin());
  return packet;
}

static_assert(sizeof(ReceivePacketVision) == 28);
static_assert(sizeof(SendPacketVision) == 48);
static_assert(sizeof(SendPacketTwist) == 15);
static_assert(sizeof(SendPacketRobotControl) == 8);
static_assert(sizeof(ReceivePacketAllRobotHP) == 35);
static_assert(sizeof(ReceivePacketGameStatus) == 6);
static_assert(sizeof(ReceivePacketRobotStatus) == 10);

} // namespace yolo_cpp::rm_handoff

#endif // YOLO_CPP__RM_HANDOFF_PACKET_HPP_
