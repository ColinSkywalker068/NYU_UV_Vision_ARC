#ifndef YOLO_CPP__NYUSH_GIMBAL_PROTOCOL_HPP_
#define YOLO_CPP__NYUSH_GIMBAL_PROTOCOL_HPP_

#include <cstdint>
#include <vector>

namespace yolo_cpp::nyush_gimbal {

constexpr uint8_t NYUSH_GIMBAL_HEADER_0 = 'S';
constexpr uint8_t NYUSH_GIMBAL_HEADER_1 = 'P';
constexpr uint8_t NYUSH_GIMBAL_MODE_IDLE = 0;
constexpr uint8_t NYUSH_GIMBAL_MODE_AIM = 1;
constexpr uint8_t NYUSH_GIMBAL_MODE_AIM_AND_FIRE = 2;
constexpr double DEFAULT_RX_POLL_PERIOD_SECONDS = 0.005;

struct GimbalToVision {
  uint8_t head[2] = {NYUSH_GIMBAL_HEADER_0, NYUSH_GIMBAL_HEADER_1};
  uint8_t mode{0};
  float q[4]{1.0F, 0.0F, 0.0F, 0.0F};
  float yaw{0.0F};
  float yaw_vel{0.0F};
  float pitch{0.0F};
  float pitch_vel{0.0F};
  float bullet_speed{0.0F};
  uint16_t bullet_count{0};
  uint16_t crc16{0};
} __attribute__((packed));

struct VisionToGimbal {
  uint8_t head[2] = {NYUSH_GIMBAL_HEADER_0, NYUSH_GIMBAL_HEADER_1};
  uint8_t mode{0};
  float yaw{0.0F};
  float yaw_vel{0.0F};
  float yaw_acc{0.0F};
  float pitch{0.0F};
  float pitch_vel{0.0F};
  float pitch_acc{0.0F};
  uint16_t crc16{0};
} __attribute__((packed));

struct AttitudePacket {
  float yaw{0.0F};
  float pitch{0.0F};
  float roll{0.0F};
  float bullet_speed{0.0F};
  uint16_t bullet_count{0};
  uint8_t mode{0};
};

static_assert(sizeof(GimbalToVision) == 43);
static_assert(sizeof(VisionToGimbal) == 29);

uint16_t compute_crc16(const uint8_t *data, uint32_t len);
bool verify_crc16(const uint8_t *data, uint32_t len);
void append_crc16(uint8_t *data, uint32_t len);

VisionToGimbal build_vision_packet(bool control, bool fire, float yaw,
                                   float pitch, float yaw_vel = 0.0F,
                                   float pitch_vel = 0.0F, float yaw_acc = 0.0F,
                                   float pitch_acc = 0.0F);

std::vector<uint8_t> to_bytes(const VisionToGimbal &packet);
std::vector<AttitudePacket>
parse_gimbal_packets(std::vector<uint8_t> &rx_buffer);

} // namespace yolo_cpp::nyush_gimbal

#endif // YOLO_CPP__NYUSH_GIMBAL_PROTOCOL_HPP_
