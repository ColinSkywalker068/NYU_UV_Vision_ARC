#ifndef YOLO_CPP__SEASKY_PROTOCOL_HPP_
#define YOLO_CPP__SEASKY_PROTOCOL_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace yolo_cpp
{

constexpr uint8_t SEASKY_SOF = 0xA5;
constexpr uint16_t SEASKY_COMMAND_VISION_RECV = 0x0001;
constexpr uint16_t SEASKY_COMMAND_ATTITUDE_SEND = 0x0002;
constexpr int DEFAULT_FIRE_MODE = 2;
constexpr int DEFAULT_TARGET_TYPE = 0;
constexpr int TARGET_STATE_NO_TARGET = 0;
constexpr int TARGET_STATE_CONVERGING = 1;
constexpr double DEFAULT_RX_POLL_PERIOD_SECONDS = 0.01;

struct AttitudeFrame
{
  float yaw{0.0F};
  float pitch{0.0F};
  float roll{0.0F};
  int enemy_color{0};
  int work_mode{0};
  int bullet_speed{0};
  int flags_register{0};
  int command_id{0};
};

uint8_t compute_crc8(const std::vector<uint8_t> & data);
uint16_t compute_crc16(const std::vector<uint8_t> & data);
uint16_t build_vision_flags(int fire_mode, int target_state, int target_type);
std::vector<uint8_t> build_seasky_frame(
  uint16_t command_id, uint16_t flags_register, const std::vector<float> & float_payload);
std::vector<AttitudeFrame> parse_attitude_frames(std::vector<uint8_t> & rx_buffer);

}  // namespace yolo_cpp

#endif
