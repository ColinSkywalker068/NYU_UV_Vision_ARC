/// @file seasky_protocol.cpp
/// @brief Implements Seasky serial framing, CRCs, flags, and attitude parsing.

#include "yolo_cpp/seasky_protocol.hpp"

#include <algorithm>
#include <cstring>
#include <tuple>

namespace yolo_cpp
{

namespace
{

constexpr uint16_t FLAG_FIRE_MODE_MASK = 0x0003;
constexpr uint16_t FLAG_TARGET_STATE_MASK = 0x0003;
constexpr uint16_t FLAG_TARGET_STATE_SHIFT = 2;
constexpr uint16_t FLAG_TARGET_TYPE_MASK = 0x000F;
constexpr uint16_t FLAG_TARGET_TYPE_SHIFT = 4;
constexpr uint16_t FLAG_ENEMY_COLOR_MASK = 0x0001;
constexpr uint16_t FLAG_WORK_MODE_MASK = 0x0003;
constexpr uint16_t FLAG_WORK_MODE_SHIFT = 1;
constexpr uint16_t FLAG_BULLET_SPEED_MASK = 0x00FF;
constexpr uint16_t FLAG_BULLET_SPEED_SHIFT = 8;
constexpr uint8_t SEASKY_CRC8_INIT = 0xFF;
constexpr uint8_t SEASKY_CRC8_POLY = 0x31;
constexpr uint16_t SEASKY_CRC16_INIT = 0xFFFF;
constexpr uint16_t SEASKY_CRC16_POLY = 0x1021;
constexpr int SEASKY_HEADER_SIZE_BYTES = 4;
constexpr int SEASKY_TAIL_SIZE_BYTES = 2;
constexpr int SEASKY_MIN_DATA_LENGTH = 4;
constexpr int SEASKY_MAX_DATA_LENGTH = 128;
constexpr float DEG_TO_RAD = 0.017453292519943295F;

/// Append a little-endian uint16 field to a byte vector.
void append_u16(std::vector<uint8_t>& data, uint16_t value)
{
  data.push_back(static_cast<uint8_t>(value & 0xFF));
  data.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

/// Read a little-endian uint16 field from a byte vector.
uint16_t read_u16(const std::vector<uint8_t>& data, std::size_t offset)
{
  return static_cast<uint16_t>(data[offset]) | (static_cast<uint16_t>(data[offset + 1]) << 8);
}

/// Decode enemy color, work mode, and bullet speed from an attitude flags register.
std::tuple<int, int, int> parse_attitude_flags(uint16_t flags_register)
{
  const int enemy_color = flags_register & FLAG_ENEMY_COLOR_MASK;
  const int work_mode = (flags_register >> FLAG_WORK_MODE_SHIFT) & FLAG_WORK_MODE_MASK;
  const int bullet_speed = (flags_register >> FLAG_BULLET_SPEED_SHIFT) & FLAG_BULLET_SPEED_MASK;
  return {enemy_color, work_mode, bullet_speed};
}

}  // namespace

uint8_t compute_crc8(const std::vector<uint8_t>& data)
{
  uint8_t crc = SEASKY_CRC8_INIT;
  for (const auto byte_value : data) {
    crc ^= byte_value;
    for (int i = 0; i < 8; ++i) {
      if ((crc & 0x80U) != 0U) {
        crc = static_cast<uint8_t>(((crc << 1U) ^ SEASKY_CRC8_POLY) & 0xFFU);
      } else {
        crc = static_cast<uint8_t>((crc << 1U) & 0xFFU);
      }
    }
  }
  return crc;
}

uint16_t compute_crc16(const std::vector<uint8_t>& data)
{
  uint16_t crc = SEASKY_CRC16_INIT;
  for (const auto byte_value : data) {
    crc ^= static_cast<uint16_t>(byte_value) << 8;
    for (int i = 0; i < 8; ++i) {
      if ((crc & 0x8000U) != 0U) {
        crc = static_cast<uint16_t>(((crc << 1U) ^ SEASKY_CRC16_POLY) & 0xFFFFU);
      } else {
        crc = static_cast<uint16_t>((crc << 1U) & 0xFFFFU);
      }
    }
  }
  return crc;
}

uint16_t build_vision_flags(int fire_mode, int target_state, int target_type)
{
  return static_cast<uint16_t>(
    (fire_mode & FLAG_FIRE_MODE_MASK)
    | ((target_state & FLAG_TARGET_STATE_MASK) << FLAG_TARGET_STATE_SHIFT)
    | ((target_type & FLAG_TARGET_TYPE_MASK) << FLAG_TARGET_TYPE_SHIFT));
}

std::vector<uint8_t> build_seasky_frame(
  uint16_t command_id, uint16_t flags_register, const std::vector<float>& float_payload)
{
  std::vector<uint8_t> payload_bytes;
  payload_bytes.reserve(float_payload.size() * sizeof(float));
  for (const auto value : float_payload) {
    const auto* raw = reinterpret_cast<const uint8_t*>(&value);
    payload_bytes.insert(payload_bytes.end(), raw, raw + sizeof(float));
  }

  const uint16_t data_length = static_cast<uint16_t>(2 + 2 + payload_bytes.size());

  std::vector<uint8_t> header_without_crc{SEASKY_SOF, static_cast<uint8_t>(data_length & 0xFF),
    static_cast<uint8_t>((data_length >> 8) & 0xFF)};

  std::vector<uint8_t> frame = header_without_crc;
  frame.push_back(compute_crc8(header_without_crc));
  append_u16(frame, command_id);
  append_u16(frame, flags_register);
  frame.insert(frame.end(), payload_bytes.begin(), payload_bytes.end());
  append_u16(frame, compute_crc16(frame));
  return frame;
}

std::vector<AttitudeFrame> parse_attitude_frames(std::vector<uint8_t>& rx_buffer)
{
  // Keep incomplete trailing bytes in rx_buffer so the next serial read can complete a frame.
  std::vector<AttitudeFrame> parsed_frames;

  while (true) {
    auto sof_it = std::find(rx_buffer.begin(), rx_buffer.end(), SEASKY_SOF);
    if (sof_it == rx_buffer.end()) {
      rx_buffer.clear();
      break;
    }
    if (sof_it != rx_buffer.begin()) {
      rx_buffer.erase(rx_buffer.begin(), sof_it);
    }
    if (rx_buffer.size() < static_cast<std::size_t>(SEASKY_HEADER_SIZE_BYTES)) {
      break;
    }

    const std::vector<uint8_t> header_no_crc(rx_buffer.begin(), rx_buffer.begin() + 3);
    if (compute_crc8(header_no_crc) != rx_buffer[3]) {
      rx_buffer.erase(rx_buffer.begin());
      continue;
    }

    const uint16_t data_length = read_u16(rx_buffer, 1);
    if (data_length < SEASKY_MIN_DATA_LENGTH || data_length > SEASKY_MAX_DATA_LENGTH) {
      rx_buffer.erase(rx_buffer.begin());
      continue;
    }

    const std::size_t full_frame_size =
      SEASKY_HEADER_SIZE_BYTES + data_length + SEASKY_TAIL_SIZE_BYTES;
    if (rx_buffer.size() < full_frame_size) {
      break;
    }

    std::vector<uint8_t> frame(rx_buffer.begin(), rx_buffer.begin() + full_frame_size);
    const uint16_t received_crc16 = read_u16(frame, frame.size() - 2);
    std::vector<uint8_t> frame_without_crc(frame.begin(), frame.end() - 2);
    if (compute_crc16(frame_without_crc) != received_crc16) {
      rx_buffer.erase(rx_buffer.begin());
      continue;
    }

    const uint16_t command_id = read_u16(frame, 4);
    const uint16_t flags_register = read_u16(frame, 6);
    if (command_id == SEASKY_COMMAND_ATTITUDE_SEND) {
      const std::size_t payload_offset = 8;
      const std::size_t payload_size = frame.size() - payload_offset - 2;
      if (payload_size == 3 * sizeof(float)) {
        AttitudeFrame parsed{};
        std::memcpy(&parsed.yaw, frame.data() + payload_offset, sizeof(float));
        std::memcpy(&parsed.pitch, frame.data() + payload_offset + sizeof(float), sizeof(float));
        std::memcpy(
          &parsed.roll, frame.data() + payload_offset + (2 * sizeof(float)), sizeof(float));
        parsed.yaw *= DEG_TO_RAD;
        parsed.pitch *= DEG_TO_RAD;
        parsed.roll *= DEG_TO_RAD;
        auto [enemy_color, work_mode, bullet_speed] = parse_attitude_flags(flags_register);
        parsed.enemy_color = enemy_color;
        parsed.work_mode = work_mode;
        parsed.bullet_speed = bullet_speed;
        parsed.flags_register = flags_register;
        parsed.command_id = command_id;
        parsed_frames.push_back(parsed);
      }
    }

    rx_buffer.erase(rx_buffer.begin(), rx_buffer.begin() + full_frame_size);
  }

  return parsed_frames;
}

}  // namespace yolo_cpp
