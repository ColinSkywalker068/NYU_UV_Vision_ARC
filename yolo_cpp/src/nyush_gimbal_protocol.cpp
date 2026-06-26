#include "yolo_cpp/nyush_gimbal_protocol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "yolo_cpp/rm_handoff_crc.hpp"

namespace yolo_cpp::nyush_gimbal {
namespace {

float quaternion_roll(const float q[4]) {
  const float w = q[0];
  const float x = q[1];
  const float y = q[2];
  const float z = q[3];
  const float sinr_cosp = 2.0F * ((w * x) + (y * z));
  const float cosr_cosp = 1.0F - (2.0F * ((x * x) + (y * y)));
  return std::atan2(sinr_cosp, cosr_cosp);
}

template <typename PacketT>
PacketT from_bytes(const std::vector<uint8_t> &data) {
  PacketT packet;
  std::memcpy(&packet, data.data(), std::min(data.size(), sizeof(PacketT)));
  return packet;
}

} // namespace

uint16_t compute_crc16(const uint8_t *data, uint32_t len) {
  return rm_handoff::crc16::get_crc16_check_sum(data, len, 0xFFFF);
}

bool verify_crc16(const uint8_t *data, uint32_t len) {
  return rm_handoff::crc16::verify_crc16_check_sum(data, len);
}

void append_crc16(uint8_t *data, uint32_t len) {
  rm_handoff::crc16::append_crc16_check_sum(data, len);
}

VisionToGimbal build_vision_packet(bool control, bool fire, float yaw,
                                   float pitch, float yaw_vel, float pitch_vel,
                                   float yaw_acc, float pitch_acc) {
  VisionToGimbal packet;
  packet.mode =
      control ? (fire ? NYUSH_GIMBAL_MODE_AIM_AND_FIRE : NYUSH_GIMBAL_MODE_AIM)
              : NYUSH_GIMBAL_MODE_IDLE;
  packet.yaw = yaw;
  packet.yaw_vel = yaw_vel;
  packet.yaw_acc = yaw_acc;
  packet.pitch = pitch;
  packet.pitch_vel = pitch_vel;
  packet.pitch_acc = pitch_acc;
  append_crc16(reinterpret_cast<uint8_t *>(&packet), sizeof(packet));
  return packet;
}

std::vector<uint8_t> to_bytes(const VisionToGimbal &packet) {
  const auto *begin = reinterpret_cast<const uint8_t *>(&packet);
  return std::vector<uint8_t>(begin, begin + sizeof(packet));
}

std::vector<AttitudePacket>
parse_gimbal_packets(std::vector<uint8_t> &rx_buffer) {
  std::vector<AttitudePacket> parsed_packets;
  const std::array<uint8_t, 2> header{NYUSH_GIMBAL_HEADER_0,
                                      NYUSH_GIMBAL_HEADER_1};

  while (true) {
    const auto header_it = std::search(rx_buffer.begin(), rx_buffer.end(),
                                       header.begin(), header.end());
    if (header_it == rx_buffer.end()) {
      if (!rx_buffer.empty() && rx_buffer.back() == NYUSH_GIMBAL_HEADER_0) {
        rx_buffer.erase(rx_buffer.begin(), rx_buffer.end() - 1);
        break;
      }
      rx_buffer.clear();
      break;
    }
    if (header_it != rx_buffer.begin()) {
      rx_buffer.erase(rx_buffer.begin(), header_it);
    }
    if (rx_buffer.size() < sizeof(GimbalToVision)) {
      break;
    }

    const std::vector<uint8_t> packet_bytes(
        rx_buffer.begin(), rx_buffer.begin() + sizeof(GimbalToVision));
    if (!verify_crc16(packet_bytes.data(), packet_bytes.size())) {
      rx_buffer.erase(rx_buffer.begin());
      continue;
    }

    const auto packet = from_bytes<GimbalToVision>(packet_bytes);
    parsed_packets.push_back(
        AttitudePacket{packet.yaw, packet.pitch, quaternion_roll(packet.q),
                       packet.bullet_speed, packet.bullet_count, packet.mode});
    rx_buffer.erase(rx_buffer.begin(),
                    rx_buffer.begin() + sizeof(GimbalToVision));
  }

  return parsed_packets;
}

} // namespace yolo_cpp::nyush_gimbal
