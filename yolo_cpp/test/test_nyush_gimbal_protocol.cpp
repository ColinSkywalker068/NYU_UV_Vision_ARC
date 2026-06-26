#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "yolo_cpp/nyush_gimbal_protocol.hpp"

namespace {

using namespace yolo_cpp::nyush_gimbal;

TEST(NYUSHGimbalProtocol, PacketSizesMatchUpstreamLayout) {
  EXPECT_EQ(sizeof(GimbalToVision), 43U);
  EXPECT_EQ(sizeof(VisionToGimbal), 29U);
}

TEST(NYUSHGimbalProtocol, BuildsVisionPacketWithModeAndCrc) {
  const auto packet = build_vision_packet(true, true, 0.25F, -0.125F);
  const auto bytes = to_bytes(packet);

  ASSERT_EQ(bytes.size(), sizeof(VisionToGimbal));
  EXPECT_EQ(bytes[0], 'S');
  EXPECT_EQ(bytes[1], 'P');
  EXPECT_EQ(bytes[2], NYUSH_GIMBAL_MODE_AIM_AND_FIRE);
  EXPECT_TRUE(verify_crc16(bytes.data(), bytes.size()));
}

TEST(NYUSHGimbalProtocol,
     ParsesGimbalFeedbackAndLeavesIncompleteBytesBuffered) {
  GimbalToVision packet;
  packet.mode = NYUSH_GIMBAL_MODE_AIM;
  packet.q[0] = 1.0F;
  packet.q[1] = 0.0F;
  packet.q[2] = 0.0F;
  packet.q[3] = 0.0F;
  packet.yaw = 0.5F;
  packet.pitch = -0.25F;
  packet.bullet_speed = 22.0F;
  packet.bullet_count = 7;
  append_crc16(reinterpret_cast<uint8_t *>(&packet), sizeof(packet));

  std::vector<uint8_t> buffer{0x00, 0x01};
  const auto *raw = reinterpret_cast<const uint8_t *>(&packet);
  buffer.insert(buffer.end(), raw, raw + sizeof(packet));
  buffer.push_back('S');

  const auto parsed = parse_gimbal_packets(buffer);
  ASSERT_EQ(parsed.size(), 1U);
  EXPECT_FLOAT_EQ(parsed[0].yaw, 0.5F);
  EXPECT_FLOAT_EQ(parsed[0].pitch, -0.25F);
  EXPECT_FLOAT_EQ(parsed[0].bullet_speed, 22.0F);
  EXPECT_EQ(parsed[0].bullet_count, 7);
  EXPECT_EQ(parsed[0].mode, NYUSH_GIMBAL_MODE_AIM);
  EXPECT_EQ(buffer.size(), 1U);
  EXPECT_EQ(buffer[0], 'S');
}

TEST(NYUSHGimbalProtocol, RejectsBadCrc) {
  GimbalToVision packet;
  append_crc16(reinterpret_cast<uint8_t *>(&packet), sizeof(packet));
  auto *raw = reinterpret_cast<uint8_t *>(&packet);
  raw[5] ^= 0x01;

  std::vector<uint8_t> buffer(raw, raw + sizeof(packet));
  const auto parsed = parse_gimbal_packets(buffer);
  EXPECT_TRUE(parsed.empty());
  EXPECT_TRUE(buffer.empty());
}

} // namespace
