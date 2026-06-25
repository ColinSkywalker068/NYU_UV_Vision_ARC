#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "yolo_cpp/rm_handoff_crc.hpp"
#include "yolo_cpp/rm_handoff_packet.hpp"

namespace {

using namespace yolo_cpp::rm_handoff;

TEST(RMHandoffProtocol, PacketSizesMatchHandoffLayout) {
  EXPECT_EQ(sizeof(ReceivePacketVision), 28U);
  EXPECT_EQ(sizeof(SendPacketVision), 48U);
  EXPECT_EQ(sizeof(SendPacketTwist), 15U);
  EXPECT_EQ(sizeof(SendPacketRobotControl), 8U);
  EXPECT_EQ(sizeof(ReceivePacketAllRobotHP), 35U);
  EXPECT_EQ(sizeof(ReceivePacketGameStatus), 6U);
  EXPECT_EQ(sizeof(ReceivePacketRobotStatus), 10U);
}

TEST(RMHandoffProtocol, Crc8UsesUpstreamPolynomial) {
  const std::vector<uint8_t> data{'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  EXPECT_EQ(crc8::get_crc8_check_sum(data.data(), data.size()), 0xF4);
}

TEST(RMHandoffProtocol, AppendsAndVerifiesCrc16Packets) {
  SendPacketVision packet;
  packet.tracking = true;
  packet.id = 6;
  packet.armors_num = 4;
  packet.x = 1.0F;
  packet.y = 2.0F;
  packet.z = 3.0F;
  packet.yaw = 0.25F;
  packet.vx = 0.1F;
  packet.vy = 0.2F;
  packet.vz = 0.3F;
  packet.v_yaw = 0.4F;
  packet.r1 = 0.5F;
  packet.r2 = 0.6F;
  packet.dz = 0.7F;

  crc16::append_crc16_check_sum(reinterpret_cast<uint8_t *>(&packet),
                                sizeof(packet));

  EXPECT_TRUE(crc16::verify_crc16_check_sum(
      reinterpret_cast<const uint8_t *>(&packet), sizeof(packet)));

  auto bytes = to_vector(packet);
  ASSERT_EQ(bytes.size(), sizeof(SendPacketVision));
  EXPECT_EQ(bytes[0], 0xA5);
  bytes[4] ^= 0x01;
  EXPECT_FALSE(crc16::verify_crc16_check_sum(bytes.data(), bytes.size()));
}

TEST(RMHandoffProtocol, TwistPacketSerializesWithCrc8) {
  SendPacketTwist packet;
  packet.vx = 1.5F;
  packet.vy = -2.5F;
  packet.wz = 3.5F;

  crc8::append_crc8_check_sum(reinterpret_cast<uint8_t *>(&packet),
                              sizeof(packet));
  const auto bytes = to_vector(packet);

  ASSERT_EQ(bytes.size(), sizeof(SendPacketTwist));
  EXPECT_EQ(bytes[0], 0xA5);
  EXPECT_EQ(bytes[1], 0x5A);
  EXPECT_EQ(bytes.back(),
            crc8::get_crc8_check_sum(bytes.data(), bytes.size() - 1));
}

} // namespace
