#include <gtest/gtest.h>

#include "yolo_cpp/utils.hpp"

TEST(AngleGainTest, ScalesAnglesAboveUnityGain)
{
  const auto [yaw, pitch] = yolo_cpp::apply_angle_gain(0.1F, -0.2F, 1.35F);
  EXPECT_NEAR(yaw, 0.135F, 1e-6F);
  EXPECT_NEAR(pitch, -0.27F, 1e-6F);
}

TEST(AngleGainTest, ClampsGainAtUnityMinimum)
{
  const auto [yaw, pitch] = yolo_cpp::apply_angle_gain(0.3F, -0.4F, 0.5F);
  EXPECT_NEAR(yaw, 0.3F, 1e-6F);
  EXPECT_NEAR(pitch, -0.4F, 1e-6F);
}
