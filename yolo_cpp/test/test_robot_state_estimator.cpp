#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "yolo_cpp/robot_state_estimator.hpp"

namespace
{
using yolo_cpp::RobotPlateObservation;
using yolo_cpp::RobotStateEstimator;
using yolo_cpp::Vector3d;

RobotPlateObservation plate(
  double center_x, double center_z, double phase, int slot, int track_id, double radius = 0.26,
  int count = 4)
{
  const double angle = phase + 2.0 * M_PI * static_cast<double>(slot) / count;
  const Vector3d normal{std::sin(angle), 0.0, -std::cos(angle)};
  return {{center_x + radius * normal.x, 0.0, center_z + radius * normal.z}, normal, track_id};
}

TEST(RobotStateEstimator, SingleStationaryPlateKeepsDirectFallback)
{
  RobotStateEstimator estimator;
  for (int frame = 0; frame < 120; ++frame) {
    // Simulate occasional tracker ID churn; it is not sufficient evidence to
    // apply a robot radius.
    estimator.update({plate(0.2, 4.0, 0.0, 0, 7 + frame / 10)}, frame / 60.0);
  }
  EXPECT_TRUE(estimator.state().initialized);
  EXPECT_FALSE(estimator.state().model_confirmed);
  EXPECT_NEAR(estimator.state().center.x, 0.2, 1e-6);
  EXPECT_NEAR(estimator.state().center.z, 4.0, 1e-6);
  EXPECT_NEAR(estimator.state().angular_velocity_rad_s, 0.0, 1e-6);
}

TEST(RobotStateEstimator, NonSpinningRobotAcceptsMultiplePlates)
{
  RobotStateEstimator estimator;
  for (int frame = 0; frame < 120; ++frame) {
    const double t = frame / 60.0;
    const double x = -0.4 + 0.3 * t;
    estimator.update({plate(x, 4.0, 0.0, 0, 1), plate(x, 4.0, 0.0, 1, 2)}, t);
  }
  EXPECT_TRUE(estimator.state().model_confirmed);
  EXPECT_FALSE(estimator.state().rotating);
  EXPECT_NEAR(estimator.state().center.x, -0.4 + 0.3 * 119.0 / 60.0, 0.015);
  EXPECT_NEAR(estimator.state().velocity.x, 0.3, 0.03);
}

TEST(RobotStateEstimator, SpinSurvivesVisiblePlateSwitches)
{
  RobotStateEstimator estimator;
  constexpr double omega = 6.0;
  for (int frame = 0; frame < 240; ++frame) {
    const double t = frame / 120.0;
    const double phase = omega * t;
    const int slot = static_cast<int>(std::floor((phase + M_PI / 4.0) / (M_PI / 2.0))) & 3;
    estimator.update({plate(0.3, 4.5, phase, slot, slot + 10)}, t);
  }
  EXPECT_TRUE(estimator.state().model_confirmed);
  EXPECT_TRUE(estimator.state().rotating);
  EXPECT_NEAR(estimator.state().center.x, 0.3, 1e-5);
  EXPECT_NEAR(estimator.state().center.z, 4.5, 1e-5);
  EXPECT_NEAR(estimator.state().angular_velocity_rad_s, omega, 0.8);
}

TEST(RobotStateEstimator, CoastAndDistractorDoNotMoveRobotTrack)
{
  RobotStateEstimator estimator;
  for (int frame = 0; frame < 40; ++frame) {
    const double t = frame / 60.0;
    estimator.update({plate(0.0, 4.0, 2.0 * t, 0, 1), plate(0.0, 4.0, 2.0 * t, 1, 2)}, t);
  }
  const auto before = estimator.state();
  estimator.update({plate(2.0, 7.0, 0.0, 0, 99)}, 40.0 / 60.0);
  EXPECT_GT(estimator.state().rejected_observations, before.rejected_observations);
  EXPECT_NEAR(estimator.state().center.x, before.center.x, 0.05);
  EXPECT_NEAR(estimator.state().center.z, before.center.z, 0.05);
  for (int frame = 41; frame < 46; ++frame) {
    estimator.update({}, frame / 60.0);
  }
  EXPECT_TRUE(estimator.state().model_confirmed);
  EXPECT_EQ(estimator.state().coast_frames, 6);
}

TEST(RobotStateEstimator, NearbyCrossingRobotDoesNotMergeCenters)
{
  RobotStateEstimator estimator;
  for (int frame = 0; frame < 40; ++frame) {
    const double t = frame / 60.0;
    estimator.update({plate(0.0, 4.0, 2.0 * t, 0, 1), plate(0.0, 4.0, 2.0 * t, 1, 2)}, t);
  }
  const auto before = estimator.state();
  const double next_t = 40.0 / 60.0;
  estimator.update({plate(0.0, 4.0, 2.0 * next_t, 0, 1), plate(0.0, 4.35, 0.0, 0, 90)}, next_t);
  EXPECT_GT(estimator.state().rejected_observations, before.rejected_observations);
  EXPECT_NEAR(estimator.state().center.x, 0.0, 0.02);
  EXPECT_NEAR(estimator.state().center.z, 4.0, 0.02);
}

TEST(RobotStateEstimator, SpinStopReverseChangesRateWithoutReset)
{
  RobotStateEstimator estimator;
  double phase = 0.0;
  int previous_slot = -1;
  int observed_switches = 0;
  for (int frame = 0; frame < 360; ++frame) {
    const double t = frame / 120.0;
    const double omega = t < 1.0 ? 5.0 : (t < 1.6 ? 0.0 : -4.0);
    phase += omega / 120.0;
    const int slot = static_cast<int>(std::floor((phase + M_PI / 4.0) / (M_PI / 2.0))) & 3;
    if (previous_slot >= 0 && slot != previous_slot) {
      ++observed_switches;
    }
    previous_slot = slot;
    estimator.update({plate(-0.2, 4.2, phase, slot, slot + 20)}, t);
  }
  EXPECT_GT(observed_switches, 3);
  EXPECT_TRUE(estimator.state().model_confirmed);
  EXPECT_TRUE(estimator.state().rotating);
  EXPECT_LT(estimator.state().angular_velocity_rad_s, -3.0);
  EXPECT_NEAR(estimator.state().center.x, -0.2, 1e-5);
  EXPECT_NEAR(estimator.state().center.z, 4.2, 1e-5);
}

TEST(RobotStateEstimator, NoiseAndDropoutKeepCenterBounded)
{
  RobotStateEstimator estimator;
  std::mt19937 generator(1234);
  std::normal_distribution<double> position_noise(0.0, 0.008);
  std::normal_distribution<double> normal_noise(0.0, 0.01);
  double squared_error_sum = 0.0;
  int measured_frames = 0;
  for (int frame = 0; frame < 300; ++frame) {
    const double t = frame / 60.0;
    const double center_x = -0.5 + 0.2 * t;
    if (frame % 7 == 0 || frame % 7 == 1) {
      estimator.update({}, t);
      continue;
    }
    auto first = plate(center_x, 4.5, 3.0 * t, 0, 1);
    auto second = plate(center_x, 4.5, 3.0 * t, 1, 2);
    for (auto * observation : {&first, &second}) {
      observation->position.x += position_noise(generator);
      observation->position.z += position_noise(generator);
      observation->outward_normal.x += normal_noise(generator);
      observation->outward_normal.z += normal_noise(generator);
    }
    estimator.update({first, second}, t);
    const double error_x = estimator.state().center.x - center_x;
    const double error_z = estimator.state().center.z - 4.5;
    squared_error_sum += error_x * error_x + error_z * error_z;
    ++measured_frames;
  }
  const double rmse = std::sqrt(squared_error_sum / measured_frames);
  EXPECT_TRUE(estimator.state().model_confirmed);
  EXPECT_LT(rmse, 0.03);
  EXPECT_NEAR(estimator.state().velocity.x, 0.2, 0.08);
}

TEST(RobotStateEstimator, FutureArmorUsesPredictedCenterAndPhase)
{
  yolo_cpp::RobotEstimate state;
  state.initialized = true;
  state.center = {0.0, 0.0, 4.0};
  state.velocity = {1.0, 0.0, 0.0};
  state.phase_rad = 0.0;
  state.angular_velocity_rad_s = M_PI;
  const auto armor = yolo_cpp::select_future_armor(state, 0.5, 0.25, 4);
  EXPECT_NEAR(armor.position.x, 0.5, 1e-6);
  EXPECT_NEAR(armor.position.z, 3.75, 1e-6);
}
}  // namespace
