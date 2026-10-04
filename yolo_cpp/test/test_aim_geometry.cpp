#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "yolo_cpp/aim_geometry.hpp"
#include "yolo_cpp/aim_prediction.hpp"
#include "yolo_cpp/preview_geometry.hpp"

using namespace yolo_cpp;

TEST(AttitudeHistory, RejectsUnknownFutureStaleAndOutOfOrderTime)
{
  AttitudeHistory history;
  EXPECT_FALSE(history.at(0, 1000000000));
  ASSERT_TRUE(history.insert({1000000000, 0.1, 0.2, 0.3}));
  EXPECT_FALSE(history.insert({990000000, 1, 1, 1}));
  EXPECT_FALSE(history.at(1000000001, 1000000000));
  EXPECT_FALSE(history.latest(1300000000));
  ASSERT_TRUE(history.latest(1000000000));
  EXPECT_DOUBLE_EQ(history.latest(1000000000)->yaw, 0.1);
  EXPECT_FALSE(history.insert({1010000000, std::numeric_limits<double>::quiet_NaN(), 0, 0}));
  EXPECT_FALSE(history.latest(1010000000));
}

TEST(AttitudeHistory, InterpolatesAcrossAngleWrapButNotTelemetryOutage)
{
  AttitudeHistory history;
  history.insert({1000000000, 3.13, 0, 0});
  history.insert({1020000000, -3.13, 0.2, 0.4});
  const auto mid = history.at(1010000000, 1020000000);
  ASSERT_TRUE(mid);
  EXPECT_NEAR(std::abs(mid->yaw), 3.141592653589793, 1e-12);
  EXPECT_NEAR(mid->pitch, 0.1, 1e-12);
  history.insert({1200000000, 0, 0, 0});
  EXPECT_FALSE(history.at(1100000000, 1200000000));
  EXPECT_FALSE(history.at(1250000000, 1250000000));
}

TEST(AimGeometry, StationaryPlateAndNormalStayFixedDuringPanPitchAndRoll)
{
  const cv::Point3d fixed(1.3, -0.7, 4.2);
  const auto plate_rotation = camera_to_inertial_rotation({1, 0.45, 0.15, -0.1});
  for (int i = 0; i < 1000; ++i) {
    const double t = i / 1000.0;
    const auto camera_rotation = camera_to_inertial_rotation({1, -0.5 + t, 0.2 * t, -0.3 * t});
    const cv::Point3d offset(0.04, -0.02, 0.12);
    const auto camera_position = rotate_point(camera_rotation, offset);
    const auto observed = rotate_point(camera_rotation.t(), fixed - camera_position);
    const auto restored = rotate_point(camera_rotation, observed) + camera_position;
    EXPECT_NEAR(cv::norm(restored - fixed), 0, 1e-12);
    const auto q = rotation_quaternion(camera_rotation.t() * plate_rotation);
    const auto restored_orientation = camera_rotation * quaternion_rotation(q);
    EXPECT_NEAR(cv::norm(cv::Mat(restored_orientation - plate_rotation)), 0, 1e-12);
  }
  const auto right = rotate_point(camera_to_inertial_rotation({1, 0.2, 0, 0}), {0, 0, 1});
  const auto up = rotate_point(camera_to_inertial_rotation({1, 0, 0.2, 0}), {0, 0, 1});
  EXPECT_GT(right.x, 0);
  EXPECT_LT(up.y, 0);
}

static AimPredictionInput input_at_4m()
{
  AimPredictionInput in;
  in.frame_stamp_ns = in.measurement_stamp_ns = 1000000000;
  in.now_ns = 1050000000;
  in.position = {0, 0, 4};
  return in;
}

TEST(AimPrediction, InertialPositionIsNotRotatedTwice)
{
  auto in = input_at_4m();
  in.position = {1, -0.2, 4};
  AimPredictionConfig config;
  config.compensate_gravity = false;
  const auto first = predict_aim(in, config);
  ASSERT_TRUE(first.valid);
  in.current_attitude = {in.now_ns, 0.2, 0.1, -0.2};
  const auto panned = predict_aim(in, config);
  ASSERT_TRUE(panned.valid);
  EXPECT_DOUBLE_EQ(first.yaw, panned.yaw);
  EXPECT_DOUBLE_EQ(first.pitch, panned.pitch);
  EXPECT_GT(panned.yaw, 0);
  EXPECT_GT(panned.pitch, 0);
}

TEST(AimPrediction, UsesMeasuredFrameAgeAndDoesNotRefreshCoastedMeasurement)
{
  auto in = input_at_4m();
  in.velocity = {1, 0, 0};
  in.has_velocity = true;
  AimPredictionConfig config;
  config.compensate_gravity = false;
  config.transport_delay_s = 0.01;
  config.actuator_delay_s = 0.03;
  const auto out = predict_aim(in, config);
  ASSERT_TRUE(out.valid);
  EXPECT_NEAR(out.horizon_s, 0.05 + 0.01 + 0.03 + out.flight_time_s, 1e-6);
  EXPECT_NEAR(std::tan(out.yaw) * 4, out.horizon_s, 1e-6);
  in.now_ns = 1200000000;
  in.frame_stamp_ns = 1190000000;  // Fresh prediction of an old observation.
  EXPECT_FALSE(predict_aim(in, config).valid);
  in.measurement_stamp_ns = 1210000000;
  EXPECT_FALSE(predict_aim(in, config).valid);
}

TEST(AimPrediction, ReportsSentryMotionLimitsAndPlateHysteresis)
{
  EXPECT_NEAR(minimum_slew_time(0.1, 2, 4), 2 * std::sqrt(0.1 / 4), 1e-12);
  EXPECT_NEAR(minimum_slew_time(3, 2, 4), 2, 1e-12);
  auto in = input_at_4m();
  in.has_robot_state = true;
  in.robot.center = {0, 0, 4};
  in.robot.phase_rad = 0.78539816339;
  in.previous_slot = 3;
  AimPredictionConfig config;
  config.compensate_gravity = false;
  const auto out = predict_aim(in, config);
  ASSERT_TRUE(out.valid);
  EXPECT_EQ(out.selected_slot, 3);
  EXPECT_FALSE(out.reachable);
  EXPECT_GT(out.minimum_slew_s, 0);
  in.robot.phase_rad = 0;
  EXPECT_EQ(predict_aim(in, config).selected_slot, 0);
}

TEST(AimPrediction, InvalidConfigurationAndRobotStateNeverProduceCommands)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  auto in = input_at_4m();
  for (double invalid : {nan, -1.0, 0.0}) {
    AimPredictionConfig c;
    c.max_prediction_s = invalid;
    EXPECT_FALSE(predict_aim(in, c).valid);
    c = AimPredictionConfig();
    c.yaw_accel_rad_s2 = invalid;
    EXPECT_FALSE(predict_aim(in, c).valid);
    c = AimPredictionConfig();
    c.max_measurement_age_s = invalid;
    EXPECT_FALSE(predict_aim(in, c).valid);
  }
  in.has_robot_state = true;
  in.robot.center = {0, 0, 4};
  in.robot.phase_rad = nan;
  EXPECT_FALSE(predict_aim(in, {}).valid);
  in.robot.phase_rad = 0;
  in.robot.center.x = nan;
  EXPECT_FALSE(predict_aim(in, {}).valid);
  in.robot.center.x = 0;
  in.robot.alternating_radius_m = -.4;
  EXPECT_FALSE(predict_aim(in, {}).valid);
  EXPECT_TRUE(std::isinf(minimum_slew_time(.1, nan, 4)));
}

TEST(PreviewFreshness, RejectsQueuedStaleFutureAndMissingSamples)
{
  EXPECT_TRUE(preview_sample_fresh(1000000000, 1010000000));
  EXPECT_TRUE(preview_sample_fresh(1000000000, 1250000000));
  EXPECT_FALSE(preview_sample_fresh(1000000000, 1250000001));
  EXPECT_FALSE(preview_sample_fresh(1000000001, 1000000000));
  EXPECT_FALSE(preview_sample_fresh(0, 0));
}

namespace
{
void advance_rate_input(AimPredictionInput & in, double seconds = .02)
{
  const auto delta = static_cast<std::int64_t>(std::llround(seconds * 1e9));
  in.frame_stamp_ns += delta;
  in.measurement_stamp_ns += delta;
  in.now_ns += delta;
  in.position += in.velocity * seconds;
  in.robot.center.x += in.robot.velocity.x * seconds;
  in.robot.center.y += in.robot.velocity.y * seconds;
  in.robot.center.z += in.robot.velocity.z * seconds;
  in.robot.phase_rad += in.robot.angular_velocity_rad_s * seconds;
}
}  // namespace

TEST(AimRates, WorldMotionHasCorrectSignsAndIgnoresCameraPanAndArrivalJitter)
{
  auto in = input_at_4m();
  in.has_velocity = true;
  in.velocity = {1, -.5, 0};
  AimPredictionConfig config;
  config.compensate_gravity = false;
  config.bullet_speed_mps = 18;
  AimRateEstimator estimator;
  EXPECT_FALSE(estimator.update(in, config, predict_aim(in, config), 7, true).valid);
  for (int frame = 0; frame < 20; ++frame) {
    advance_rate_input(in);
    // Source dt stays20ms; arrival age alternates45/55ms. No spurious angular
    // velocity from camera yaw/pitch or receive-time jitter enters the model.
    in.now_ns = in.frame_stamp_ns + (frame % 2 ? 55000000 : 45000000);
    in.current_attitude = {in.now_ns, frame * .03, frame * -.01, 0};
    const auto rate = estimator.update(in, config, predict_aim(in, config), 7, true);
    ASSERT_TRUE(rate.valid);
    EXPECT_GT(rate.yaw_rad_s, .23);
    EXPECT_LT(rate.yaw_rad_s, .26);
    EXPECT_GT(rate.pitch_rad_s, .11);
    EXPECT_LT(rate.pitch_rad_s, .13);
  }
  in.velocity = {0, 0, 0};
  estimator.reset();
  estimator.update(in, config, predict_aim(in, config), 7, true);
  advance_rate_input(in);
  in.current_attitude = {in.now_ns, -1.2, .7, .1};
  const auto stationary = estimator.update(in, config, predict_aim(in, config), 7, true);
  ASSERT_TRUE(stationary.valid);
  EXPECT_DOUBLE_EQ(stationary.yaw_rad_s, 0);
  EXPECT_DOUBLE_EQ(stationary.pitch_rad_s, 0);
}

TEST(AimRates, AcquisitionIdentityCoastingStaleAndBadCadenceClearRates)
{
  auto in = input_at_4m();
  in.has_velocity = true;
  in.velocity = {1, 0, 0};
  AimPredictionConfig config;
  AimRateEstimator estimator;
  auto update = [&](int id = 7, bool fresh = true) {
    return estimator.update(in, config, predict_aim(in, config), id, fresh);
  };
  EXPECT_FALSE(update().valid);
  advance_rate_input(in);
  EXPECT_TRUE(update().valid);
  advance_rate_input(in);
  EXPECT_FALSE(update(8).valid);
  advance_rate_input(in);
  EXPECT_TRUE(update(8).valid);
  EXPECT_FALSE(update(8).valid);  // Replayed source frame.
  advance_rate_input(in);
  EXPECT_FALSE(update(8, false).valid);  // Coasting/2D/offset/quality-loss eligibility.
  advance_rate_input(in);
  EXPECT_FALSE(update(8).valid);  // First reacquired measurement.
  advance_rate_input(in);
  EXPECT_TRUE(update(8).valid);
  advance_rate_input(in, .12);
  EXPECT_FALSE(update(8).valid);
  in.now_ns += 200000000;
  EXPECT_FALSE(update(8).valid);
}

TEST(AimRates, PoseJumpExcessRateAccelerationAndBulletChangeFallBackToZero)
{
  auto in = input_at_4m();
  in.has_velocity = true;
  in.velocity = {1, 0, 0};
  AimPredictionConfig config;
  AimRateEstimator estimator;
  estimator.update(in, config, predict_aim(in, config), 7, true);
  advance_rate_input(in);
  ASSERT_TRUE(estimator.update(in, config, predict_aim(in, config), 7, true).valid);
  advance_rate_input(in);
  in.position.x += .8;
  const auto jump = estimator.update(in, config, predict_aim(in, config), 7, true);
  EXPECT_FALSE(jump.valid);
  EXPECT_DOUBLE_EQ(jump.yaw_rad_s, 0);
  advance_rate_input(in);
  EXPECT_TRUE(estimator.update(in, config, predict_aim(in, config), 7, true).valid);
  advance_rate_input(in);
  config.bullet_speed_mps += 1;
  EXPECT_FALSE(estimator.update(in, config, predict_aim(in, config), 7, true).valid);
  advance_rate_input(in);
  in.velocity.x = -4;
  EXPECT_FALSE(estimator.update(in, config, predict_aim(in, config), 7, true).valid);
  in.velocity.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(estimator.update(in, config, predict_aim(in, config), 7, true).valid);
}

TEST(AimRates, IncludesChangingBallisticDropAndRejectsPlateSwitches)
{
  auto in = input_at_4m();
  in.has_velocity = true;
  in.velocity = {0, 0, -1};
  AimPredictionConfig config;
  config.bullet_speed_mps = 18;
  AimRateEstimator estimator;
  estimator.update(in, config, predict_aim(in, config), 7, true);
  advance_rate_input(in);
  const auto radial = estimator.update(in, config, predict_aim(in, config), 7, true);
  ASSERT_TRUE(radial.valid);
  EXPECT_DOUBLE_EQ(radial.yaw_rad_s, 0);
  EXPECT_LT(radial.pitch_rad_s,
            -.01);  // Less flight time/drop when approaching.
  in.has_robot_state = true;
  in.robot.center = {0, 0, 4};
  in.robot.phase_rad = 0;
  in.robot.angular_velocity_rad_s = 1;
  in.velocity = {0, 0, 0};
  estimator.reset();
  estimator.update(in, config, predict_aim(in, config), 7, true);
  advance_rate_input(in);
  auto solution = predict_aim(in, config);
  const auto spinning = estimator.update(in, config, solution, 7, true);
  EXPECT_TRUE(spinning.valid);
  EXPECT_GT(std::abs(spinning.yaw_rad_s), .03);
  advance_rate_input(in);
  in.robot.phase_rad += 1.57079632679;
  solution = predict_aim(in, config);
  EXPECT_FALSE(estimator.update(in, config, solution, 7, true).valid);
}
