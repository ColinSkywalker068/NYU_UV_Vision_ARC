#include <gtest/gtest.h>
#include <cmath>

#include "yolo_cpp/tracker.hpp"

namespace
{

yolo_cpp::TrackObservation make_detection(float center_x, float center_y, float confidence,
  bool is_enemy = true, int image_width = 640, int image_height = 480)
{
  yolo_cpp::TrackObservation observation;
  observation.center = cv::Point2f(center_x, center_y);
  observation.bbox =
    cv::Rect(static_cast<int>(center_x - 20.0F), static_cast<int>(center_y - 10.0F), 40, 20);
  observation.confidence = confidence;
  observation.class_id = 0;
  observation.is_enemy = is_enemy;
  observation.image_width = image_width;
  observation.image_height = image_height;
  return observation;
}

yolo_cpp::TrackObservation make_detection_3d(
  float center_x, float center_y, float confidence, cv::Point3d position)
{
  auto observation = make_detection(center_x, center_y, confidence);
  observation.has_pose = true;
  observation.position = position;
  return observation;
}

}  // namespace

TEST(ArmorPlateTrackerTest, PredictAndUpdateRemainStable)
{
  yolo_cpp::ArmorPlateTracker tracker(100.0, 100.0, 40.0, 20.0);
  const auto predicted = tracker.predict(1.0);
  EXPECT_NEAR(predicted.first, 100.0, 1e-6);
  EXPECT_NEAR(predicted.second, 100.0, 1e-6);

  const auto updated = tracker.update(110.0, 100.0, 42.0, 22.0);
  EXPECT_GT(updated.first, 100.0);
  EXPECT_LT(updated.first, 110.0);
  EXPECT_NEAR(updated.second, 100.0, 1e-3);
  EXPECT_DOUBLE_EQ(tracker.width(), 42.0);
  EXPECT_DOUBLE_EQ(tracker.height(), 22.0);
}

TEST(TrackManagerTest, ConfirmsCoastsAndDropsTracks)
{
  yolo_cpp::TrackConfig config;
  config.track_min_hits = 2;
  config.track_max_coast_frames = 2;
  config.track_drop_frames = 3;
  yolo_cpp::TrackManager manager(config);

  auto result = manager.update({make_detection(100.0F, 120.0F, 0.8F)}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_EQ(result.tracks.front().state, yolo_cpp::TrackState::kDetecting);
  const int track_id = result.tracks.front().track_id;

  result = manager.update({make_detection(102.0F, 120.0F, 0.82F)}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_EQ(result.tracks.front().track_id, track_id);
  EXPECT_EQ(result.tracks.front().state, yolo_cpp::TrackState::kTracking);
  ASSERT_TRUE(result.selected_target.has_value());
  EXPECT_EQ(result.selected_target->track_id, track_id);

  result = manager.update({}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_EQ(result.tracks.front().state, yolo_cpp::TrackState::kCoasting);
  ASSERT_TRUE(result.selected_target.has_value());
  EXPECT_EQ(result.selected_target->track_id, track_id);

  result = manager.update({}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_TRUE(result.selected_target.has_value());

  result = manager.update({}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_FALSE(result.selected_target.has_value());

  result = manager.update({}, 1.0 / 60.0);
  EXPECT_TRUE(result.tracks.empty());
}

TEST(TrackManagerTest, IgnoresNonEnemyDetectionsByDefault)
{
  yolo_cpp::TrackManager manager;
  const auto result = manager.update({make_detection(200.0F, 200.0F, 0.95F, false)}, 1.0 / 60.0);
  EXPECT_TRUE(result.tracks.empty());
  EXPECT_FALSE(result.selected_target.has_value());
}

TEST(TrackManagerTest, RetainsLockedTargetUntilChallengerClearlyWins)
{
  yolo_cpp::TrackConfig config;
  config.target_switch_min_score_delta = 0.15;
  yolo_cpp::TrackManager manager(config);

  auto result = manager.update(
    {make_detection(120.0F, 220.0F, 0.90F), make_detection(310.0F, 220.0F, 0.60F)}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 2U);
  const int initially_selected = result.selected_target->track_id;

  result = manager.update(
    {make_detection(125.0F, 220.0F, 0.90F), make_detection(305.0F, 220.0F, 0.70F)}, 1.0 / 60.0);
  ASSERT_TRUE(result.selected_target.has_value());
  EXPECT_EQ(result.selected_target->track_id, initially_selected);

  result = manager.update(
    {make_detection(130.0F, 220.0F, 0.70F), make_detection(300.0F, 220.0F, 0.85F)}, 1.0 / 60.0);
  ASSERT_TRUE(result.selected_target.has_value());
  EXPECT_NE(result.selected_target->track_id, initially_selected);
}

TEST(TrackManagerTest, MaintainsSeparateTracksForTwoTargets)
{
  yolo_cpp::TrackManager manager;

  auto result = manager.update(
    {make_detection(150.0F, 140.0F, 0.80F), make_detection(420.0F, 140.0F, 0.78F)}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 2U);

  result = manager.update(
    {make_detection(165.0F, 140.0F, 0.82F), make_detection(405.0F, 140.0F, 0.79F)}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 2U);
  EXPECT_NE(result.tracks[0].track_id, result.tracks[1].track_id);
  EXPECT_EQ(result.tracks[0].state, yolo_cpp::TrackState::kTracking);
  EXPECT_EQ(result.tracks[1].state, yolo_cpp::TrackState::kTracking);
}

TEST(TrackManagerTest, Uses3DGateWhenPoseIsAvailable)
{
  yolo_cpp::TrackConfig config;
  config.association_max_3d_distance_m = 0.25;
  yolo_cpp::TrackManager manager(config);

  auto result = manager.update(
    {make_detection_3d(100.0F, 120.0F, 0.8F, cv::Point3d(0.0, 0.0, 2.0))}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 1U);
  const int first_track = result.tracks.front().track_id;
  EXPECT_TRUE(result.tracks.front().has_pose);

  result = manager.update(
    {make_detection_3d(102.0F, 121.0F, 0.82F, cv::Point3d(0.02, 0.0, 2.0))}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_EQ(result.tracks.front().track_id, first_track);
  EXPECT_TRUE(result.tracks.front().has_velocity);

  result = manager.update(
    {make_detection_3d(104.0F, 122.0F, 0.9F, cv::Point3d(1.0, 0.0, 2.0))}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 2U);
}

TEST(TrackManagerTest, DifferentArmorNumbersDoNotSwapIdentitiesAtCrossing)
{
  yolo_cpp::TrackManager manager;
  auto first = make_detection_3d(300, 240, 0.9, {-0.1, 0, 4});
  auto second = make_detection_3d(340, 240, 0.9, {0.1, 0, 4});
  first.armor_number = 3;
  second.armor_number = 4;
  auto before = manager.update({first, second}, 1.0/60);
  ASSERT_EQ(before.tracks.size(), 2U);
  const auto first_id = before.tracks[0].track_id;
  const auto second_id = before.tracks[1].track_id;
  std::swap(first.position, second.position);
  std::swap(first.center, second.center);
  const auto after = manager.update({first, second}, 1.0/60);
  ASSERT_EQ(after.tracks.size(), 2U);
  for (const auto & track : after.tracks) {
    EXPECT_EQ(track.observation.armor_number, track.track_id == first_id ? 3 : 4);
    EXPECT_TRUE(track.track_id == first_id || track.track_id == second_id);
  }
}

TEST(TrackManagerTest, FreshBoxesCannotRefreshAnOldThreeDimensionalMeasurement)
{
  yolo_cpp::TrackManager manager;
  auto observation = make_detection_3d(300, 200, .9, {.1, 0, 4});
  observation.measurement_stamp_ns = 1000000000;
  auto result = manager.update({observation}, .02);
  ASSERT_TRUE(result.tracks[0].has_pose);
  for (int i = 1; i < 20; ++i) {
    observation.has_pose = false;
    observation.has_camera_pose = false;
    observation.measurement_stamp_ns += 20000000;
    result = manager.update({observation}, .02);
    ASSERT_EQ(result.tracks.size(), 1U);
    EXPECT_EQ(result.tracks[0].observation.measurement_stamp_ns, 1000000000);
  }
  observation.has_pose = true;
  observation.measurement_stamp_ns += 20000000;
  result = manager.update({observation}, .02);
  EXPECT_EQ(result.tracks[0].observation.measurement_stamp_ns, observation.measurement_stamp_ns);
}

TEST(TrackManagerTest, AcquisitionUsesTheSameCenterAndConfidenceScoreAsTargetLock)
{
  yolo_cpp::TrackManager manager;
  auto center = make_detection(320, 240, .80);
  auto edge = make_detection(40, 40, .85);
  const auto result = manager.update({edge, center}, .02);
  ASSERT_TRUE(result.selected_target);
  EXPECT_FLOAT_EQ(result.selected_target->observation.center.x, 320);
  EXPECT_FLOAT_EQ(result.selected_target->observation.center.y, 240);
}

TEST(TrackManagerTest, ShortDropoutsKeepIdentityAndNeverRefreshMeasurementTime)
{
  yolo_cpp::TrackManager manager;
  auto observation = make_detection_3d(320, 240, .9, {.1, 0, 2});
  observation.measurement_stamp_ns = 1000000000;
  manager.update({observation}, .01);
  observation.measurement_stamp_ns += 10000000;
  auto result = manager.update({observation}, .01);
  const int id = result.selected_target->track_id;
  for (int missed = 1; missed <= 6; ++missed) {
    result = manager.update({}, .01);
    ASSERT_TRUE(result.selected_target);
    EXPECT_EQ(result.selected_target->track_id, id);
    EXPECT_EQ(result.selected_target->state, yolo_cpp::TrackState::kCoasting);
    EXPECT_EQ(result.selected_target->observation.measurement_stamp_ns, 1010000000);
  }
  result = manager.update({}, .01);
  EXPECT_FALSE(result.selected_target);
  observation.measurement_stamp_ns = 1090000000;
  result = manager.update({observation}, .01);
  ASSERT_TRUE(result.selected_target);
  EXPECT_EQ(result.selected_target->track_id, id);
  EXPECT_EQ(result.selected_target->observation.measurement_stamp_ns, 1090000000);
}

TEST(TrackManagerTest, TimeLimitStopsCoastingEvenWhenFrameBudgetIsLarge)
{
  yolo_cpp::TrackConfig config;
  config.track_max_coast_frames = 100;
  config.track_drop_frames = 200;
  yolo_cpp::TrackManager manager(config);
  auto observation = make_detection(320, 240, .9);
  manager.update({observation}, .01);
  manager.update({observation}, .01);
  EXPECT_TRUE(manager.update({}, .05).selected_target);
  EXPECT_TRUE(manager.update({}, .05).selected_target);
  auto result = manager.update({}, .05);
  EXPECT_FALSE(result.selected_target);
  EXPECT_EQ(result.tracks.size(), 1U);
  result = manager.update({}, .21);
  EXPECT_TRUE(result.tracks.empty());
}

TEST(TrackManagerTest, ReacquisitionVelocityUsesTheActualMeasurementGap)
{
  yolo_cpp::TrackConfig config;
  config.position_filter_tau_s = 0;
  config.velocity_filter_tau_s = 0;
  yolo_cpp::TrackManager manager(config);
  auto observation = make_detection_3d(320, 240, .9, {0, 0, 2});
  observation.measurement_stamp_ns = 1000000000;
  manager.update({observation}, .02);
  observation.position.x = .02;
  observation.measurement_stamp_ns += 20000000;
  auto result = manager.update({observation}, .02);
  EXPECT_NEAR(result.selected_target->velocity[0], 1.0, 1e-9);
  for (int i = 0; i < 4; ++i) {manager.update({}, .02);}
  observation.position.x = .12;
  observation.measurement_stamp_ns += 100000000;
  result = manager.update({observation}, .02);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_NEAR(result.selected_target->velocity[0], 1.0, 1e-9);
  EXPECT_NEAR(result.selected_target->observation.position.x, .12, 1e-9);
}

TEST(TrackManagerTest, TimeBasedFilteringReducesJitterWhileFollowingTranslation)
{
  for (double dt : {1.0 / 30, 1.0 / 75, 1.0 / 120}) {
    yolo_cpp::TrackManager manager;
    double raw_error = 0, filtered_error = 0, velocity_error = 0;
    int count = 0;
    for (int frame = 0; frame < 250; ++frame) {
      const double truth = .4 * frame * dt;
      const double noise = frame % 2 ? .015 : -.015;
      auto observation = make_detection_3d(320, 240, .9, {truth + noise, 0, 4});
      observation.measurement_stamp_ns = 1000000000 + static_cast<int64_t>(frame * dt * 1e9);
      const auto result = manager.update({observation}, dt);
      ASSERT_EQ(result.tracks.size(), 1U);
      ASSERT_TRUE(result.selected_target);
      if (frame > 50) {
        const double error = result.selected_target->observation.position.x - truth;
        const double speed_error = result.selected_target->velocity[0] - .4;
        raw_error += noise * noise;
        filtered_error += error * error;
        velocity_error += speed_error * speed_error;
        ++count;
      }
    }
    EXPECT_LT(std::sqrt(filtered_error / count), .75 * std::sqrt(raw_error / count));
    EXPECT_LT(std::sqrt(velocity_error / count), .3);
  }
}
