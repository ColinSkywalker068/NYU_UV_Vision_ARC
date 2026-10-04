#include <gtest/gtest.h>
#include <cmath>
#include <limits>

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
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_EQ(result.innovation_rejected_count, 1U);
  result = manager.update(
    {make_detection_3d(104.0F, 122.0F, 0.9F, cv::Point3d(1.0, 0.0, 2.0))}, 1.0 / 60.0);
  ASSERT_EQ(result.tracks.size(), 2U);
  EXPECT_NE(result.tracks.back().track_id, first_track);
  EXPECT_DOUBLE_EQ(cv::norm(result.tracks.back().velocity), 0);
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

TEST(TrackManagerTest, ExplicitFrameCapKeepsLegacyBehaviorWithoutRefreshingMeasurementTime)
{
  yolo_cpp::TrackConfig config;
  config.track_max_coast_frames = 6;
  yolo_cpp::TrackManager manager(config);
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

TEST(TrackManagerTest, TimeAuthoritativeCoastAndIdentityAcrossFrameRates)
{
  for (double fps : {30., 60., 90., 120.}) {
    yolo_cpp::TrackManager manager;
    auto observation = make_detection_3d(320, 240, .9, {0, 0, 4});
    observation.measurement_stamp_ns = 1000000000;
    manager.update({observation}, 1 / fps);
    manager.update({observation}, 1 / fps);
    double age = 0;
    for (int frame = 1; age < .40; ++frame) {
      age += 1 / fps;
      auto result = manager.update({}, 1 / fps);
      EXPECT_EQ(result.selected_target.has_value(), age <= .12 + 1e-9) << fps << ": " << age;
      EXPECT_EQ(!result.tracks.empty(), age <= .35 + 1e-9) << fps << ": " << age;
      if (!result.tracks.empty()) {
        EXPECT_NEAR(result.tracks.front().real_measurement_age_s, age, 1e-9);
        EXPECT_EQ(result.tracks.front().observation.measurement_stamp_ns, 1000000000);
      }
    }
  }
}

TEST(TrackManagerTest, VariableCadenceAndLongGapCannotReviveExpiredIdentity)
{
  yolo_cpp::TrackManager manager;
  auto observation = make_detection_3d(320, 240, .9, {0, 0, 4});
  manager.update({observation}, .01);
  const auto id = manager.update({observation}, .02).selected_target->track_id;
  double age = 0;
  for (double dt : {.008, .032, .011, .024, .025, .019, .002}) {
    age += dt;
    auto result = manager.update({}, dt);
    EXPECT_EQ(result.selected_target.has_value(), age <= .12);
  }
  auto result = manager.update({observation}, .7);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_NE(result.selected_target->track_id, id);
  result = manager.update({}, 1.4);
  EXPECT_TRUE(result.tracks.empty());
}

TEST(TrackManagerTest, IsolatedPoseJumpIsQuarantinedAndRecoveryKeepsRealTimestamp)
{
  for (double fps : {30., 60., 90., 120.}) {
    yolo_cpp::TrackManager manager;
    auto observation = make_detection_3d(320, 240, .9, {0, 0, 4});
    observation.measurement_stamp_ns = 1000000000;
    manager.update({observation}, 1/fps);
    const auto id = manager.update({observation}, 1/fps).selected_target->track_id;
    observation.position.x = .28;
    observation.measurement_stamp_ns += 10000000;
    auto result = manager.update({observation}, 1/fps);
    ASSERT_EQ(result.tracks.size(), 1U);
    EXPECT_EQ(result.matched_count, 0U);
    EXPECT_EQ(result.innovation_rejected_count, 1U);
    EXPECT_EQ(result.selected_target->state, yolo_cpp::TrackState::kCoasting);
    EXPECT_EQ(result.selected_target->observation.measurement_stamp_ns, 1000000000);
    EXPECT_EQ(result.selected_target->association_reason, "innovation_pending");
    observation.position.x = 0;
    result = manager.update({observation}, 1/fps);
    ASSERT_TRUE(result.selected_target);
    EXPECT_EQ(result.selected_target->track_id, id);
    EXPECT_EQ(result.matched_count, 1U);
    EXPECT_DOUBLE_EQ(cv::norm(result.selected_target->velocity), 0);
  }
}

TEST(TrackManagerTest, SustainedRapidManeuverConfirmsWithoutVelocityImpulse)
{
  yolo_cpp::TrackManager manager;
  auto observation = make_detection_3d(320, 240, .9, {0, 0, 4});
  manager.update({observation}, .02);
  const auto id = manager.update({observation}, .02).selected_target->track_id;
  observation.position.x = .24;
  manager.update({observation}, .02);
  observation.position.x = .27;
  auto result = manager.update({observation}, .02);
  ASSERT_EQ(result.tracks.size(), 1U);
  EXPECT_EQ(result.selected_target->track_id, id);
  EXPECT_EQ(result.selected_target->association_reason, "innovation_confirmed");
  EXPECT_DOUBLE_EQ(cv::norm(result.selected_target->velocity), 0);
  EXPECT_DOUBLE_EQ(result.selected_target->observation.position.x, .27);
  observation.position.x = .30;
  result = manager.update({observation}, .02);
  EXPECT_GT(result.selected_target->velocity[0], 0);
}

TEST(TrackManagerTest, PoorQualityPoseAndDistractorsCannotRefreshMeasurement)
{
  yolo_cpp::TrackManager manager;
  auto observation = make_detection_3d(320, 240, .9, {0, 0, 4});
  observation.measurement_stamp_ns = 1000000000;
  manager.update({observation}, .02);
  manager.update({observation}, .02);
  auto friendly = observation;
  friendly.is_enemy = false;
  observation.reprojection_rms_px = 8;
  observation.measurement_stamp_ns += 20000000;
  auto result = manager.update({observation, friendly}, .02);
  EXPECT_EQ(result.raw_count, 2U);
  EXPECT_EQ(result.enemy_count, 1U);
  EXPECT_EQ(result.posed_count, 1U);
  EXPECT_EQ(result.quality_rejected_count, 1U);
  EXPECT_EQ(result.selected_target->observation.measurement_stamp_ns, 1000000000);
  EXPECT_EQ(result.selected_target->association_reason, "pose_quality");
  observation.reprojection_rms_px = 3;
  observation.position.x = .15;
  result = manager.update({observation}, .02);
  EXPECT_EQ(result.innovation_rejected_count, 1U);
  EXPECT_EQ(result.matched_count, 0U);
  observation.position.x = std::numeric_limits<double>::quiet_NaN();
  result = manager.update({observation}, .02);
  EXPECT_EQ(result.quality_rejected_count, 1U);
}

TEST(TrackManagerTest, SimultaneouslyVisiblePlatesDoNotEnterJumpProbation)
{
  yolo_cpp::TrackManager manager;
  auto first = make_detection_3d(320, 240, .9, {0, 0, 4});
  auto second = make_detection_3d(360, 240, .95, {.6, 0, 4});
  manager.update({first}, .02);
  manager.update({first}, .02);
  auto result = manager.update({first, second}, .02);
  ASSERT_EQ(result.tracks.size(), 2U);
  EXPECT_EQ(result.innovation_rejected_count, 0U);
  EXPECT_EQ(result.matched_count, 1U);
  result = manager.update({second, first}, .02);
  EXPECT_EQ(result.matched_count, 2U);
  EXPECT_EQ(result.tracks[0].observation.position.x, 0);
  EXPECT_EQ(result.tracks[1].observation.position.x, .6);
}

TEST(TrackManagerTest, CameraPanLeavesStationaryWorldPoseAndVelocityUnchanged)
{
  yolo_cpp::TrackManager manager;
  for (int frame = 0; frame < 100; ++frame) {
    // ROS wrapper rotates camera poses into the fixed tracking frame before this call.
    auto observation = make_detection_3d(100 + frame * 4, 240, .9, {.1, 0, 4});
    auto result = manager.update({observation}, frame % 2 ? .008 : .025);
    ASSERT_EQ(result.tracks.size(), 1U);
    EXPECT_NEAR(cv::norm(result.selected_target->velocity), 0, 1e-9);
    EXPECT_EQ(result.innovation_rejected_count, 0U);
  }
}

TEST(TrackManagerTest, TranslationAndReversalStayResponsiveAcrossCadences)
{
  for (double fps : {30., 60., 90., 120.}) {
    yolo_cpp::TrackManager manager;
    double x = 0;
    for (int frame = 0; frame < 2 * fps; ++frame) {
      const double speed = frame < fps ? 2. : -2.;
      x += speed / fps;
      auto observation = make_detection_3d(320, 240, .9, {x, 0, 4});
      auto result = manager.update({observation}, 1 / fps);
      ASSERT_EQ(result.tracks.size(), 1U);
      EXPECT_EQ(result.innovation_rejected_count, 0U);
      if (frame > fps + .3 * fps) {EXPECT_LT(result.selected_target->velocity[0], -1.7);}
    }
  }
}


TEST(TrackManagerTest, OptionalResidualGateAndTwoDimensionalFallback)
{
  yolo_cpp::TrackConfig config;
  config.pose_max_reprojection_rms_px = 0;
  yolo_cpp::TrackManager manager(config);
  auto observation = make_detection_3d(320, 240, .9, {0, 0, 4});
  observation.reprojection_rms_px = 8;
  observation.measurement_stamp_ns = 1000000000;
  auto result = manager.update({observation}, .02);
  ASSERT_TRUE(result.selected_target);
  EXPECT_EQ(result.quality_rejected_count, 0U);
  observation.has_pose = false;
  observation.measurement_stamp_ns += 20000000;
  result = manager.update({observation}, .02);
  EXPECT_EQ(result.matched_count, 1U);
  EXPECT_EQ(result.posed_count, 0U);
  EXPECT_EQ(result.selected_target->association_reason, "matched_2d");
  EXPECT_EQ(result.selected_target->observation.measurement_stamp_ns, 1000000000);
  EXPECT_NEAR(result.selected_target->real_measurement_age_s, .02, 1e-9);
}

TEST(TrackManagerTest, SweptSpinPlateTransitionsRetainMeasuredCandidates)
{
  for (double fps : {30., 60., 90., 120.}) {
    for (double omega : {2., 4., 8.}) {
      yolo_cpp::TrackManager manager;
      int no_measured_target = 0;
      int consecutive_coast = 0;
      int frames = 0;
      for (double time = 0; time < 2; time += 1/fps) {
        std::vector<yolo_cpp::TrackObservation> visible;
        for (int plate = 0; plate < 4; ++plate) {
          const double phase = omega * time + plate * 3.141592653589793 / 2;
          if (std::cos(phase) < .1) {continue;}
          visible.push_back(make_detection_3d(
            320 + 70 * std::sin(phase), 240, .9,
            {.25 * std::sin(phase), 0, 4 - .25 * std::cos(phase)}));
        }
        auto result = manager.update(visible, 1/fps);
        ASSERT_TRUE(result.selected_target);
        if (result.selected_target->state == yolo_cpp::TrackState::kCoasting) {
          ++no_measured_target;
          ++consecutive_coast;
        } else {consecutive_coast = 0;}
        EXPECT_LE(consecutive_coast, 2);
        ++frames;
        EXPECT_LT(cv::norm(result.selected_target->velocity), 3.0);
      }
      EXPECT_LT(no_measured_target, frames / 5) << fps << " fps, " << omega << " rad/s";
    }
  }
}
