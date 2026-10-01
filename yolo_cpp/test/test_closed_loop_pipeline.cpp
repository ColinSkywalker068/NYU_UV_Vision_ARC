#include <gtest/gtest.h>
#include <rcl/time.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <opencv2/calib3d.hpp>
#include <random>
#include <sstream>
#include <thread>

#include "fixtures/typec_vision_reference.hpp"
#include "yolo_cpp/aim_solver_node.hpp"
#include "yolo_cpp/armor_processor_node.hpp"
#include "yolo_cpp/armor_tracker_node.hpp"
#include "yolo_cpp/detection_postprocess.hpp"
#include "yolo_cpp/pnp_solver.hpp"

using namespace yolo_cpp;
namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr double kDegrees = 180.0 / kPi;
const cv::Matx33d kCamera(900, 0, 320, 0, 900, 192, 0, 0, 1);
const std::vector<cv::Point3f> kCorners{
  {-0.0675F, 0.0275F, 0}, {-0.0675F, -0.0275F, 0}, {0.0675F, -0.0275F, 0}, {0.0675F, 0.0275F, 0}};

struct Scenario
{
  const char * name;
  double spin;
  bool solo;
  bool distractor;
  double range{3.0};
  double corner_noise{0.10};
  bool expect_pose_rejection{false};
};
std::string precise(double value)
{
  std::ostringstream out;
  out << std::setprecision(12) << value;
  return out.str();
}
double percentile(std::vector<double> values, double quantile)
{
  if (values.empty()) {
    return 1e9;
  }
  std::sort(values.begin(), values.end());
  return values[std::min(values.size() - 1, static_cast<std::size_t>(quantile * values.size()))];
}

class ClosedLoopPipeline : public ::testing::TestWithParam<Scenario>
{
};
}  // namespace

TEST_P(ClosedLoopPipeline, NoisyCornersDelayedFramesAndSentryReference)
{
  const auto scenario = GetParam();
  rclcpp::init(0, nullptr);
  auto source = std::make_shared<rclcpp::Node>("closed_loop_source");
  rclcpp::NodeOptions tracker_options;
  tracker_options.parameter_overrides({rclcpp::Parameter("track_max_coast_frames", 3)});
  auto tracker = std::make_shared<ArmorTrackerNode>(tracker_options);
  auto processor = std::make_shared<ArmorProcessorNode>();
  rclcpp::NodeOptions solver_options;
  solver_options.parameter_overrides(
    {rclcpp::Parameter("target_topic", "processor/target"),
     rclcpp::Parameter("output_mode", "absolute"), rclcpp::Parameter("compensate_gravity", false),
     rclcpp::Parameter("actuator_delay_s", 0.12)});
  auto solver = std::make_shared<AimSolverNode>(solver_options);
  rclcpp::executors::SingleThreadedExecutor executor;
  const std::vector<rclcpp::Node::SharedPtr> nodes{source, tracker, processor, solver};
  for (const auto & node : nodes) {
    executor.add_node(node);
    ASSERT_EQ(rcl_enable_ros_time_override(node->get_clock()->get_clock_handle()), RCL_RET_OK);
  }
  const auto set_time = [&](double seconds) {
    const auto nanoseconds = static_cast<int64_t>((100 + seconds) * 1e9);
    for (const auto & node : nodes) {
      EXPECT_EQ(
        rcl_set_ros_time_override(node->get_clock()->get_clock_handle(), nanoseconds), RCL_RET_OK);
    }
    return nanoseconds;
  };
  set_time(0);
  auto attitude_pub =
    source->create_publisher<cv_auto_aim_interfaces::msg::GimbalAttitude>("gimbal/attitude", 100);
  auto detection_pub =
    source->create_publisher<cv_auto_aim_interfaces::msg::Armors>("detector/armors", 10);
  cv_auto_aim_interfaces::msg::AimSolution solution;
  cv_auto_aim_interfaces::msg::Target target;
  bool fresh_solution = false;
  auto solution_sub = source->create_subscription<cv_auto_aim_interfaces::msg::AimSolution>(
    "aim_solution", 100, [&](cv_auto_aim_interfaces::msg::AimSolution::SharedPtr m) {
      solution = *m;
      fresh_solution = true;
    });
  auto target_sub = source->create_subscription<cv_auto_aim_interfaces::msg::Target>(
    "processor/target", 100,
    [&](cv_auto_aim_interfaces::msg::Target::SharedPtr m) { target = *m; });
  const auto spin = [&]() {
    for (int i = 0; i < 5; ++i) {
      executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };
  spin();
  ArmorPnPSolver pnp({kCamera, {}}, .135, .055, .23, .055);
  std::mt19937 random(82431);
  std::normal_distribution<double> pixel_noise(0, scenario.corner_noise);
  std::deque<cv_auto_aim_interfaces::msg::Armors> delayed;
  sentry_reference::ResetVisionCmdFilter(0, 0);
  double actual_yaw = 0, actual_pitch = 0;
  double previous_yaw_rate = 0, previous_pitch_rate = 0;
  double maximum_yaw_rate = 0, maximum_pitch_rate = 0, maximum_yaw_accel = 0,
         maximum_pitch_accel = 0;
  std::vector<double> center_errors, omega_errors, command_errors, pitch_errors, reference_errors;
  int visible_target_frames = 0;
  int valid_commands = 0, confirmed_robot_frames = 0, eligible_frames = 0, generated_poses = 0;
  const auto center_at = [&](double t, bool distractor) {
    if (distractor) {
      return cv::Point3d(-0.65, .15, 3.7);
    }
    return cv::Point3d(scenario.solo ? .25 : .25 + .25 * std::sin(.8 * t), -.08, scenario.range);
  };
  for (int frame = 0; frame < 500; ++frame) {
    const double t = frame * .02;
    // 200-Hz Type C reference updates; the fixture contains the firmware's real
    // filter/slew functions. A separate 35-ms first-order plant is an explicit
    // stress assumption, not evidence of our real motor/PID response.
    for (int step = 0; step < 4; ++step) {
      const float old_yaw = sentry_reference::vision_yaw_cmd_filtered;
      const float old_pitch = sentry_reference::vision_pitch_cmd_filtered;
      sentry_reference::ApplyVisionAxisSlew(
        sentry_reference::vision_yaw_target_filtered, &sentry_reference::vision_yaw_cmd_filtered,
        &sentry_reference::vision_yaw_slew_rate_deg_s, VISION_YAW_MAX_SLEW_DEG_S,
        VISION_YAW_MAX_ACCEL_DEG_S2, .005F);
      sentry_reference::ApplyVisionAxisSlew(
        sentry_reference::vision_pitch_target_filtered,
        &sentry_reference::vision_pitch_cmd_filtered,
        &sentry_reference::vision_pitch_slew_rate_deg_s, VISION_PITCH_MAX_SLEW_DEG_S,
        VISION_PITCH_MAX_ACCEL_DEG_S2, .005F);
      const double yaw_rate = (sentry_reference::vision_yaw_cmd_filtered - old_yaw) / .005;
      const double pitch_rate = (sentry_reference::vision_pitch_cmd_filtered - old_pitch) / .005;
      maximum_yaw_rate = std::max(maximum_yaw_rate, std::abs(yaw_rate));
      maximum_pitch_rate = std::max(maximum_pitch_rate, std::abs(pitch_rate));
      maximum_yaw_accel =
        std::max(maximum_yaw_accel, std::abs(yaw_rate - previous_yaw_rate) / .005);
      maximum_pitch_accel =
        std::max(maximum_pitch_accel, std::abs(pitch_rate - previous_pitch_rate) / .005);
      previous_yaw_rate = yaw_rate;
      previous_pitch_rate = pitch_rate;
      actual_yaw +=
        (sentry_reference::vision_yaw_cmd_filtered / kDegrees - actual_yaw) * .005 / .035;
      actual_pitch +=
        (sentry_reference::vision_pitch_cmd_filtered / kDegrees - actual_pitch) * .005 / .035;
      const auto stamp = set_time(t + step * .005);
      cv_auto_aim_interfaces::msg::GimbalAttitude attitude;
      attitude.header.stamp = rclcpp::Time(stamp);
      attitude.valid = true;
      attitude.yaw = actual_yaw;
      attitude.pitch = actual_pitch;
      attitude.roll = .015 * std::sin(2 * (t + step * .005));
      attitude_pub->publish(attitude);
    }
    spin();
    const double capture_time = t + .015;
    const auto camera_rotation =
      camera_to_inertial_rotation({1, actual_yaw, actual_pitch, .015 * std::sin(2 * capture_time)});
    cv_auto_aim_interfaces::msg::Armors message;
    message.header.stamp = source->now();
    message.header.frame_id = "camera_optical_frame";
    for (int robot = 0; robot < (scenario.distractor ? 2 : 1); ++robot) {
      const auto center = center_at(capture_time, robot == 1);
      const double phase = scenario.solo ? 0 : (robot ? -2 : scenario.spin) * capture_time + .55;
      for (int slot = 0; slot < (scenario.solo ? 1 : 4); ++slot) {
        const double angle = phase + slot * kPi / 2;
        if (std::cos(angle) < .3) {
          continue;
        }
        const double radius = scenario.solo ? 0 : (slot % 2 ? .29 : .24);
        const cv::Point3d normal(std::sin(angle), 0, -std::cos(angle));
        auto point = center + radius * normal;
        if (!scenario.solo && slot % 2) {
          point.y += .045;
        }
        const auto camera_point = rotate_point(camera_rotation.t(), point);
        // Object +Z points into the robot; the processor recovers the outward normal.
        const auto object_rotation = camera_to_inertial_rotation({1, -angle, .15, 0});
        cv::Mat rotation_vector;
        cv::Rodrigues(camera_rotation.t() * object_rotation, rotation_vector);
        std::vector<cv::Point2f> corners;
        cv::projectPoints(
          kCorners, rotation_vector, cv::Vec3d(camera_point.x, camera_point.y, camera_point.z),
          kCamera, cv::Mat(), corners);
        std::array<float, 14> row{};
        row[0] = row[1] = 10000;
        for (int k = 0; k < 4; ++k) {
          corners[k].x += pixel_noise(random);
          corners[k].y += pixel_noise(random);
          row[0] = std::min(row[0], corners[k].x);
          row[1] = std::min(row[1], corners[k].y);
          row[2] = std::max(row[2], corners[k].x);
          row[3] = std::max(row[3], corners[k].y);
          row[6 + 2 * k] = corners[k].x;
          row[7 + 2 * k] = corners[k].y;
        }
        row[4] = robot ? .7 : .95;
        row[5] = robot ? 6 : 5;
        const auto decoded = parse_aruw_pose_tensor(row.data(), {1, 1, 14}, 640, 384, .5);
        if (decoded.empty()) {
          continue;
        }
        const auto & d = decoded.front();
        cv_auto_aim_interfaces::msg::Armor armor;
        PoseQuality quality;
        armor.has_pose = pnp.solve(d.corners_model, 0, armor.pose_camera, &quality);
        if (armor.has_pose) {
          ++generated_poses;
        }
        armor.has_corners = armor.corners_are_keypoints = true;
        armor.pose_normal_reliable = quality.normal_reliable;
        armor.is_enemy = true;
        armor.class_id = d.class_id;
        armor.armor_number = d.armor_number;
        armor.confidence = d.confidence;
        armor.image_width = 640;
        armor.image_height = 384;
        armor.bbox_xyxy = {
          d.bbox_model.x, d.bbox_model.y, d.bbox_model.x + d.bbox_model.width,
          d.bbox_model.y + d.bbox_model.height};
        armor.center_x = d.bbox_model.x + d.bbox_model.width * .5;
        armor.center_y = d.bbox_model.y + d.bbox_model.height * .5;
        message.armors.push_back(armor);
      }
    }
    // Deterministic two-frame bursts model missed detections, not fake measurements.
    if (frame % 59 < 2) {
      message.armors.clear();
    }
    delayed.push_back(message);
    if (delayed.size() <= 2) {
      continue;
    }
    fresh_solution = false;
    detection_pub->publish(delayed.front());
    delayed.pop_front();
    spin();
    EXPECT_FALSE(solution.suggest_fire);
    if (fresh_solution && solution.has_solution) {
      sentry_reference::gimbal_fetch_data.yaw_actual_angle = actual_yaw * kDegrees;
      sentry_reference::gimbal_fetch_data.pitch_actual_angle = actual_pitch * kDegrees;
      sentry_reference::UpdateVisionTargetFilter(
        solution.yaw * kDegrees, solution.pitch * kDegrees, 0, 0, 0, 0);
    }
    if (frame < 100) {
      continue;
    }
    ++eligible_frames;
    if (target.has_target) {
      ++visible_target_frames;
    }
    if (!fresh_solution || !solution.has_solution) {
      continue;
    }
    ++valid_commands;
    const double measurement_time = rclcpp::Time(target.header.stamp).seconds() - 100;
    if (target.has_robot_state) {
      ++confirmed_robot_frames;
      center_errors.push_back(cv::norm(
        cv::Point3d(target.robot_center.x, target.robot_center.y, target.robot_center.z) -
        center_at(measurement_time, false)));
      omega_errors.push_back(std::abs(target.robot_angular_velocity - scenario.spin));
    }
    if (scenario.solo) {
      EXPECT_FALSE(target.has_robot_state);
    }
    const double future_time = measurement_time + solution.prediction_horizon_s;
    const auto future_center = center_at(future_time, false);
    double best_error = 1e9, best_pitch_error = 1e9, best_combined_error = 1e9;
    for (int slot = 0; slot < (scenario.solo ? 1 : 4); ++slot) {
      const double phase = scenario.solo ? 0 : scenario.spin * future_time + .55 + slot * kPi / 2;
      const double radius = scenario.solo ? 0 : (slot % 2 ? .29 : .24);
      auto future_plate =
        future_center + radius * cv::Point3d(std::sin(phase), 0, -std::cos(phase));
      if (!scenario.solo && slot % 2) {
        future_plate.y += .045;
      }
      const cv::Point3d normal(std::sin(phase), 0, -std::cos(phase));
      if (!scenario.solo && -normal.dot(future_plate) / cv::norm(future_plate) <= 0) {
        continue;
      }
      const double yaw_error =
        std::abs(angle_difference(solution.yaw, std::atan2(future_plate.x, future_plate.z)));
      const double pitch_error = std::abs(
        solution.pitch - std::atan2(-future_plate.y, std::hypot(future_plate.x, future_plate.z)));
      if (std::hypot(yaw_error, pitch_error) < best_combined_error) {
        best_combined_error = std::hypot(yaw_error, pitch_error);
        best_error = yaw_error;
        best_pitch_error = pitch_error;
      }
    }
    command_errors.push_back(best_error);
    pitch_errors.push_back(best_pitch_error);
    reference_errors.push_back(std::abs(angle_difference(solution.yaw, actual_yaw)));
  }
  if (scenario.expect_pose_rejection) {
    EXPECT_GE(visible_target_frames, eligible_frames * .90);
    EXPECT_EQ(generated_poses, 0);
    EXPECT_EQ(valid_commands, 0);
    EXPECT_EQ(confirmed_robot_frames, 0);
  } else {
    EXPECT_GT(generated_poses, 400);
    EXPECT_GE(valid_commands, eligible_frames * .90);
  }
  EXPECT_LE(maximum_yaw_rate, VISION_YAW_MAX_SLEW_DEG_S + .01);
  EXPECT_LE(maximum_pitch_rate, VISION_PITCH_MAX_SLEW_DEG_S + .01);
  EXPECT_LE(maximum_yaw_accel, VISION_YAW_MAX_ACCEL_DEG_S2 * 1.002 + .01);
  EXPECT_LE(maximum_pitch_accel, VISION_PITCH_MAX_ACCEL_DEG_S2 * 1.002 + .01);
  if (!scenario.expect_pose_rejection) {
    EXPECT_LT(percentile(command_errors, .95), scenario.solo ? .005 : .04);
    EXPECT_LT(percentile(pitch_errors, .95), scenario.solo ? .005 : .02);
  }
  if (!scenario.solo && !scenario.expect_pose_rejection) {
    EXPECT_GE(confirmed_robot_frames, valid_commands * .8);
    EXPECT_LT(percentile(center_errors, .95), .15);
    EXPECT_LT(percentile(omega_errors, .95), .8);
  }
  RecordProperty("valid_commands", valid_commands);
  RecordProperty("generated_poses", generated_poses);
  RecordProperty("visible_target_frames", visible_target_frames);
  RecordProperty("confirmed_robot_frames", confirmed_robot_frames);
  RecordProperty(
    "p95_center_error_m",
    center_errors.empty() ? "not_applicable" : precise(percentile(center_errors, .95)));
  RecordProperty(
    "p95_omega_error_rad_s",
    omega_errors.empty() ? "not_applicable" : precise(percentile(omega_errors, .95)));
  RecordProperty(
    "p95_future_command_error_rad",
    command_errors.empty() ? "not_applicable" : precise(percentile(command_errors, .95)));
  RecordProperty(
    "p95_future_pitch_error_rad",
    pitch_errors.empty() ? "not_applicable" : precise(percentile(pitch_errors, .95)));
  RecordProperty(
    "p95_command_minus_current_attitude_rad",
    reference_errors.empty() ? "not_applicable" : precise(percentile(reference_errors, .95)));
  RecordProperty("max_yaw_rate_deg_s", precise(maximum_yaw_rate));
  RecordProperty("max_pitch_rate_deg_s", precise(maximum_pitch_rate));
  RecordProperty("max_yaw_accel_deg_s2", precise(maximum_yaw_accel));
  RecordProperty("max_pitch_accel_deg_s2", precise(maximum_pitch_accel));
  executor.cancel();
  rclcpp::shutdown();
}

INSTANTIATE_TEST_SUITE_P(
  Scenarios, ClosedLoopPipeline,
  ::testing::Values(
    Scenario{"stationary_solo", 0, true, false}, Scenario{"nonspinning_robot", 0, false, false},
    Scenario{"spinning_robot", 4, false, false}, Scenario{"multiple_robots", 4, false, true},
    Scenario{"far_noisy_spin_rejected", 4, false, false, 6.0, .35, true},
    Scenario{"near_nonspinning", 0, false, false, 1.5, .20}),
  [](const ::testing::TestParamInfo<Scenario> & info) { return info.param.name; });
