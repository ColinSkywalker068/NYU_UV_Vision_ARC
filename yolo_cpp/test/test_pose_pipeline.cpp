#include <gtest/gtest.h>
#include <rcl/time.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <opencv2/calib3d.hpp>
#include <sstream>
#include <thread>

#include "yolo_cpp/aim_solver_node.hpp"
#include "yolo_cpp/armor_processor_node.hpp"
#include "yolo_cpp/armor_tracker_node.hpp"
#include "yolo_cpp/detection_postprocess.hpp"
#include "yolo_cpp/pnp_solver.hpp"

using namespace yolo_cpp;

namespace
{
const cv::Matx33d kCamera(600, 0, 320, 0, 600, 192, 0, 0, 1);
const std::vector<cv::Point3f> kCorners{
  {-0.0675F, 0.0275F, 0}, {-0.0675F, -0.0275F, 0}, {0.0675F, -0.0275F, 0}, {0.0675F, 0.0275F, 0}};

std::array<cv::Point2f, 4> project(const cv::Point3d & p, const cv::Matx33d & rotation)
{
  cv::Mat rvec;
  cv::Rodrigues(rotation, rvec);
  std::vector<cv::Point2f> corners;
  cv::projectPoints(kCorners, rvec, cv::Vec3d(p.x, p.y, p.z), kCamera, cv::Mat(), corners);
  return {corners[0], corners[1], corners[2], corners[3]};
}

std::vector<Detection> decode(const std::array<cv::Point2f, 4> & corners)
{
  std::array<float, 14> row{};
  row[0] = row[1] = 10000;
  for (std::size_t i = 0; i < 4; ++i) {
    row[0] = std::min(row[0], corners[i].x);
    row[1] = std::min(row[1], corners[i].y);
    row[2] = std::max(row[2], corners[i].x);
    row[3] = std::max(row[3], corners[i].y);
    row[6 + 2 * i] = corners[i].x;
    row[7 + 2 * i] = corners[i].y;
  }
  row[4] = 0.95F;
  row[5] = 4;  // Red sentry metadata class.
  return parse_aruw_pose_tensor(row.data(), {1, 1, 14}, 640, 384, 0.5F);
}
}  // namespace

TEST(ArmorPose, RealCornerDecodeAndPnPRecoverPose)
{
  ArmorPnPSolver solver({kCamera, {}}, 0.135, 0.055, 0.23, 0.055);
  const cv::Point3d ground_truth(0.3, -0.2, 3.0);
  const auto rotation = camera_to_inertial_rotation({1, 0.4, 0.1, 0.08});
  const auto points = project(ground_truth, rotation);
  const auto detections = decode(points);
  ASSERT_EQ(detections.size(), 1U);
  EXPECT_EQ(detections[0].class_id, RED_CLASS);
  EXPECT_EQ(detections[0].armor_number, 7);
  EXPECT_TRUE(detections[0].corners_are_keypoints);
  geometry_msgs::msg::Pose pose;
  PoseQuality quality;
  ASSERT_TRUE(solver.solve(detections[0].corners_model, 0, pose, &quality, 0.05));
  EXPECT_NEAR(pose.position.x, ground_truth.x, 1e-4);
  EXPECT_NEAR(pose.position.y, ground_truth.y, 1e-4);
  EXPECT_NEAR(pose.position.z, ground_truth.z, 1e-4);
  EXPECT_LT(quality.reprojection_rms_px, 0.001);
  // At this pixel size IPPE has two plausible normals: retain translation but
  // do not use a weakly constrained normal as confident spin evidence.
  EXPECT_FALSE(quality.normal_reliable);
  const auto near_points = project({0.1, -0.05, 1.0}, rotation);
  ASSERT_TRUE(solver.solve(near_points, 0, pose, &quality, 0.05));
  EXPECT_TRUE(quality.normal_reliable);
  auto crossed = points;
  std::swap(crossed[1], crossed[2]);
  EXPECT_FALSE(solver.solve(crossed, 0, pose));
  auto duplicate = points;
  duplicate[1] = duplicate[0];
  EXPECT_FALSE(solver.solve(duplicate, 0, pose));
}

TEST(ModelProfile, AruwUsesBgrStretchWithCorrectCoordinateInverse)
{
  cv::Mat bgr(120, 200, CV_8UC3, cv::Scalar(255, 128, 0));
  const auto tensor = preprocess_aruw_bgr(bgr);
  EXPECT_FLOAT_EQ(tensor.ptr<float>()[0], 1.0F);
  EXPECT_NEAR(tensor.ptr<float>()[640 * 384], 128.0 / 255, 1e-7);
  EXPECT_FLOAT_EQ(tensor.ptr<float>()[2 * 640 * 384], 0.0F);
  const auto metadata = aruw_resize_metadata(1000, 800);
  const auto point = unletterbox_point({320, 192}, metadata);
  EXPECT_FLOAT_EQ(point.x, 500);
  EXPECT_FLOAT_EQ(point.y, 400);
}

TEST(ArmorPose, TinyPlateDepthUncertaintyDoesNotBecomeAConfidentPose)
{
  ArmorPnPSolver solver({kCamera, {}}, .135, .055, .225, .055);
  geometry_msgs::msg::Pose pose;
  PoseQuality quality;
  const auto rotation = camera_to_inertial_rotation({1, .3, .1, 0});
  const auto far_corners = project({.2, 0, 6}, rotation);
  // A perfect reprojection does not remove uncertainty from tiny keypoints.
  EXPECT_FALSE(solver.solve(far_corners, 0, pose, &quality));
  EXPECT_GT(quality.relative_depth_uncertainty, .05);
  const auto near_corners = project({.2, 0, 1.0}, rotation);
  ASSERT_TRUE(solver.solve(near_corners, 0, pose, &quality));
  EXPECT_LT(quality.relative_depth_uncertainty, .05);
  // Enlarging source coordinates must scale the model's pixel noise as well.
  auto enlarged = far_corners;
  for (auto & point : enlarged) {
    point *= 3;
  }
  auto enlarged_camera = kCamera * 3;
  enlarged_camera(2, 2) = 1;
  ArmorPnPSolver source_solver({enlarged_camera, {}}, .135, .055, .225, .055);
  EXPECT_FALSE(source_solver.solve(enlarged, 0, pose, &quality, 1.5));
}

TEST(ModelProfile, AruwTracksKeypointRectangleInsteadOfContextBox)
{
  // An asymmetric context box must not move the centering point off the armor.
  const std::array<float, 14> row{40,  50,  240, 180, 0.95, 4,   100,
                                  130, 100, 100, 180, 100,  180, 130};
  const auto result = parse_aruw_pose_tensor(row.data(), {1, 1, 14}, 640, 384, 0.5);
  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0].bbox_model, cv::Rect(100, 100, 80, 30));
}

TEST(ModelProfile, AruwDropsDuplicateQuadsButKeepsDistinctAdjacentPlates)
{
  const std::array<float, 14> first{100, 100, 180, 130, 0.95, 4,   100,
                                    130, 100, 100, 180, 100,  180, 130};
  auto duplicate = first;
  duplicate[4] = 0.8;
  duplicate[5] = 5;  // A conflicting number on the same physical quad is not a new plate.
  auto neighbor = first;
  neighbor[0] += 70;
  neighbor[2] += 70;
  for (int i = 6; i < 14; i += 2) {
    neighbor[i] += 70;
  }
  std::vector<float> rows(first.begin(), first.end());
  rows.insert(rows.end(), duplicate.begin(), duplicate.end());
  rows.insert(rows.end(), neighbor.begin(), neighbor.end());
  const auto result = parse_aruw_pose_tensor(rows.data(), {1, 3, 14}, 640, 384, 0.5);
  ASSERT_EQ(result.size(), 2U);
  EXPECT_EQ(result[0].armor_number, 7);
  EXPECT_EQ(result[1].bbox_model.x, 170);
}

TEST(ProductionPosePipeline, CameraPanDoesNotCreateTargetMotionOrChangeAbsoluteCommand)
{
  rclcpp::init(0, nullptr);
  auto publisher = std::make_shared<rclcpp::Node>("pose_pipeline_test");
  auto tracker = std::make_shared<ArmorTrackerNode>();
  auto processor = std::make_shared<ArmorProcessorNode>();
  rclcpp::NodeOptions options;
  options.parameter_overrides(
    {rclcpp::Parameter("target_topic", "processor/target"),
     rclcpp::Parameter("output_mode", "absolute"), rclcpp::Parameter("compensate_gravity", false)});
  auto solver = std::make_shared<AimSolverNode>(options);
  rclcpp::executors::SingleThreadedExecutor executor;
  for (const auto & node :
       std::vector<rclcpp::Node::SharedPtr>{publisher, tracker, processor, solver}) {
    executor.add_node(node);
  }
  auto attitudes = publisher->create_publisher<cv_auto_aim_interfaces::msg::GimbalAttitude>(
    "gimbal/attitude", 100);
  auto armors =
    publisher->create_publisher<cv_auto_aim_interfaces::msg::Armors>("detector/armors", 10);
  std::vector<cv_auto_aim_interfaces::msg::AimSolution> solutions;
  std::vector<cv_auto_aim_interfaces::msg::Target> targets;
  auto solution_sub = publisher->create_subscription<cv_auto_aim_interfaces::msg::AimSolution>(
    "aim_solution", 100, [&](const cv_auto_aim_interfaces::msg::AimSolution::SharedPtr msg) {
      solutions.push_back(*msg);
    });
  auto target_sub = publisher->create_subscription<cv_auto_aim_interfaces::msg::Target>(
    "processor/target", 100,
    [&](const cv_auto_aim_interfaces::msg::Target::SharedPtr msg) { targets.push_back(*msg); });
  const auto spin = [&]() {
    for (int i = 0; i < 10; ++i) {
      executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };
  spin();
  ArmorPnPSolver pnp({kCamera, {}}, 0.135, 0.055, 0.23, 0.055);
  const cv::Point3d fixed(0.4, -0.1, 4);
  const auto fixed_rotation = camera_to_inertial_rotation({1, 0.25, 0.08, 0});
  for (int frame = 0; frame < 80; ++frame) {
    const auto stamp = publisher->now() - rclcpp::Duration::from_seconds(0.005);
    const AttitudeSample attitude{
      stamp.nanoseconds(), -0.1 + frame * 0.004, 0.02 * std::sin(frame * 0.1),
      0.02 * std::cos(frame * 0.1)};
    cv_auto_aim_interfaces::msg::GimbalAttitude telemetry;
    telemetry.header.stamp = stamp;
    telemetry.valid = true;
    telemetry.yaw = attitude.yaw;
    telemetry.pitch = attitude.pitch;
    telemetry.roll = attitude.roll;
    attitudes->publish(telemetry);
    spin();
    const auto rotation = camera_to_inertial_rotation(attitude);
    const auto camera_point = rotate_point(rotation.t(), fixed);
    const auto decoded = decode(project(camera_point, rotation.t() * fixed_rotation));
    EXPECT_EQ(decoded.size(), 1U);
    if (decoded.empty()) {
      continue;
    }
    const auto & d = decoded.front();
    cv_auto_aim_interfaces::msg::Armor armor;
    PoseQuality quality;
    armor.has_pose = pnp.solve(d.corners_model, 0, armor.pose_camera, &quality, 0.05);
    EXPECT_TRUE(armor.has_pose);
    armor.pose_normal_reliable = quality.normal_reliable;
    armor.corners_are_keypoints = true;
    armor.has_corners = true;
    armor.is_enemy = true;
    armor.class_id = d.class_id;
    armor.armor_number = d.armor_number;
    armor.confidence = d.confidence;
    armor.image_width = 640;
    armor.image_height = 384;
    armor.bbox_xyxy = {
      d.bbox_model.x, d.bbox_model.y, d.bbox_model.x + d.bbox_model.width,
      d.bbox_model.y + d.bbox_model.height};
    armor.center_x = d.bbox_model.x + d.bbox_model.width * 0.5;
    armor.center_y = d.bbox_model.y + d.bbox_model.height * 0.5;
    cv_auto_aim_interfaces::msg::Armors message;
    message.header.stamp = stamp;
    message.header.frame_id = "camera_optical_frame";
    message.armors = {armor};
    armors->publish(message);
    spin();
  }
  std::size_t valid_count = 0;
  double max_yaw_error = 0;
  double max_velocity = 0;
  for (const auto & result : solutions) {
    EXPECT_FALSE(result.suggest_fire);
    if (!result.has_solution) {
      continue;
    }
    ++valid_count;
    max_yaw_error = std::max(max_yaw_error, std::abs(result.yaw - std::atan2(fixed.x, fixed.z)));
  }
  for (const auto & target : targets) {
    EXPECT_FALSE(target.has_robot_state);  // A single stationary plate remains direct.
    if (target.has_velocity) {
      max_velocity = std::max(
        max_velocity,
        std::sqrt(
          target.velocity.x * target.velocity.x + target.velocity.y * target.velocity.y +
          target.velocity.z * target.velocity.z));
    }
  }
  EXPECT_GE(valid_count, 70U);
  EXPECT_LT(max_yaw_error, 0.001);  // <0.06 deg, no phantom pan correction.
  EXPECT_LT(max_velocity, 0.01);
  RecordProperty("valid_commands", static_cast<int>(valid_count));
  const auto precise = [](double value) {
    std::ostringstream text;
    text << std::setprecision(17) << value;
    return text.str();
  };
  RecordProperty("max_yaw_error_rad", precise(max_yaw_error));
  RecordProperty("max_phantom_velocity_mps", precise(max_velocity));
  executor.cancel();
  rclcpp::shutdown();
}

TEST(ProductionPosePipeline, FixedPlateWithOffsetUsesCaptureAttitudeAndRejectsStaleAttitude)
{
  rclcpp::init(0, nullptr);
  struct ShutdownGuard
  {
    ~ShutdownGuard() { rclcpp::shutdown(); }
  } shutdown_guard;
  auto source = std::make_shared<rclcpp::Node>("frame_contract_test");
  // Simulator camera_offset_m=(forward, left, up); CV uses (right, down, forward).
  const cv::Point3d offset_link(0.12, 0.04, -0.02);
  const cv::Point3d offset(-offset_link.y, -offset_link.z, offset_link.x);
  rclcpp::NodeOptions tracker_options;
  tracker_options.parameter_overrides({rclcpp::Parameter(
    "camera_translation_m", std::vector<double>{offset.x, offset.y, offset.z})});
  auto tracker = std::make_shared<ArmorTrackerNode>(tracker_options);
  auto processor = std::make_shared<ArmorProcessorNode>();
  rclcpp::NodeOptions solver_options;
  solver_options.parameter_overrides(
    {rclcpp::Parameter("target_topic", "processor/target"),
     rclcpp::Parameter("output_mode", "absolute"),
     rclcpp::Parameter("compensate_gravity", false)});
  auto solver = std::make_shared<AimSolverNode>(solver_options);
  rclcpp::executors::SingleThreadedExecutor executor;
  const std::vector<rclcpp::Node::SharedPtr> nodes{source, tracker, processor, solver};
  for (const auto & node : nodes) {
    executor.add_node(node);
    ASSERT_EQ(rcl_enable_ros_time_override(node->get_clock()->get_clock_handle()), RCL_RET_OK);
  }
  const auto set_time = [&](int64_t ns) {
    for (const auto & node : nodes) {
      EXPECT_EQ(rcl_set_ros_time_override(node->get_clock()->get_clock_handle(), ns), RCL_RET_OK);
    }
  };
  set_time(100000000000);
  auto attitudes = source->create_publisher<cv_auto_aim_interfaces::msg::GimbalAttitude>(
    "gimbal/attitude", 100);
  auto armors = source->create_publisher<cv_auto_aim_interfaces::msg::Armors>("detector/armors", 10);
  std::vector<cv_auto_aim_interfaces::msg::Target> targets;
  std::vector<cv_auto_aim_interfaces::msg::AimSolution> solutions;
  int64_t observed_attitude_stamp = 0;
  auto attitude_sub = source->create_subscription<cv_auto_aim_interfaces::msg::GimbalAttitude>(
    "gimbal/attitude", 100, [&](cv_auto_aim_interfaces::msg::GimbalAttitude::SharedPtr msg) {
      observed_attitude_stamp = rclcpp::Time(msg->header.stamp).nanoseconds();
    });
  auto target_sub = source->create_subscription<cv_auto_aim_interfaces::msg::Target>(
    "processor/target", 100, [&](cv_auto_aim_interfaces::msg::Target::SharedPtr msg) {
      targets.push_back(*msg);
    });
  auto solution_sub = source->create_subscription<cv_auto_aim_interfaces::msg::AimSolution>(
    "aim_solution", 100, [&](cv_auto_aim_interfaces::msg::AimSolution::SharedPtr msg) {
      solutions.push_back(*msg);
    });
  // Wall time is only a transport timeout; all algorithm time is explicitly controlled.
  const auto wait_for = [&](const std::function<bool()> & ready) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    do {
      executor.spin_some();
      if (ready()) {
        executor.spin_some();
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
  };
  ASSERT_TRUE(wait_for([&]() {
    return attitudes->get_subscription_count() == 3 && armors->get_subscription_count() == 1 &&
           target_sub->get_publisher_count() == 1 && solution_sub->get_publisher_count() == 1 &&
           processor->count_subscribers("processor/target") == 2 &&
           tracker->count_subscribers("tracker/tracked_armors") == 1;
  }));
  const auto publish_attitude = [&](const AttitudeSample & a) {
    cv_auto_aim_interfaces::msg::GimbalAttitude msg;
    msg.header.stamp = rclcpp::Time(a.stamp_ns);
    msg.header.frame_id = "gimbal_inertial";
    msg.valid = true;
    msg.yaw = a.yaw;
    msg.pitch = a.pitch;
    attitudes->publish(msg);
    return wait_for([&]() { return observed_attitude_stamp == a.stamp_ns; });
  };
  const cv::Point3d fixed(0.4, -0.1, 4.0);
  const auto detection = [&](const AttitudeSample & a) {
    const auto rotation = camera_to_inertial_rotation(a);
    const auto point = rotate_point(rotation.t(), fixed - rotate_point(rotation, offset));
    cv_auto_aim_interfaces::msg::Armor armor;
    armor.has_pose = true;
    armor.pose_camera.position.x = point.x;
    armor.pose_camera.position.y = point.y;
    armor.pose_camera.position.z = point.z;
    const auto q = rotation_quaternion(rotation.t());
    armor.pose_camera.orientation.x = q[0];
    armor.pose_camera.orientation.y = q[1];
    armor.pose_camera.orientation.z = q[2];
    armor.pose_camera.orientation.w = q[3];
    armor.is_enemy = true;
    armor.confidence = 1.0;
    armor.armor_number = 3;
    armor.image_width = 640;
    armor.image_height = 384;
    armor.center_x = 320 + 600 * point.x / point.z;
    armor.center_y = 192 + 600 * point.y / point.z;
    const int x = static_cast<int>(armor.center_x), y = static_cast<int>(armor.center_y);
    armor.bbox_xyxy = {x - 10, y - 4, x + 10, y + 4};
    cv_auto_aim_interfaces::msg::Armors msg;
    msg.header.stamp = rclcpp::Time(a.stamp_ns);
    msg.header.frame_id = "camera_optical_frame";
    msg.armors = {armor};
    return msg;
  };
  const auto publish_detection = [&](const cv_auto_aim_interfaces::msg::Armors & msg) {
    targets.clear();
    solutions.clear();
    armors->publish(msg);
    return wait_for([&]() { return !targets.empty() && !solutions.empty(); });
  };
  int64_t delivery_ns = 0;
  for (int frame = 0; frame < 24; ++frame) {
    SCOPED_TRACE(frame);
    const int64_t capture_ns = 100000000000 + frame * 100000000LL;
    const AttitudeSample capture{
      capture_ns, 0.15 * std::sin(frame * 0.4), 0.08 * std::cos(frame * 0.4), 0};
    set_time(capture_ns);
    ASSERT_TRUE(publish_attitude(capture));
    auto msg = detection(capture);
    delivery_ns = capture_ns + 50000000;  // Exactly 50 ms; keep the capture stamp.
    set_time(delivery_ns);
    ASSERT_TRUE(publish_attitude({delivery_ns, capture.yaw + 0.04, capture.pitch - 0.02, 0}));
    ASSERT_TRUE(publish_detection(msg));
    if (frame == 0) {
      continue;  // Production track_min_hits=2.
    }
    const auto & target = targets.back();
    ASSERT_TRUE(target.has_target);
    ASSERT_TRUE(target.has_pose);
    ASSERT_TRUE(target.camera_pose_valid);
    ASSERT_TRUE(target.has_velocity);
    EXPECT_EQ(target.header.frame_id, "gimbal_inertial");
    EXPECT_EQ(rclcpp::Time(target.header.stamp).nanoseconds(), capture_ns);
    EXPECT_EQ(target.measurement_stamp_ns, capture_ns);
    EXPECT_NEAR(target.pose.position.x, fixed.x, 1e-6);
    EXPECT_NEAR(target.pose.position.y, fixed.y, 1e-6);
    EXPECT_NEAR(target.pose.position.z, fixed.z, 1e-6);
    EXPECT_LT(cv::norm(cv::Point3d(target.velocity.x, target.velocity.y, target.velocity.z)), 1e-6);
    const auto & solution = solutions.back();
    ASSERT_TRUE(solution.has_solution);
    ASSERT_TRUE(solution.used_3d);
    EXPECT_NEAR(solution.yaw, std::atan2(fixed.x, fixed.z), 1e-6);
    EXPECT_NEAR(solution.pitch, std::atan2(-fixed.y, std::hypot(fixed.x, fixed.z)), 1e-6);
    EXPECT_NEAR(solution.frame_age_s, 0.05, 1e-7);
    EXPECT_NEAR(solution.measurement_age_s, 0.05, 1e-7);
  }
  const int64_t stale_now = delivery_ns + 300000000;
  set_time(stale_now);
  // Fresh image, no new telemetry: the last attitude is exactly 300 ms old.
  ASSERT_TRUE(publish_detection(detection({stale_now, -0.1, 0.05, 0})));
  EXPECT_EQ(rclcpp::Time(targets.back().header.stamp).nanoseconds(), stale_now);
  EXPECT_FALSE(targets.back().camera_pose_valid);
  EXPECT_FALSE(targets.back().has_pose);
  EXPECT_FALSE(solutions.back().has_solution);
  EXPECT_FALSE(solutions.back().used_3d);
  EXPECT_FALSE(solutions.back().suggest_fire);
}
