#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <iomanip>
#include <opencv2/calib3d.hpp>
#include <sstream>
#include <thread>

#include "yolo_cpp/aim_solver_node.hpp"
#include "yolo_cpp/armor_processor_node.hpp"
#include "yolo_cpp/armor_tracker_node.hpp"
#include "yolo_cpp/detection_postprocess.hpp"
#include "yolo_cpp/mjpeg_preview_server.hpp"
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

TEST(ProductionPosePipeline, CameraPoseSurvivesMissingAttitudeButAbsoluteAimRequiresAlignment)
{
  // Use genuine ARUW keypoints and production PnP defaults, then exercise the
  // tracker -> processor -> solver boundary that previously hid missing RX.
  const auto detections = decode(project({0.1, -0.02, 1.0}, cv::Matx33d::eye()));
  ASSERT_EQ(detections.size(), 1U);
  const auto & detection = detections.front();
  ArmorPnPSolver pnp({kCamera, {}}, 0.135, 0.055, 0.23, 0.055);
  cv_auto_aim_interfaces::msg::Armor armor;
  PoseQuality quality;
  ASSERT_TRUE(pnp.solve(detection.corners_model, 0, armor.pose_camera, &quality));
  armor.has_pose = true;
  armor.pose_normal_reliable = quality.normal_reliable;
  armor.reprojection_rms_px = quality.reprojection_rms_px;
  armor.has_corners = armor.corners_are_keypoints = true;
  armor.is_enemy = true;
  armor.class_id = detection.class_id;
  armor.armor_number = detection.armor_number;
  armor.confidence = detection.confidence;
  armor.image_width = 640;
  armor.image_height = 384;
  armor.bbox_xyxy = {
    detection.bbox_model.x, detection.bbox_model.y,
    detection.bbox_model.x + detection.bbox_model.width,
    detection.bbox_model.y + detection.bbox_model.height};
  armor.center_x = detection.bbox_model.x + detection.bbox_model.width * 0.5;
  armor.center_y = detection.bbox_model.y + detection.bbox_model.height * 0.5;

  rclcpp::init(0, nullptr);
  auto publisher = std::make_shared<rclcpp::Node>("pose_attitude_dependency_test");
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
  std::vector<cv_auto_aim_interfaces::msg::TrackedArmors> tracks;
  auto solution_sub = publisher->create_subscription<cv_auto_aim_interfaces::msg::AimSolution>(
    "aim_solution", 100, [&](const cv_auto_aim_interfaces::msg::AimSolution::SharedPtr msg) {
      solutions.push_back(*msg);
      EXPECT_FALSE(msg->suggest_fire);
      if (!msg->has_solution) {
        EXPECT_FALSE(msg->has_angular_velocity);
      }
    });
  auto target_sub = publisher->create_subscription<cv_auto_aim_interfaces::msg::Target>(
    "processor/target", 100,
    [&](const cv_auto_aim_interfaces::msg::Target::SharedPtr msg) { targets.push_back(*msg); });
  auto track_sub = publisher->create_subscription<cv_auto_aim_interfaces::msg::TrackedArmors>(
    "tracker/tracked_armors", 100,
    [&](const cv_auto_aim_interfaces::msg::TrackedArmors::SharedPtr msg) {
      tracks.push_back(*msg);
    });
  const auto spin = [&]() {
    for (int i = 0; i < 20; ++i) {
      executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  };
  spin();
  const auto frame = [&](bool send_attitude, double capture_delay_s = 0.005) {
    const auto now = publisher->now();
    const auto stamp = now - rclcpp::Duration::from_seconds(capture_delay_s);
    if (send_attitude) {
      cv_auto_aim_interfaces::msg::GimbalAttitude telemetry;
      telemetry.header.stamp = now;
      telemetry.valid = true;
      attitudes->publish(telemetry);
      spin();
    }
    cv_auto_aim_interfaces::msg::Armors message;
    message.header.stamp = stamp;
    message.header.frame_id = "camera_optical_frame";
    message.armors = {armor};
    armors->publish(message);
    spin();
  };
  const auto expect_state = [&](bool valid, const char * reason) {
    EXPECT_FALSE(solutions.empty());
    EXPECT_FALSE(targets.empty());
    EXPECT_FALSE(tracks.empty());
    if (solutions.empty() || targets.empty() || tracks.empty()) {
      return;
    }
    EXPECT_EQ(solutions.back().has_solution, valid);
    if (reason) {
      EXPECT_EQ(solutions.back().status, reason);
    }
    EXPECT_TRUE(targets.back().has_target);
    EXPECT_EQ(targets.back().has_pose, valid);
    EXPECT_EQ(targets.back().camera_pose_valid, valid);
    EXPECT_FALSE(targets.back().has_robot_state);  // A single plate needs no robot model.
    EXPECT_FALSE(tracks.back().tracks.empty());
    for (const auto & track : tracks.back().tracks) {
      EXPECT_TRUE(track.armor.has_pose);  // Raw camera PnP remains available for diagnostics.
      EXPECT_NEAR(track.armor.pose_camera.position.z, 1.0, 1e-4);
      EXPECT_EQ(track.has_pose, valid);
    }
  };
  for (int i = 0; i < 5; ++i) {
    frame(false);
  }
  expect_state(false, "stale_or_missing_attitude");
  for (int i = 0; i < 5; ++i) {
    frame(true);
  }
  expect_state(true, nullptr);
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  for (int i = 0; i < 5; ++i) {
    frame(false);
  }
  expect_state(false, "stale_or_missing_attitude");
  // Fresh receipt cannot validate an older capture across a telemetry gap.
  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  frame(true, 0.08);
  expect_state(false, "capture_attitude_mismatch");
  for (int i = 0; i < 5; ++i) {
    frame(true);
  }
  expect_state(true, nullptr);
  executor.cancel();
  rclcpp::shutdown();
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
  std::size_t valid_rate_count = 0;
  double max_yaw_error = 0;
  double max_velocity = 0;
  for (const auto & result : solutions) {
    EXPECT_FALSE(result.suggest_fire);
    if (!result.has_solution) {
      continue;
    }
    ++valid_count;
    if (result.has_angular_velocity) {
      ++valid_rate_count;
      EXPECT_LT(std::abs(result.yaw_velocity), .03);
      EXPECT_LT(std::abs(result.pitch_velocity), .03);
    }
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
  EXPECT_GE(valid_rate_count, 60U);
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

TEST(ControllerPreview, HttpStatusExpiresWithoutCameraFrames)
{
  // Reserve an ephemeral loopback port. No hardware, host networking, or camera.
  const int probe = ::socket(AF_INET, SOCK_STREAM, 0);
  ASSERT_GE(probe, 0);
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  ASSERT_EQ(::bind(probe, reinterpret_cast<sockaddr *>(&address), sizeof(address)), 0);
  socklen_t address_size = sizeof(address);
  ASSERT_EQ(::getsockname(probe, reinterpret_cast<sockaddr *>(&address), &address_size), 0);
  ::close(probe);
  MjpegPreviewServer server;
  MjpegPreviewServer::Config config;
  config.bind_address = "127.0.0.1";
  config.port = ntohs(address.sin_port);
  server.start(config);
  auto stats = [&]() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return std::string();
    }
    timeval timeout{2, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    if (::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
      ::close(fd);
      return std::string();
    }
    const std::string request = "GET /stats.json HTTP/1.1\r\nHost: localhost\r\n\r\n";
    ::send(fd, request.data(), request.size(), 0);
    std::string response;
    char buffer[1024];
    ssize_t size;
    while ((size = ::recv(fd, buffer, sizeof(buffer), 0)) > 0) {
      response.append(buffer, size);
    }
    ::close(fd);
    return response;
  };
  EXPECT_NE(stats().find("\"auto_aim_status\":\"unknown\""), std::string::npos);
  server.update_controller_status(1);
  EXPECT_NE(stats().find("\"auto_aim_status\":\"on\""), std::string::npos);
  std::this_thread::sleep_for(std::chrono::milliseconds(550));
  EXPECT_NE(stats().find("\"auto_aim_status\":\"unknown\""), std::string::npos);
  server.update_controller_status(0);
  EXPECT_NE(stats().find("\"auto_aim_status\":\"off\""), std::string::npos);
  server.update_controller_status(7);
  EXPECT_NE(stats().find("\"auto_aim_status\":\"unknown\""), std::string::npos);
  ASSERT_TRUE(server.update_frame(cv::Mat(2, 2, CV_8UC3, cv::Scalar()), "not an object", 60));
  const auto response = stats();
  EXPECT_EQ(response.find("not an object"), std::string::npos);
  EXPECT_NE(response.find("{\"auto_aim_status\":\"unknown\""), std::string::npos);
}
