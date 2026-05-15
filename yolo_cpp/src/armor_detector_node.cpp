/// @file armor_detector_node.cpp
/// @brief Implements camera capture, YOLO armor detection, color filtering, and Armor messages.

#include "yolo_cpp/armor_detector_node.hpp"

#include "cv_auto_aim_interfaces/msg/armor.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yolo_cpp
{

namespace
{

constexpr int kDefaultMaxHsvCandidates = 3;
constexpr int kPreviewMaxWidth = 1280;
constexpr int kPreviewMaxHeight = 720;
constexpr char kVisionLogDirName[] = "vision_logs";
constexpr double kPi = 3.14159265358979323846;

/// Return local wall-clock time formatted for human-readable runtime logs.
std::string timestamp_now_iso8601_ms()
{
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
    now - seconds).count();
  const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &now_c);
#else
  localtime_r(&now_c, &local_tm);
#endif

  std::ostringstream out;
  out << std::put_time(&local_tm, "%Y-%m-%dT%H:%M:%S")
      << '.' << std::setw(3) << std::setfill('0') << milliseconds;
  return out.str();
}

/// Return compact local wall-clock time suitable for log file names.
std::string timestamp_now_compact()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &now_c);
#else
  localtime_r(&now_c, &local_tm);
#endif

  std::ostringstream out;
  out << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
  return out.str();
}

/// Copy an internal BGR frame packet into a ROS Image message.
sensor_msgs::msg::Image::UniquePtr frame_to_image_message(
  const FramePacket & packet, const rclcpp::Time & stamp)
{
  if (packet.empty()) {
    return nullptr;
  }

  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  msg->header.stamp = stamp;
  msg->header.frame_id = packet.source_id;
  msg->height = static_cast<uint32_t>(packet.frame.rows);
  msg->width = static_cast<uint32_t>(packet.frame.cols);
  msg->encoding = sensor_msgs::image_encodings::BGR8;
  msg->is_bigendian = false;
  msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(packet.frame.step);
  const std::size_t byte_count = packet.frame.total() * packet.frame.elemSize();
  msg->data.resize(byte_count);
  std::memcpy(msg->data.data(), packet.frame.data, byte_count);
  return msg;
}

}  // namespace

/// Declare parameters, choose the inference runtime, and start capture processing.
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("armor_detector_node", options)
{
  camera_config_ = load_camera_config(*this);

  const std::string default_model = resolve_default_model_path();
  const std::string default_deepstream_config = resolve_default_deepstream_config_path();
  this->declare_parameter<std::string>("model_path", default_model);
  this->declare_parameter<std::string>("task", "obb");
  this->declare_parameter<std::string>("type", "yolo");
  this->declare_parameter<std::string>("detector_backend", "native_tensorrt");
  this->declare_parameter<std::string>("deepstream_config_path", default_deepstream_config);
  this->declare_parameter<bool>("allow_native_tensorrt_fallback", false);
  this->declare_parameter<bool>("half", true);
  this->declare_parameter<bool>("show", false);
  this->declare_parameter<int>("show_preview_stride", 1);
  this->declare_parameter<bool>("show_tracker_target_overlay", false);
  this->declare_parameter<double>("conf", 0.30);
  this->declare_parameter<std::string>("inference_mode", "predict");
  this->declare_parameter<bool>("enable_color_verification", true);
  this->declare_parameter<bool>("use_model_color_classes", true);
  this->declare_parameter<std::string>("target_color", "blue");
  this->declare_parameter<bool>("use_robot_id_targeting", false);
  this->declare_parameter<double>("hfov_degrees", 79.2);
  this->declare_parameter<double>("vfov_degrees", 63.78);
  this->declare_parameter<bool>("enable_3d_pose", true);
  this->declare_parameter<std::string>("camera_frame_id", "");
  this->declare_parameter<std::string>("camera_info_topic", "camera_info");
  this->declare_parameter<double>("camera_fx", 0.0);
  this->declare_parameter<double>("camera_fy", 0.0);
  this->declare_parameter<double>("camera_cx", 0.0);
  this->declare_parameter<double>("camera_cy", 0.0);
  this->declare_parameter<std::vector<double>>("camera_distortion", std::vector<double>{});
  this->declare_parameter<double>("small_armor_width_m", 0.135);
  this->declare_parameter<double>("small_armor_height_m", 0.055);
  this->declare_parameter<double>("large_armor_width_m", 0.225);
  this->declare_parameter<double>("large_armor_height_m", 0.055);
  this->declare_parameter<double>("large_armor_min_aspect_ratio", 3.2);
  this->declare_parameter<int>("enemy_color_red_value", 0);
  this->declare_parameter<int>("enemy_color_blue_value", 1);
  this->declare_parameter<int>("max_hsv_candidates", kDefaultMaxHsvCandidates);
  this->declare_parameter<std::string>("runtime_log_dir", "");
  this->declare_parameter<std::string>("vis_window_name", "Armor Detector");
  this->declare_parameter<std::string>("preview_target_topic", "tracker/target");
  this->declare_parameter<bool>("debug_publish_image_raw", false);
  this->declare_parameter<int>("stage_log_interval", 300);

  model_path_ = this->get_parameter("model_path").as_string();
  task_ = this->get_parameter("task").as_string();
  model_type_ = this->get_parameter("type").as_string();
  detector_backend_kind_ = this->get_parameter("detector_backend").as_string();
  deepstream_config_path_ = this->get_parameter("deepstream_config_path").as_string();
  allow_native_tensorrt_fallback_ =
    this->get_parameter("allow_native_tensorrt_fallback").as_bool();
  half_ = this->get_parameter("half").as_bool();
  imgsz_width_ = this->get_parameter("imgsz_width").as_int();
  imgsz_height_ = this->get_parameter("imgsz_height").as_int();
  show_ = this->get_parameter("show").as_bool();
  show_preview_stride_ = std::max(1, this->get_parameter("show_preview_stride").as_int());
  show_tracker_target_overlay_ = this->get_parameter("show_tracker_target_overlay").as_bool();
  conf_ = static_cast<float>(this->get_parameter("conf").as_double());
  inference_mode_ = this->get_parameter("inference_mode").as_string();
  enable_color_verification_ = this->get_parameter("enable_color_verification").as_bool();
  use_model_color_classes_ = this->get_parameter("use_model_color_classes").as_bool();
  target_color_ = this->get_parameter("target_color").as_string();
  use_robot_id_targeting_ = this->get_parameter("use_robot_id_targeting").as_bool();
  hfov_rad_ = this->get_parameter("hfov_degrees").as_double() * kPi / 180.0;
  vfov_rad_ = this->get_parameter("vfov_degrees").as_double() * kPi / 180.0;
  enable_3d_pose_ = this->get_parameter("enable_3d_pose").as_bool();
  camera_frame_id_ = this->get_parameter("camera_frame_id").as_string();
  camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
  camera_fx_ = this->get_parameter("camera_fx").as_double();
  camera_fy_ = this->get_parameter("camera_fy").as_double();
  camera_cx_ = this->get_parameter("camera_cx").as_double();
  camera_cy_ = this->get_parameter("camera_cy").as_double();
  camera_distortion_ = this->get_parameter("camera_distortion").as_double_array();
  small_armor_width_m_ = this->get_parameter("small_armor_width_m").as_double();
  small_armor_height_m_ = this->get_parameter("small_armor_height_m").as_double();
  large_armor_width_m_ = this->get_parameter("large_armor_width_m").as_double();
  large_armor_height_m_ = this->get_parameter("large_armor_height_m").as_double();
  large_armor_min_aspect_ratio_ = this->get_parameter("large_armor_min_aspect_ratio").as_double();
  enemy_color_red_value_ = this->get_parameter("enemy_color_red_value").as_int();
  enemy_color_blue_value_ = this->get_parameter("enemy_color_blue_value").as_int();
  max_hsv_candidates_ = this->get_parameter("max_hsv_candidates").as_int();
  runtime_log_dir_ = this->get_parameter("runtime_log_dir").as_string();
  vis_window_name_ = this->get_parameter("vis_window_name").as_string();
  preview_target_topic_ = this->get_parameter("preview_target_topic").as_string();
  debug_publish_image_raw_ = this->get_parameter("debug_publish_image_raw").as_bool();
  stage_log_interval_ = std::max(
    1, static_cast<int>(this->get_parameter("stage_log_interval").as_int()));

  runtime_log_path_ =
    (std::filesystem::path(resolve_runtime_log_dir()) /
    ("vision_runtime_" + timestamp_now_compact() + ".log")).string();
  runtime_log_stream_.open(runtime_log_path_, std::ios::app);
  if (!runtime_log_stream_.is_open()) {
    RCLCPP_WARN(
      this->get_logger(),
      "Failed to open runtime vision log file: %s",
      runtime_log_path_.c_str());
  } else {
    RCLCPP_INFO(this->get_logger(), "Runtime logs: %s", runtime_log_path_.c_str());
  }

  target_class_ = (target_color_ == "red") ? RED_CLASS : BLUE_CLASS;

  robot_id_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
    "robot_id", 10, std::bind(&ArmorDetectorNode::id_listener_callback, this, std::placeholders::_1));
  attitude_flags_subscription_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
    "attitude_flags", 10,
    std::bind(&ArmorDetectorNode::handle_attitude_flags, this, std::placeholders::_1));
  if (show_tracker_target_overlay_) {
    preview_target_subscription_ =
      this->create_subscription<cv_auto_aim_interfaces::msg::Target>(
      preview_target_topic_, 10,
      std::bind(&ArmorDetectorNode::handle_preview_target, this, std::placeholders::_1));
  }
  if (enable_3d_pose_) {
    camera_info_subscription_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::SensorDataQoS(),
      std::bind(&ArmorDetectorNode::handle_camera_info, this, std::placeholders::_1));
    configure_static_camera_model();
  }
  armors_publisher_ =
    this->create_publisher<cv_auto_aim_interfaces::msg::Armors>("detector/armors", 10);
  if (debug_publish_image_raw_) {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    debug_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", qos);
  }

  if (detector_backend_kind_ == "deepstream") {
    try {
      initialize_deepstream_runtime();
    } catch (const std::exception & exc) {
      if (!allow_native_tensorrt_fallback_) {
        throw;
      }
      RCLCPP_WARN(
        this->get_logger(),
        "DeepStream startup failed (%s). Falling back to native TensorRT because allow_native_tensorrt_fallback=true.",
        exc.what());
      detector_backend_kind_ = "native_tensorrt";
      initialize_native_pipeline();
    }
    catch (...) {
      if (!allow_native_tensorrt_fallback_) {
        throw std::runtime_error(
                "DeepStream startup failed with a non-std exception, likely from the Basler/Pylon GStreamer source");
      }
      RCLCPP_WARN(
        this->get_logger(),
        "DeepStream startup failed with a non-std exception, likely from the Basler/Pylon GStreamer source. Falling back to native TensorRT because allow_native_tensorrt_fallback=true.");
      detector_backend_kind_ = "native_tensorrt";
      initialize_native_pipeline();
    }
  } else {
    initialize_native_pipeline();
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Initialized ArmorDetectorNode camera=%s backend=%s fps=%.2f",
    camera_config_.camera_type.c_str(),
    deepstream_runtime_ != nullptr ?
    deepstream_runtime_->backend_name().c_str() :
    detector_backend_->backend_name().c_str(),
    fps_);
  log_runtime_note("INFO", "Initializing Modular YOLOv8 Node...");
  log_runtime_note("INFO", "Detector Initialized.");
}

ArmorDetectorNode::~ArmorDetectorNode()
{
  timer_.reset();
  detector_backend_.reset();
  deepstream_runtime_.reset();
  if (camera_backend_) {
    camera_backend_->release();
    camera_backend_.reset();
  }
  if (runtime_log_stream_.is_open()) {
    runtime_log_stream_.close();
  }
}

void ArmorDetectorNode::id_listener_callback(const std_msgs::msg::Int32::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!use_robot_id_targeting_ || stm_color_locked_) {
    return;
  }
  if (is_red_team(msg->data)) {
    target_class_ = BLUE_CLASS;
  } else if (is_blue_team(msg->data)) {
    target_class_ = RED_CLASS;
  }
}

void ArmorDetectorNode::handle_attitude_flags(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
{
  if (msg->data.empty()) {
    return;
  }

  std::lock_guard<std::mutex> lock(state_mutex_);
  const int enemy_color = msg->data[0];
  const int previous_target_class = target_class_;
  if (enemy_color == enemy_color_red_value_) {
    target_class_ = RED_CLASS;
    stm_color_locked_ = true;
  } else if (enemy_color == enemy_color_blue_value_) {
    target_class_ = BLUE_CLASS;
    stm_color_locked_ = true;
  }

  if (target_class_ != previous_target_class) {
    const char * color_name = target_class_ == RED_CLASS ? "RED" : "BLUE";
    log_runtime_note(
      "INFO",
      "attitude_flags enemy_color=" + std::to_string(enemy_color) +
      " -> Targeting " + color_name + ".");
  }
}

void ArmorDetectorNode::handle_preview_target(
  const cv_auto_aim_interfaces::msg::Target::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  preview_has_target_ = msg->has_target;
  preview_target_track_id_ = msg->track_id;
  preview_target_bbox_xyxy_ = msg->bbox_xyxy;
}

void ArmorDetectorNode::handle_camera_info(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  if (!enable_3d_pose_) {
    return;
  }

  std::lock_guard<std::mutex> lock(state_mutex_);
  camera_calibration_.model.camera_matrix = cv::Matx33d(
    msg->k[0], msg->k[1], msg->k[2],
    msg->k[3], msg->k[4], msg->k[5],
    msg->k[6], msg->k[7], msg->k[8]);
  camera_calibration_.model.distortion_coefficients = msg->d;
  camera_calibration_.available = true;
  camera_calibration_.from_camera_info = true;
  camera_calibration_.image_width = static_cast<int>(msg->width);
  camera_calibration_.image_height = static_cast<int>(msg->height);
  pnp_solver_.emplace(
    camera_calibration_.model, small_armor_width_m_, small_armor_height_m_,
    large_armor_width_m_, large_armor_height_m_);
}

void ArmorDetectorNode::configure_static_camera_model()
{
  if (camera_fx_ <= 0.0 || camera_fy_ <= 0.0) {
    return;
  }

  camera_calibration_.model.camera_matrix = cv::Matx33d(
    camera_fx_, 0.0, camera_cx_ > 0.0 ? camera_cx_ : static_cast<double>(imgsz_width_) / 2.0,
    0.0, camera_fy_, camera_cy_ > 0.0 ? camera_cy_ : static_cast<double>(imgsz_height_) / 2.0,
    0.0, 0.0, 1.0);
  camera_calibration_.model.distortion_coefficients = camera_distortion_;
  camera_calibration_.available = true;
  camera_calibration_.from_camera_info = false;
  camera_calibration_.image_width = imgsz_width_;
  camera_calibration_.image_height = imgsz_height_;
  pnp_solver_.emplace(
    camera_calibration_.model, small_armor_width_m_, small_armor_height_m_,
    large_armor_width_m_, large_armor_height_m_);
}

PnPCameraModel ArmorDetectorNode::make_fov_camera_model(int image_width, int image_height) const
{
  const double fx = static_cast<double>(image_width) / (2.0 * std::tan(hfov_rad_ / 2.0));
  const double fy = static_cast<double>(image_height) / (2.0 * std::tan(vfov_rad_ / 2.0));
  PnPCameraModel model;
  model.camera_matrix = cv::Matx33d(
    fx, 0.0, static_cast<double>(image_width) / 2.0,
    0.0, fy, static_cast<double>(image_height) / 2.0,
    0.0, 0.0, 1.0);
  model.distortion_coefficients = {};
  return model;
}

std::optional<geometry_msgs::msg::Pose> ArmorDetectorNode::solve_candidate_pose(
  const Candidate & candidate, int image_width, int image_height)
{
  if (!enable_3d_pose_ || !candidate.has_corners) {
    return std::nullopt;
  }

  const double aspect = static_cast<double>(std::max(1, candidate.bbox_source.width)) /
    static_cast<double>(std::max(1, candidate.bbox_source.height));
  const int armor_type = aspect >= large_armor_min_aspect_ratio_ ?
    cv_auto_aim_interfaces::msg::Armor::ARMOR_LARGE :
    cv_auto_aim_interfaces::msg::Armor::ARMOR_SMALL;

  geometry_msgs::msg::Pose pose;
  if (pnp_solver_.has_value()) {
    if (pnp_solver_->solve(candidate.corners_source, armor_type, pose)) {
      return pose;
    }
    return std::nullopt;
  }

  const auto fov_model = make_fov_camera_model(image_width, image_height);
  ArmorPnPSolver solver(
    fov_model, small_armor_width_m_, small_armor_height_m_, large_armor_width_m_,
    large_armor_height_m_);
  if (solver.solve(candidate.corners_source, armor_type, pose)) {
    return pose;
  }
  return std::nullopt;
}

void ArmorDetectorNode::initialize_native_pipeline()
{
  camera_backend_ = create_camera_backend(camera_config_, this->get_logger());
  camera_backend_->initialize();
  fps_ = camera_backend_->actual_fps() > 0.0 ? camera_backend_->actual_fps() : 60.0;

  detector_backend_ = create_detector_backend(
    detector_backend_kind_, model_path_, task_, model_type_, half_, conf_);
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / std::max(1.0, fps_)),
    std::bind(&ArmorDetectorNode::timer_callback, this));
}

void ArmorDetectorNode::initialize_deepstream_runtime()
{
  if (camera_config_.camera_type != "basler") {
    throw std::runtime_error(
            "DeepStream runtime currently supports camera_type=basler only");
  }
  if (inference_mode_ != "predict") {
    throw std::runtime_error(
            "DeepStream runtime currently supports inference_mode=predict only");
  }

  DeepStreamRuntime::Config config{};
  config.model_path = model_path_;
  config.config_template_path = deepstream_config_path_;
  config.task = task_;
  config.imgsz_width = imgsz_width_;
  config.imgsz_height = imgsz_height_;
  config.source_width = camera_config_.imgsz_width;
  config.source_height = camera_config_.imgsz_height;
  config.source_fps = camera_config_.basler_fps;
  config.source_fps_uncapped = camera_config_.basler_uncapped_fps;
  config.source_use_full_sensor = camera_config_.basler_use_full_sensor;
  config.flip_source = camera_config_.basler_flip;
  config.conf_threshold = conf_;
  config.half = half_;
  config.enable_color_verification = enable_color_verification_ && !use_model_color_classes_;
  config.debug_publish_image_raw = debug_publish_image_raw_ || show_;

  fps_ = static_cast<double>(std::max(1, camera_config_.basler_fps));
  deepstream_runtime_ = std::make_unique<DeepStreamRuntime>(
    this->get_logger(),
    config,
    [this](DeepStreamRuntime::Result && result) {
      this->handle_deepstream_result(std::move(result));
    });
}

void ArmorDetectorNode::timer_callback()
{
  const auto capture_start = std::chrono::steady_clock::now();
  FramePacket packet{};
  if (!camera_backend_->capture(packet)) {
    return;
  }
  const auto capture_done = std::chrono::steady_clock::now();

  DetectorStageTimings detector_timings{};
  const auto batch = detector_backend_->infer(packet, this->get_logger(), &detector_timings);
  process_detection_cycle(
    batch,
    &packet,
    detector_timings,
    std::chrono::duration<double, std::milli>(capture_done - capture_start).count());
}

void ArmorDetectorNode::handle_deepstream_result(DeepStreamRuntime::Result result)
{
  const FramePacket * packet = result.source_packet.has_value() ? &(*result.source_packet) : nullptr;
  process_detection_cycle(result.batch, packet, result.timings, result.capture_ms);
}

void ArmorDetectorNode::process_detection_cycle(
  const DetectionBatch & batch,
  const FramePacket * packet,
  const DetectorStageTimings & detector_timings,
  double capture_ms)
{
  // This is the detector's central data boundary: model outputs become candidates,
  // optional debug images, runtime telemetry, and ROS interface messages.
  std::lock_guard<std::mutex> lock(state_mutex_);
  const auto stamp = this->get_clock()->now();
  if (debug_publish_image_raw_ && packet != nullptr) {
    publish_debug_image(*packet, stamp);
  }

  double dt = 1.0 / 60.0;
  if (have_last_frame_time_) {
    dt = (stamp - last_frame_time_).seconds();
    if (dt <= 0.0) {
      dt = 0.001;
    }
  }
  last_frame_time_ = stamp;
  have_last_frame_time_ = true;
  const double instant_fps = 1.0 / std::max(dt, 1e-6);
  preview_fps_ = preview_fps_ <= 0.0 ? instant_fps : (0.9 * preview_fps_) + (0.1 * instant_fps);

  double color_ms = 0.0;
  const auto publish_start = std::chrono::steady_clock::now();
  auto candidates = extract_candidates(batch, packet, &color_ms);
  const auto detected_opponent = std::find_if(
    candidates.begin(), candidates.end(),
    [](const Candidate & candidate) {return candidate.is_enemy;});
  const std::string source_id = packet != nullptr && !packet->source_id.empty() ?
    packet->source_id :
    camera_config_.camera_type;
  const std::string frame_id = camera_frame_id_.empty() ? source_id : camera_frame_id_;
  const int image_width = batch.letterbox.src_w > 0 ?
    batch.letterbox.src_w :
    (packet != nullptr ? packet->width() : imgsz_width_);
  const int image_height = batch.letterbox.src_h > 0 ?
    batch.letterbox.src_h :
    (packet != nullptr ? packet->height() : imgsz_height_);
  publish_armors(candidates, frame_id, stamp, image_width, image_height);
  const auto publish_done = std::chrono::steady_clock::now();

  double preview_ms = 0.0;
  if (
    show_ && packet != nullptr && !packet->empty() &&
    (frame_count_ % show_preview_stride_) == 0)
  {
    const auto preview_start = std::chrono::steady_clock::now();
    render_preview(*packet, candidates);
    const auto preview_done = std::chrono::steady_clock::now();
    preview_ms = std::chrono::duration<double, std::milli>(
      preview_done - preview_start).count();
  }

  log_runtime_fps_once_per_second();
  if (detected_opponent != candidates.end()) {
    log_runtime_opponent_event(
      *detected_opponent,
      true,
      capture_ms + detector_timings.preprocess_ms + detector_timings.upload_ms +
      detector_timings.inference_ms + detector_timings.download_ms +
      detector_timings.parse_ms + color_ms +
      std::chrono::duration<double, std::milli>(publish_done - publish_start).count());
  }

  ++frame_count_;
  stage_totals_.capture_ms += capture_ms;
  stage_totals_.preprocess_ms += detector_timings.preprocess_ms;
  stage_totals_.upload_ms += detector_timings.upload_ms;
  stage_totals_.inference_ms += detector_timings.inference_ms;
  stage_totals_.download_ms += detector_timings.download_ms;
  stage_totals_.parse_ms += detector_timings.parse_ms;
  stage_totals_.color_ms += color_ms;
  stage_totals_.publish_ms += std::chrono::duration<double, std::milli>(
    publish_done - publish_start).count();
  stage_totals_.preview_ms += preview_ms;

  if (frame_count_ % stage_log_interval_ == 0) {
    log_stage_timing();
  }
}

void ArmorDetectorNode::publish_armors(
  const std::vector<Candidate> & candidates,
  const std::string & source_id,
  const rclcpp::Time & stamp,
  int image_width,
  int image_height)
{
  cv_auto_aim_interfaces::msg::Armors msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = source_id;
  msg.source_id = source_id;
  msg.armors.reserve(candidates.size());
  for (const auto & candidate : candidates) {
    cv_auto_aim_interfaces::msg::Armor armor;
    armor.bbox_xyxy = {
      candidate.bbox_source.x,
      candidate.bbox_source.y,
      candidate.bbox_source.x + candidate.bbox_source.width,
      candidate.bbox_source.y + candidate.bbox_source.height};
    armor.center_x = candidate.center_source.x;
    armor.center_y = candidate.center_source.y;
    armor.confidence = candidate.confidence;
    armor.class_id = candidate.class_id;
    armor.is_enemy = candidate.is_enemy;
    armor.has_obb_angle = candidate.obb_angle.has_value();
    armor.obb_angle = candidate.obb_angle.value_or(0.0F);
    armor.image_width = image_width;
    armor.image_height = image_height;
    armor.has_corners = candidate.has_corners;
    if (candidate.has_corners) {
      for (std::size_t i = 0; i < candidate.corners_source.size(); ++i) {
        armor.corners[i].x = candidate.corners_source[i].x;
        armor.corners[i].y = candidate.corners_source[i].y;
        armor.corners[i].z = 0.0F;
      }
    }
    const double aspect = static_cast<double>(std::max(1, candidate.bbox_source.width)) /
      static_cast<double>(std::max(1, candidate.bbox_source.height));
    armor.armor_type = aspect >= large_armor_min_aspect_ratio_ ?
      cv_auto_aim_interfaces::msg::Armor::ARMOR_LARGE :
      cv_auto_aim_interfaces::msg::Armor::ARMOR_SMALL;
    armor.armor_number = -1;
    const double principal_x = camera_calibration_.available ?
      camera_calibration_.model.camera_matrix(0, 2) :
      static_cast<double>(image_width) / 2.0;
    const double principal_y = camera_calibration_.available ?
      camera_calibration_.model.camera_matrix(1, 2) :
      static_cast<double>(image_height) / 2.0;
    armor.distance_to_image_center = static_cast<float>(std::hypot(
      static_cast<double>(candidate.center_source.x) - principal_x,
      static_cast<double>(candidate.center_source.y) - principal_y));
    if (const auto pose = solve_candidate_pose(candidate, image_width, image_height)) {
      armor.has_pose = true;
      armor.pose_camera = *pose;
    }
    msg.armors.push_back(std::move(armor));
  }
  armors_publisher_->publish(std::move(msg));
}

void ArmorDetectorNode::publish_debug_image(const FramePacket & packet, const rclcpp::Time & stamp)
{
  if (!debug_image_publisher_) {
    return;
  }
  auto msg = frame_to_image_message(packet, stamp);
  if (msg) {
    debug_image_publisher_->publish(std::move(msg));
  }
}

void ArmorDetectorNode::render_preview(
  const FramePacket & packet, const std::vector<Candidate> & candidates)
{
  if (preview_failed_) {
    return;
  }

  try {
    const double preview_scale = std::min(
      1.0,
      std::min(
        static_cast<double>(kPreviewMaxWidth) / std::max(1, packet.frame.cols),
        static_cast<double>(kPreviewMaxHeight) / std::max(1, packet.frame.rows)));
    cv::Mat preview;
    if (preview_scale < 1.0) {
      cv::resize(packet.frame, preview, cv::Size(), preview_scale, preview_scale, cv::INTER_AREA);
    } else {
      preview = packet.frame.clone();
    }
    const cv::Rect image_rect(0, 0, preview.cols, preview.rows);
    const auto scale_rect = [&](const cv::Rect & rect) {
        if (preview_scale >= 1.0) {
          return rect;
        }
        const int x1 = static_cast<int>(std::floor(rect.x * preview_scale));
        const int y1 = static_cast<int>(std::floor(rect.y * preview_scale));
        const int x2 = static_cast<int>(std::ceil((rect.x + rect.width) * preview_scale));
        const int y2 = static_cast<int>(std::ceil((rect.y + rect.height) * preview_scale));
        return cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
      };
    const auto scale_point = [&](const cv::Point2f & point) {
        return cv::Point(
          static_cast<int>(std::lround(point.x * preview_scale)),
          static_cast<int>(std::lround(point.y * preview_scale)));
      };
    for (const auto & candidate : candidates) {
      const cv::Scalar color = candidate.is_enemy ?
        cv::Scalar(0, 255, 0) :
        cv::Scalar(0, 165, 255);
      const cv::Rect bbox = scale_rect(candidate.bbox_source) & image_rect;
      if (bbox.width <= 0 || bbox.height <= 0) {
        continue;
      }
      cv::rectangle(preview, bbox, color, 2);
      if (candidate.has_corners) {
        std::vector<cv::Point> corners;
        corners.reserve(candidate.corners_source.size());
        for (const auto & corner : candidate.corners_source) {
          const cv::Point scaled = scale_point(corner);
          corners.emplace_back(
            std::clamp(
              scaled.x,
              image_rect.x,
              image_rect.x + image_rect.width - 1),
            std::clamp(
              scaled.y,
              image_rect.y,
              image_rect.y + image_rect.height - 1));
        }
        cv::polylines(preview, corners, true, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
      }
      const std::string label =
        std::to_string(candidate.class_id) + " " + cv::format("%.2f", candidate.confidence);
      cv::putText(
        preview,
        label,
        cv::Point(bbox.x, std::max(16, bbox.y - 4)),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv::LINE_AA);
    }

    if (show_tracker_target_overlay_ && preview_has_target_) {
      const int x1 = preview_target_bbox_xyxy_[0];
      const int y1 = preview_target_bbox_xyxy_[1];
      const int x2 = preview_target_bbox_xyxy_[2];
      const int y2 = preview_target_bbox_xyxy_[3];
      const cv::Rect target_rect =
        scale_rect(cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1))) & image_rect;
      if (target_rect.width > 0 && target_rect.height > 0) {
        const cv::Scalar target_color(0, 0, 255);
        cv::rectangle(preview, target_rect, target_color, 4);
        cv::circle(
          preview,
          cv::Point(target_rect.x + target_rect.width / 2, target_rect.y + target_rect.height / 2),
          5,
          target_color,
          -1,
          cv::LINE_AA);
        const std::string label = preview_target_track_id_ >= 0 ?
          "TARGET #" + std::to_string(preview_target_track_id_) :
          "TARGET";
        const cv::Point label_origin(target_rect.x, std::max(22, target_rect.y - 8));
        cv::putText(
          preview,
          label,
          label_origin,
          cv::FONT_HERSHEY_SIMPLEX,
          0.7,
          target_color,
          2,
          cv::LINE_AA);
      }
    }

    cv::putText(
      preview,
      "FPS " + cv::format("%.1f", preview_fps_),
      cv::Point(12, 28),
      cv::FONT_HERSHEY_SIMPLEX,
      0.8,
      cv::Scalar(255, 255, 255),
      2,
      cv::LINE_AA);

    cv::imshow(vis_window_name_, preview);
    cv::waitKey(1);
  } catch (const cv::Exception & exc) {
    preview_failed_ = true;
    RCLCPP_WARN(
      this->get_logger(),
      "Disabling detector preview after OpenCV display failure: %s",
      exc.what());
  }
}

std::vector<Candidate> ArmorDetectorNode::extract_candidates(
  const DetectionBatch & batch, const FramePacket * packet, double * color_ms) const
{
  // Rebuild source coordinates before color checks so downstream nodes never need to know
  // the model letterbox transform.
  std::vector<Candidate> candidates;
  candidates.reserve(batch.detections.size());
  for (const auto & detection : batch.detections) {
    const auto source_rect = unletterbox_rect(detection.bbox_model, batch.letterbox);
    const cv::Point2f center_source(
      static_cast<float>(source_rect.x) + static_cast<float>(source_rect.width) / 2.0F,
      static_cast<float>(source_rect.y) + static_cast<float>(source_rect.height) / 2.0F);

    candidates.push_back(Candidate{
      detection.bbox_model,
      source_rect,
      center_source,
      detection.confidence,
      detection.class_id,
      detection.track_id,
      true,
      detection.obb_angle,
      detection.has_corners,
      {}});
    if (detection.has_corners) {
      auto & candidate = candidates.back();
      for (std::size_t i = 0; i < detection.corners_model.size(); ++i) {
        candidate.corners_source[i] = unletterbox_point(detection.corners_model[i], batch.letterbox);
      }
    }
  }

  if (use_model_color_classes_) {
    for (auto & candidate : candidates) {
      candidate.is_enemy = class_id_matches_target_color(candidate.class_id, target_class_);
    }
    return candidates;
  }

  if (!enable_color_verification_ || candidates.empty() || packet == nullptr || packet->empty()) {
    return candidates;
  }

  std::sort(
    candidates.begin(), candidates.end(),
    [](const Candidate & lhs, const Candidate & rhs) {return lhs.confidence > rhs.confidence;});
  const auto check_count =
    static_cast<std::size_t>(std::min<int>(max_hsv_candidates_, static_cast<int>(candidates.size())));

  const auto color_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    auto & candidate = candidates[i];
    if (i >= check_count) {
      candidate.is_enemy = false;
      continue;
    }

    const auto clipped = candidate.bbox_source & cv::Rect(0, 0, packet->width(), packet->height());
    if (clipped.width <= 0 || clipped.height <= 0) {
      candidate.is_enemy = false;
      continue;
    }
    const auto roi = packet->frame(clipped);
    candidate.is_enemy = check_enemy(roi, target_class_).first;
  }
  const auto color_done = std::chrono::steady_clock::now();
  if (color_ms != nullptr) {
    *color_ms = std::chrono::duration<double, std::milli>(color_done - color_start).count();
  }

  return candidates;
}

std::string ArmorDetectorNode::resolve_default_model_path() const
{
  const auto cwd_candidate =
    std::filesystem::current_path() / "model_test" /
    "best-roboflow-nitish-obb.onnx_b1_gpu0_fp16.engine";
  if (std::filesystem::exists(cwd_candidate)) {
    return cwd_candidate.string();
  }
  return "/home/ultraviolet/CV_Detection/model_test/"
         "best-roboflow-nitish-obb.onnx_b1_gpu0_fp16.engine";
}

std::string ArmorDetectorNode::resolve_default_deepstream_config_path() const
{
  const auto cwd_candidate =
    std::filesystem::current_path() / "yolo_cpp" / "config" /
    "deepstream_bestmerge_nvinfer.txt";
  if (std::filesystem::exists(cwd_candidate)) {
    return cwd_candidate.string();
  }

  const auto package_candidate =
    std::filesystem::path(__FILE__).parent_path().parent_path() / "config" /
    "deepstream_bestmerge_nvinfer.txt";
  if (std::filesystem::exists(package_candidate)) {
    return package_candidate.string();
  }
  return cwd_candidate.string();
}

std::string ArmorDetectorNode::resolve_runtime_log_dir() const
{
  if (!runtime_log_dir_.empty()) {
    std::filesystem::create_directories(runtime_log_dir_);
    return runtime_log_dir_;
  }
  if (const char * env_dir = std::getenv("VISION_LOG_DIR")) {
    const std::string value(env_dir);
    if (!value.empty()) {
      std::filesystem::create_directories(value);
      return value;
    }
  }
  const auto log_dir = std::filesystem::current_path() / kVisionLogDirName;
  std::filesystem::create_directories(log_dir);
  return log_dir.string();
}

void ArmorDetectorNode::write_runtime_log_line(const std::string & line)
{
  if (!runtime_log_stream_.is_open()) {
    return;
  }
  runtime_log_stream_ << "ts=" << timestamp_now_iso8601_ms() << ' ' << line << '\n';
  runtime_log_stream_.flush();
}

void ArmorDetectorNode::log_runtime_note(const std::string & level, const std::string & message)
{
  write_runtime_log_line("level=" + level + " msg=" + message);
}

void ArmorDetectorNode::log_runtime_fps_once_per_second()
{
  const auto now = std::chrono::steady_clock::now();
  if (have_last_runtime_fps_log_time_) {
    const double elapsed_s =
      std::chrono::duration<double>(now - last_runtime_fps_log_time_).count();
    if (elapsed_s < 1.0) {
      return;
    }
  }
  have_last_runtime_fps_log_time_ = true;
  last_runtime_fps_log_time_ = now;

  std::ostringstream line;
  line << std::fixed << std::setprecision(2) << "fps=" << preview_fps_;
  write_runtime_log_line(line.str());
}

void ArmorDetectorNode::log_runtime_opponent_event(
  const Candidate & target, bool color_pass, double latency_ms)
{
  const int x1 = target.bbox_source.x;
  const int y1 = target.bbox_source.y;
  const int x2 = target.bbox_source.x + target.bbox_source.width;
  const int y2 = target.bbox_source.y + target.bbox_source.height;

  std::ostringstream line;
  line << std::fixed << std::setprecision(2)
       << "fps=" << preview_fps_ << ' '
       << "bbox=" << x1 << ',' << y1 << ',' << x2 << ',' << y2 << ' ';
  line << std::setprecision(3)
       << "conf=" << target.confidence << ' '
       << "color_pass=" << (color_pass ? 1 : 0) << ' ';
  line << std::setprecision(2)
       << "latency_ms=" << latency_ms;
  write_runtime_log_line(line.str());
}

void ArmorDetectorNode::log_stage_timing()
{
  const double n = static_cast<double>(stage_log_interval_);
  double avg_fps = preview_fps_;
  const auto now = std::chrono::steady_clock::now();
  if (have_last_stage_log_wall_time_) {
    const double elapsed_s = std::chrono::duration<double>(now - last_stage_log_wall_time_).count();
    if (elapsed_s > 0.0) {
      avg_fps = n / elapsed_s;
    }
  }
  last_stage_log_wall_time_ = now;
  have_last_stage_log_wall_time_ = true;
  RCLCPP_INFO(
    this->get_logger(),
    "Detector timing mean over %d frames: fps=%.1f capture=%.2fms preprocess=%.2fms "
    "upload=%.2fms inference=%.2fms download=%.2fms parse=%.2fms color=%.2fms "
    "publish=%.2fms preview=%.2fms",
    stage_log_interval_,
    avg_fps,
    stage_totals_.capture_ms / n,
    stage_totals_.preprocess_ms / n,
    stage_totals_.upload_ms / n,
    stage_totals_.inference_ms / n,
    stage_totals_.download_ms / n,
    stage_totals_.parse_ms / n,
    stage_totals_.color_ms / n,
    stage_totals_.publish_ms / n,
    stage_totals_.preview_ms / n);
  stage_totals_ = StageTotals{};
}

}  // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::ArmorDetectorNode)
