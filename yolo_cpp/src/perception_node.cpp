/// @file perception_node.cpp
/// @brief Implements the legacy combined camera, detector, tracker, and 2D aim node.

#include "yolo_cpp/perception_node.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yolo_cpp
{

namespace
{

constexpr int kDefaultMaxHsvCandidates = 3;
constexpr int kDefaultMaxCoastFrames = 3;
constexpr double kPi = 3.14159265358979323846;
constexpr int kPreviewMaxWidth = 1280;
constexpr int kPreviewMaxHeight = 720;
constexpr char kVisionLogDirName[] = "vision_logs";

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

/// Declare parameters and start the combined legacy perception pipeline.
PerceptionNode::PerceptionNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("perception_node", options)
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
  this->declare_parameter<double>("conf", 0.30);
  this->declare_parameter<int>("tracker_timeout", 15);
  this->declare_parameter<double>("prediction_lookahead_steps", 0.5);
  this->declare_parameter<std::string>("inference_mode", "predict");
  this->declare_parameter<bool>("persist_tracking", true);
  this->declare_parameter<bool>("use_kalman", false);
  this->declare_parameter<bool>("enable_color_verification", true);
  this->declare_parameter<bool>("use_model_color_classes", true);
  this->declare_parameter<std::string>("target_color", "blue");
  this->declare_parameter<bool>("use_robot_id_targeting", false);
  this->declare_parameter<int>("enemy_color_red_value", 0);
  this->declare_parameter<int>("enemy_color_blue_value", 1);
  this->declare_parameter<int>("max_hsv_candidates", kDefaultMaxHsvCandidates);
  this->declare_parameter<int>("max_coast_frames", kDefaultMaxCoastFrames);
  this->declare_parameter<std::string>("runtime_log_dir", "");
  this->declare_parameter<double>("hfov_degrees", 79.2);
  this->declare_parameter<double>("vfov_degrees", 63.78);
  this->declare_parameter<std::string>("vis_window_name", "Perception Node");
  this->declare_parameter<bool>("debug_publish_image_raw", false);
  this->declare_parameter<bool>("pre_mux_resize", false);
  this->declare_parameter<int>("pre_mux_width", 0);
  this->declare_parameter<int>("pre_mux_height", 0);
  this->declare_parameter<bool>("deepstream_profile_stages", false);
  this->declare_parameter<int>("deepstream_profile_stage_interval", 300);
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
  conf_ = static_cast<float>(this->get_parameter("conf").as_double());
  tracker_timeout_ = this->get_parameter("tracker_timeout").as_int();
  lookahead_steps_ =
    static_cast<float>(this->get_parameter("prediction_lookahead_steps").as_double());
  inference_mode_ = this->get_parameter("inference_mode").as_string();
  persist_tracking_ = this->get_parameter("persist_tracking").as_bool();
  use_kalman_ = this->get_parameter("use_kalman").as_bool();
  enable_color_verification_ = this->get_parameter("enable_color_verification").as_bool();
  use_model_color_classes_ = this->get_parameter("use_model_color_classes").as_bool();
  target_color_ = this->get_parameter("target_color").as_string();
  use_robot_id_targeting_ = this->get_parameter("use_robot_id_targeting").as_bool();
  enemy_color_red_value_ = this->get_parameter("enemy_color_red_value").as_int();
  enemy_color_blue_value_ = this->get_parameter("enemy_color_blue_value").as_int();
  max_hsv_candidates_ = this->get_parameter("max_hsv_candidates").as_int();
  max_coast_frames_ = this->get_parameter("max_coast_frames").as_int();
  runtime_log_dir_ = this->get_parameter("runtime_log_dir").as_string();
  hfov_rad_ = static_cast<float>(
    this->get_parameter("hfov_degrees").as_double() * kPi / 180.0);
  vfov_rad_ = static_cast<float>(
    this->get_parameter("vfov_degrees").as_double() * kPi / 180.0);
  vis_window_name_ = this->get_parameter("vis_window_name").as_string();
  debug_publish_image_raw_ = this->get_parameter("debug_publish_image_raw").as_bool();
  pre_mux_resize_ = this->get_parameter("pre_mux_resize").as_bool();
  pre_mux_width_ = this->get_parameter("pre_mux_width").as_int();
  pre_mux_height_ = this->get_parameter("pre_mux_height").as_int();
  deepstream_profile_stages_ = this->get_parameter("deepstream_profile_stages").as_bool();
  deepstream_profile_stage_interval_ = std::max(
    1, static_cast<int>(this->get_parameter("deepstream_profile_stage_interval").as_int()));
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
    "robot_id", 10, std::bind(&PerceptionNode::id_listener_callback, this, std::placeholders::_1));
  attitude_flags_subscription_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
    "attitude_flags", 10,
    std::bind(&PerceptionNode::handle_attitude_flags, this, std::placeholders::_1));
  bbox_publisher_ =
    this->create_publisher<std_msgs::msg::Int32MultiArray>("bounding_box", 10);
  override_publisher_ = this->create_publisher<std_msgs::msg::Bool>("override", 10);
  angle_publisher_ =
    this->create_publisher<std_msgs::msg::Float32MultiArray>("bounding_box_angles", 10);
  obb_angle_publisher_ =
    this->create_publisher<std_msgs::msg::Float32>("obb_angle", 10);
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
    "Initialized PerceptionNode camera=%s backend=%s fps=%.2f debug_image_raw=%s",
    camera_config_.camera_type.c_str(),
    deepstream_runtime_ != nullptr ?
    deepstream_runtime_->backend_name().c_str() :
    detector_backend_->backend_name().c_str(),
    fps_,
    debug_publish_image_raw_ ? "true" : "false");
  log_runtime_note("INFO", "Initializing Modular YOLOv8 Node...");
  log_runtime_note("INFO", "Detector Initialized.");
  (void)show_;
  (void)persist_tracking_;
  (void)vis_window_name_;
}

PerceptionNode::~PerceptionNode()
{
  timer_.reset();
  if (show_ && !preview_failed_) {
    cv::destroyWindow(vis_window_name_);
  }
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

void PerceptionNode::id_listener_callback(const std_msgs::msg::Int32::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!use_robot_id_targeting_) {
    return;
  }
  if (stm_color_locked_) {
    return;
  }
  if (is_red_team(msg->data)) {
    target_class_ = BLUE_CLASS;
  } else if (is_blue_team(msg->data)) {
    target_class_ = RED_CLASS;
  }
}

void PerceptionNode::handle_attitude_flags(const std_msgs::msg::Int32MultiArray::SharedPtr msg)
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

void PerceptionNode::initialize_native_pipeline()
{
  camera_backend_ = create_camera_backend(camera_config_, this->get_logger());
  camera_backend_->initialize();
  fps_ = camera_backend_->actual_fps() > 0.0 ? camera_backend_->actual_fps() : 60.0;

  detector_backend_ = create_detector_backend(
    detector_backend_kind_, model_path_, task_, model_type_, half_, conf_);
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / std::max(1.0, fps_)),
    std::bind(&PerceptionNode::timer_callback, this));
}

void PerceptionNode::initialize_deepstream_runtime()
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
  config.source_width = camera_config_.basler_use_full_sensor ?
    camera_config_.basler_source_width :
    imgsz_width_;
  config.source_height = camera_config_.basler_use_full_sensor ?
    camera_config_.basler_source_height :
    imgsz_height_;
  config.source_fps = camera_config_.basler_fps;
  config.source_exposure_us = camera_config_.basler_exposure_us;
  config.source_gain_db = camera_config_.basler_gain_db;
  config.source_fps_uncapped = camera_config_.basler_uncapped_fps;
  config.source_use_full_sensor = camera_config_.basler_use_full_sensor;
  config.source_pixel_format = camera_config_.basler_pixel_format;
  config.pre_mux_resize = pre_mux_resize_;
  config.pre_mux_width = pre_mux_width_;
  config.pre_mux_height = pre_mux_height_;
  config.profile_stages = deepstream_profile_stages_;
  config.profile_stage_interval = deepstream_profile_stage_interval_;
  config.flip_source = camera_config_.basler_flip;
  config.conf_threshold = conf_;
  config.half = half_;
  config.enable_color_verification = enable_color_verification_ && !use_model_color_classes_;
  config.debug_publish_image_raw = debug_publish_image_raw_ || show_;

  deepstream_runtime_ = std::make_unique<DeepStreamRuntime>(
    this->get_logger(),
    config,
    [this](DeepStreamRuntime::Result && result) {
      this->handle_deepstream_result(std::move(result));
    });
  fps_ = deepstream_runtime_->source_actual_fps() > 0.0 ?
    deepstream_runtime_->source_actual_fps() :
    static_cast<double>(std::max(1, camera_config_.basler_fps));
}

void PerceptionNode::timer_callback()
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

void PerceptionNode::handle_deepstream_result(DeepStreamRuntime::Result result)
{
  const FramePacket * packet = result.source_packet.has_value() ? &(*result.source_packet) : nullptr;
  process_detection_cycle(result.batch, packet, result.timings, result.capture_ms);
}

void PerceptionNode::process_detection_cycle(
  const DetectionBatch & batch,
  const FramePacket * packet,
  const DetectorStageTimings & detector_timings,
  double capture_ms)
{
  // The combined node keeps direct command topics alive while the modular stack evolves.
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (debug_publish_image_raw_ && packet != nullptr) {
    publish_debug_image(*packet);
  }

  double dt = 1.0 / 60.0;
  const auto now = this->get_clock()->now();
  if (have_last_frame_time_) {
    dt = (now - last_frame_time_).seconds();
    if (dt <= 0.0) {
      dt = 0.001;
    }
  }
  last_frame_time_ = now;
  have_last_frame_time_ = true;
  {
    const double instant_fps = 1.0 / std::max(dt, 1e-6);
    preview_fps_ = preview_fps_ <= 0.0 ? instant_fps : (0.9 * preview_fps_) + (0.1 * instant_fps);
  }

  double color_ms = 0.0;
  const auto tracking_start = std::chrono::steady_clock::now();
  auto candidates = rank_candidates(extract_candidates(batch, packet, &color_ms));
  const auto detected_opponent = select_target(candidates);
  std::optional<Candidate> selected_target;
  bool published = false;
  if (use_kalman_) {
    published = run_kalman_pipeline(candidates, dt, batch.letterbox);
  } else {
    selected_target = detected_opponent;
    if (selected_target) {
      publish_target(*selected_target, batch.letterbox);
      published = true;
    }
  }
  const auto tracking_done = std::chrono::steady_clock::now();

  const auto publish_start = std::chrono::steady_clock::now();
  if (!published) {
    const int src_w = batch.letterbox.src_w > 0 ?
      batch.letterbox.src_w :
      (packet != nullptr ? packet->width() : imgsz_width_);
    const int src_h = batch.letterbox.src_h > 0 ?
      batch.letterbox.src_h :
      (packet != nullptr ? packet->height() : imgsz_height_);
    publish_no_target(src_w, src_h);
  }
  const auto publish_done = std::chrono::steady_clock::now();

  if (show_ && packet != nullptr && !packet->empty()) {
    render_preview(*packet, candidates, selected_target);
  }

  log_runtime_fps_once_per_second();
  if (detected_opponent.has_value() && detected_opponent->is_enemy) {
    log_runtime_opponent_event(*detected_opponent, true, capture_ms + detector_timings.preprocess_ms +
      detector_timings.upload_ms + detector_timings.inference_ms + detector_timings.download_ms +
      detector_timings.parse_ms + std::chrono::duration<double, std::milli>(
      tracking_done - tracking_start).count() +
      std::chrono::duration<double, std::milli>(publish_done - publish_start).count());
  }

  ++frame_count_;
  stage_totals_.capture_ms += capture_ms;
  stage_totals_.preprocess_ms += detector_timings.preprocess_ms;
  stage_totals_.upload_ms += detector_timings.upload_ms;
  stage_totals_.inference_ms += detector_timings.inference_ms;
  stage_totals_.download_ms += detector_timings.download_ms;
  stage_totals_.parse_ms += detector_timings.parse_ms;
  stage_totals_.tracking_ms += std::chrono::duration<double, std::milli>(
    tracking_done - tracking_start).count();
  stage_totals_.color_ms += color_ms;
  stage_totals_.publish_ms += std::chrono::duration<double, std::milli>(
    publish_done - publish_start).count();

  if (frame_count_ % stage_log_interval_ == 0) {
    log_stage_timing();
  }
}

void PerceptionNode::publish_debug_image(const FramePacket & packet)
{
  if (!debug_image_publisher_) {
    return;
  }
  auto msg = frame_to_image_message(packet, this->now());
  if (msg) {
    debug_image_publisher_->publish(std::move(msg));
  }
}

void PerceptionNode::render_preview(
  const FramePacket & packet,
  const std::vector<Candidate> & candidates,
  const std::optional<Candidate> & target)
{
  if (preview_failed_) {
    return;
  }

  try {
    cv::Mat preview = packet.frame.clone();
    for (const auto & candidate : candidates) {
      const cv::Scalar color = candidate.is_enemy ?
        cv::Scalar(0, 255, 0) :
        cv::Scalar(0, 165, 255);
      cv::rectangle(preview, candidate.bbox_source, color, 2);
      const std::string label =
        std::to_string(candidate.class_id) + " " + cv::format("%.2f", candidate.confidence);
      cv::putText(
        preview,
        label,
        cv::Point(candidate.bbox_source.x, std::max(16, candidate.bbox_source.y - 4)),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv::LINE_AA);
    }

    if (target.has_value()) {
      cv::rectangle(preview, target->bbox_source, cv::Scalar(0, 0, 255), 3);
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

    const double scale = std::min(
      1.0,
      std::min(
        static_cast<double>(kPreviewMaxWidth) / std::max(1, preview.cols),
        static_cast<double>(kPreviewMaxHeight) / std::max(1, preview.rows)));
    if (scale < 1.0) {
      cv::resize(preview, preview, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    cv::imshow(vis_window_name_, preview);
    cv::waitKey(1);
  } catch (const cv::Exception & exc) {
    preview_failed_ = true;
    RCLCPP_WARN(
      this->get_logger(),
      "Disabling show preview after OpenCV display failure: %s",
      exc.what());
  }
}

std::vector<Candidate> PerceptionNode::extract_candidates(
  const DetectionBatch & batch, const FramePacket * packet, double * color_ms) const
{
  // Convert model-space detections back to source pixels before ranking and optional tracking.
  std::vector<Candidate> candidates;
  candidates.reserve(batch.detections.size());
  for (const auto & detection : batch.detections) {
    auto track_id = detection.track_id;
    if (use_kalman_ && !track_id.has_value()) {
      if (!candidates.empty()) {
        continue;
      }
      track_id = 0;
    }
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
      track_id,
      true,
      detection.obb_angle});
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

std::vector<Candidate> PerceptionNode::rank_candidates(std::vector<Candidate> candidates) const
{
  std::sort(
    candidates.begin(), candidates.end(),
    [this](const Candidate & lhs, const Candidate & rhs) {
      const double lhs_dist = std::hypot(
        static_cast<double>(lhs.center_source.x) - static_cast<double>(imgsz_width_) / 2.0,
        static_cast<double>(lhs.center_source.y) - static_cast<double>(imgsz_height_) / 2.0);
      const double rhs_dist = std::hypot(
        static_cast<double>(rhs.center_source.x) - static_cast<double>(imgsz_width_) / 2.0,
        static_cast<double>(rhs.center_source.y) - static_cast<double>(imgsz_height_) / 2.0);
      if (lhs.is_enemy != rhs.is_enemy) {
        return lhs.is_enemy && !rhs.is_enemy;
      }
      if (lhs.confidence != rhs.confidence) {
        return lhs.confidence > rhs.confidence;
      }
      return lhs_dist < rhs_dist;
    });
  return candidates;
}

std::optional<Candidate> PerceptionNode::select_target(const std::vector<Candidate> & candidates) const
{
  for (const auto & candidate : candidates) {
    if (candidate.is_enemy) {
      return candidate;
    }
  }
  return std::nullopt;
}

void PerceptionNode::publish_target(const Candidate & target, const LetterboxMetadata & lb)
{
  bbox_msg_.data = {
    target.bbox_source.x,
    target.bbox_source.y,
    target.bbox_source.x + target.bbox_source.width,
    target.bbox_source.y + target.bbox_source.height};
  override_msg_.data = true;
  bbox_publisher_->publish(bbox_msg_);
  override_publisher_->publish(override_msg_);

  auto [yaw, pitch] = calculate_angles(target.bbox_source, lb.src_w, lb.src_h, hfov_rad_, vfov_rad_);
  angle_msg_.data = {yaw, pitch};
  angle_publisher_->publish(angle_msg_);

  obb_angle_msg_.data = target.obb_angle.value_or(0.0F);
  obb_angle_publisher_->publish(obb_angle_msg_);
}

void PerceptionNode::publish_no_target(int src_w, int src_h)
{
  const int cx = src_w > 0 ? src_w / 2 : imgsz_width_ / 2;
  const int cy = src_h > 0 ? src_h / 2 : imgsz_height_ / 2;
  bbox_msg_.data = {cx, cy, cx, cy};
  override_msg_.data = false;
  angle_msg_.data = {0.0F, 0.0F};
  obb_angle_msg_.data = 0.0F;

  bbox_publisher_->publish(bbox_msg_);
  override_publisher_->publish(override_msg_);
  angle_publisher_->publish(angle_msg_);
  obb_angle_publisher_->publish(obb_angle_msg_);
}

bool PerceptionNode::publish_kalman_target(
  double xf, double yf, double width, double height, const LetterboxMetadata & lb,
  const std::optional<float> & obb_angle)
{
  const int bx1 = std::max(0, std::min(static_cast<int>(xf - width / 2.0), imgsz_width_ - 1));
  const int by1 = std::max(0, std::min(static_cast<int>(yf - height / 2.0), imgsz_height_ - 1));
  const int bx2 = std::max(0, std::min(static_cast<int>(xf + width / 2.0), imgsz_width_ - 1));
  const int by2 = std::max(0, std::min(static_cast<int>(yf + height / 2.0), imgsz_height_ - 1));
  if (bx1 >= bx2 || by1 >= by2) {
    return false;
  }

  Candidate projected{};
  projected.bbox_model = cv::Rect(bx1, by1, bx2 - bx1, by2 - by1);
  projected.bbox_source = unletterbox_rect(projected.bbox_model, lb);
  projected.obb_angle = obb_angle;
  publish_target(projected, lb);
  return true;
}

bool PerceptionNode::run_kalman_pipeline(
  const std::vector<Candidate> & candidates, double dt, const LetterboxMetadata & lb)
{
  std::set<int> current_ids;
  bool published = false;

  for (const auto & candidate : candidates) {
    if (!candidate.is_enemy || !candidate.track_id.has_value()) {
      continue;
    }
    current_ids.insert(*candidate.track_id);
    const double cx =
      static_cast<double>(candidate.bbox_model.x) +
      (static_cast<double>(candidate.bbox_model.width) / 2.0);
    const double cy =
      static_cast<double>(candidate.bbox_model.y) +
      (static_cast<double>(candidate.bbox_model.height) / 2.0);
    const double width = static_cast<double>(candidate.bbox_model.width);
    const double height = static_cast<double>(candidate.bbox_model.height);

    auto tracker_it = active_trackers_.find(*candidate.track_id);
    if (tracker_it == active_trackers_.end()) {
      tracker_it = active_trackers_.emplace(
        *candidate.track_id, TrackerWrapper(ArmorPlateTracker(cx, cy, width, height))).first;
    } else {
      tracker_it->second.tracker.predict(dt);
      tracker_it->second.tracker.update(cx, cy, width, height);
    }

    tracker_it->second.lost_frames = 0;
    tracker_it->second.obb_angle = candidate.obb_angle;
  }

  std::vector<int> ids_to_remove;
  for (auto & [track_id, wrapper] : active_trackers_) {
    if (current_ids.count(track_id) == 0) {
      ++wrapper.lost_frames;
    }
    if (wrapper.lost_frames >= tracker_timeout_) {
      ids_to_remove.push_back(track_id);
      continue;
    }

    if (!published && wrapper.lost_frames <= max_coast_frames_) {
      const auto & state = wrapper.tracker.state();
      const double t_look = static_cast<double>(lookahead_steps_) * dt;
      const double xf = state[0] + (state[2] * t_look) + (0.5 * state[4] * t_look * t_look);
      const double yf = state[1] + (state[3] * t_look) + (0.5 * state[5] * t_look * t_look);
      if (publish_kalman_target(
          xf, yf, wrapper.tracker.width(), wrapper.tracker.height(), lb, wrapper.obb_angle))
      {
        published = true;
      }
    }
  }

  for (const int id : ids_to_remove) {
    active_trackers_.erase(id);
  }
  return published;
}

std::string PerceptionNode::resolve_default_model_path() const
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

std::string PerceptionNode::resolve_default_deepstream_config_path() const
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

std::string PerceptionNode::resolve_runtime_log_dir() const
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

void PerceptionNode::write_runtime_log_line(const std::string & line)
{
  if (!runtime_log_stream_.is_open()) {
    return;
  }
  runtime_log_stream_ << "ts=" << timestamp_now_iso8601_ms() << ' ' << line << '\n';
  runtime_log_stream_.flush();
}

void PerceptionNode::log_runtime_note(const std::string & level, const std::string & message)
{
  write_runtime_log_line("level=" + level + " msg=" + message);
}

void PerceptionNode::log_runtime_fps_once_per_second()
{
  const auto now = std::chrono::steady_clock::now();
  if (have_last_runtime_fps_log_time_) {
    const double elapsed_s = std::chrono::duration<double>(now - last_runtime_fps_log_time_).count();
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

void PerceptionNode::log_runtime_opponent_event(
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

void PerceptionNode::log_stage_timing()
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
    "Stage timing mean over %d frames: fps=%.1f capture=%.2fms preprocess=%.2fms upload=%.2fms "
    "inference=%.2fms download=%.2fms parse=%.2fms tracking=%.2fms color=%.2fms publish=%.2fms",
    stage_log_interval_,
    avg_fps,
    stage_totals_.capture_ms / n,
    stage_totals_.preprocess_ms / n,
    stage_totals_.upload_ms / n,
    stage_totals_.inference_ms / n,
    stage_totals_.download_ms / n,
    stage_totals_.parse_ms / n,
    stage_totals_.tracking_ms / n,
    stage_totals_.color_ms / n,
    stage_totals_.publish_ms / n);
  stage_totals_ = StageTotals{};
}

}  // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::PerceptionNode)
