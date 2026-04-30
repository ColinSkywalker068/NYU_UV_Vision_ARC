#include "yolov8_ros_cpp/perception_node.hpp"

#include <opencv2/highgui.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yolov8_ros_cpp
{

namespace
{

constexpr int kDefaultMaxHsvCandidates = 3;
constexpr int kDefaultMaxCoastFrames = 3;
constexpr double kPi = 3.14159265358979323846;
constexpr int kPreviewMaxWidth = 1280;
constexpr int kPreviewMaxHeight = 720;

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
  this->declare_parameter<double>("conf", 0.45);
  this->declare_parameter<int>("tracker_timeout", 15);
  this->declare_parameter<double>("prediction_lookahead_steps", 0.5);
  this->declare_parameter<std::string>("inference_mode", "predict");
  this->declare_parameter<bool>("persist_tracking", true);
  this->declare_parameter<bool>("use_kalman", false);
  this->declare_parameter<bool>("enable_color_verification", true);
  this->declare_parameter<std::string>("target_color", "blue");
  this->declare_parameter<bool>("use_robot_id_targeting", false);
  this->declare_parameter<int>("enemy_color_red_value", 0);
  this->declare_parameter<int>("enemy_color_blue_value", 1);
  this->declare_parameter<int>("max_hsv_candidates", kDefaultMaxHsvCandidates);
  this->declare_parameter<int>("max_coast_frames", kDefaultMaxCoastFrames);
  this->declare_parameter<double>("hfov_degrees", 79.2);
  this->declare_parameter<double>("vfov_degrees", 63.78);
  this->declare_parameter<std::string>("vis_window_name", "Perception Node");
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
  conf_ = static_cast<float>(this->get_parameter("conf").as_double());
  tracker_timeout_ = this->get_parameter("tracker_timeout").as_int();
  lookahead_steps_ =
    static_cast<float>(this->get_parameter("prediction_lookahead_steps").as_double());
  inference_mode_ = this->get_parameter("inference_mode").as_string();
  persist_tracking_ = this->get_parameter("persist_tracking").as_bool();
  use_kalman_ = this->get_parameter("use_kalman").as_bool();
  enable_color_verification_ = this->get_parameter("enable_color_verification").as_bool();
  target_color_ = this->get_parameter("target_color").as_string();
  use_robot_id_targeting_ = this->get_parameter("use_robot_id_targeting").as_bool();
  enemy_color_red_value_ = this->get_parameter("enemy_color_red_value").as_int();
  enemy_color_blue_value_ = this->get_parameter("enemy_color_blue_value").as_int();
  max_hsv_candidates_ = this->get_parameter("max_hsv_candidates").as_int();
  max_coast_frames_ = this->get_parameter("max_coast_frames").as_int();
  hfov_rad_ = static_cast<float>(
    this->get_parameter("hfov_degrees").as_double() * kPi / 180.0);
  vfov_rad_ = static_cast<float>(
    this->get_parameter("vfov_degrees").as_double() * kPi / 180.0);
  vis_window_name_ = this->get_parameter("vis_window_name").as_string();
  debug_publish_image_raw_ = this->get_parameter("debug_publish_image_raw").as_bool();
  stage_log_interval_ = std::max(
    1, static_cast<int>(this->get_parameter("stage_log_interval").as_int()));

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
  (void)show_;
  (void)persist_tracking_;
  (void)vis_window_name_;
}

PerceptionNode::~PerceptionNode()
{
  if (show_ && !preview_failed_) {
    cv::destroyWindow(vis_window_name_);
  }
  deepstream_runtime_.reset();
  if (camera_backend_) {
    camera_backend_->release();
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
  if (enemy_color == enemy_color_red_value_) {
    target_class_ = RED_CLASS;
    stm_color_locked_ = true;
  } else if (enemy_color == enemy_color_blue_value_) {
    target_class_ = BLUE_CLASS;
    stm_color_locked_ = true;
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
  config.source_width = camera_config_.imgsz_width;
  config.source_height = camera_config_.imgsz_height;
  config.source_fps = camera_config_.basler_fps;
  config.conf_threshold = conf_;
  config.half = half_;
  config.enable_color_verification = enable_color_verification_;
  config.debug_publish_image_raw = debug_publish_image_raw_;

  fps_ = static_cast<double>(std::max(1, camera_config_.basler_fps));
  deepstream_runtime_ = std::make_unique<DeepStreamRuntime>(
    this->get_logger(),
    config,
    [this](DeepStreamRuntime::Result && result) {
      this->handle_deepstream_result(std::move(result));
    });
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

  double color_ms = 0.0;
  const auto tracking_start = std::chrono::steady_clock::now();
  auto candidates = rank_candidates(extract_candidates(batch, packet, &color_ms));
  std::optional<Candidate> selected_target;
  bool published = false;
  if (use_kalman_) {
    published = run_kalman_pipeline(candidates, dt, batch.letterbox);
  } else {
    selected_target = select_target(candidates);
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
  const auto cwd_candidate = std::filesystem::current_path() / "model_test" / "bestmerge.engine";
  if (std::filesystem::exists(cwd_candidate)) {
    return cwd_candidate.string();
  }
  return "/home/ultraviolet/CV_Detection/model_test/bestmerge.engine";
}

std::string PerceptionNode::resolve_default_deepstream_config_path() const
{
  const auto cwd_candidate =
    std::filesystem::current_path() / "yolov8_ros_cpp" / "config" /
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

void PerceptionNode::log_stage_timing()
{
  const double n = static_cast<double>(stage_log_interval_);
  RCLCPP_INFO(
    this->get_logger(),
    "Stage timing mean over %d frames: capture=%.2fms preprocess=%.2fms upload=%.2fms "
    "inference=%.2fms download=%.2fms parse=%.2fms tracking=%.2fms color=%.2fms publish=%.2fms",
    stage_log_interval_,
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

}  // namespace yolov8_ros_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolov8_ros_cpp::PerceptionNode)
