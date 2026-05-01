#include "yolo_cpp/armor_detector_node.hpp"

#include "cv_auto_aim_interfaces/msg/armor.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace yolo_cpp
{

namespace
{

constexpr int kDefaultMaxHsvCandidates = 3;
constexpr int kPreviewMaxWidth = 1280;
constexpr int kPreviewMaxHeight = 720;

bool allow_unsafe_preview_window()
{
  const char * env = std::getenv("YOLO_CPP_ENABLE_UNSAFE_OPENCV_PREVIEW");
  return env != nullptr && std::string(env) == "1";
}

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
  this->declare_parameter<double>("conf", 0.30);
  this->declare_parameter<std::string>("inference_mode", "predict");
  this->declare_parameter<bool>("enable_color_verification", true);
  this->declare_parameter<std::string>("target_color", "blue");
  this->declare_parameter<bool>("use_robot_id_targeting", false);
  this->declare_parameter<double>("hfov_degrees", 79.2);
  this->declare_parameter<double>("vfov_degrees", 63.78);
  this->declare_parameter<int>("enemy_color_red_value", 0);
  this->declare_parameter<int>("enemy_color_blue_value", 1);
  this->declare_parameter<int>("max_hsv_candidates", kDefaultMaxHsvCandidates);
  this->declare_parameter<std::string>("vis_window_name", "Armor Detector");
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
  inference_mode_ = this->get_parameter("inference_mode").as_string();
  enable_color_verification_ = this->get_parameter("enable_color_verification").as_bool();
  target_color_ = this->get_parameter("target_color").as_string();
  use_robot_id_targeting_ = this->get_parameter("use_robot_id_targeting").as_bool();
  enemy_color_red_value_ = this->get_parameter("enemy_color_red_value").as_int();
  enemy_color_blue_value_ = this->get_parameter("enemy_color_blue_value").as_int();
  max_hsv_candidates_ = this->get_parameter("max_hsv_candidates").as_int();
  vis_window_name_ = this->get_parameter("vis_window_name").as_string();
  debug_publish_image_raw_ = this->get_parameter("debug_publish_image_raw").as_bool();
  stage_log_interval_ = std::max(
    1, static_cast<int>(this->get_parameter("stage_log_interval").as_int()));

  if (show_ && !allow_unsafe_preview_window()) {
    RCLCPP_WARN(
      this->get_logger(),
      "Disabling OpenCV preview window because it can trigger fatal GDK/X11 errors and interrupt camera cleanup. "
      "Use debug_publish_image_raw instead, or set YOLO_CPP_ENABLE_UNSAFE_OPENCV_PREVIEW=1 to force the old preview path.");
    show_ = false;
    debug_publish_image_raw_ = true;
  }

  target_class_ = (target_color_ == "red") ? RED_CLASS : BLUE_CLASS;

  robot_id_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
    "robot_id", 10, std::bind(&ArmorDetectorNode::id_listener_callback, this, std::placeholders::_1));
  attitude_flags_subscription_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
    "attitude_flags", 10,
    std::bind(&ArmorDetectorNode::handle_attitude_flags, this, std::placeholders::_1));
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
}

ArmorDetectorNode::~ArmorDetectorNode()
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
  if (enemy_color == enemy_color_red_value_) {
    target_class_ = RED_CLASS;
    stm_color_locked_ = true;
  } else if (enemy_color == enemy_color_blue_value_) {
    target_class_ = BLUE_CLASS;
    stm_color_locked_ = true;
  }
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
  const std::string source_id = packet != nullptr && !packet->source_id.empty() ?
    packet->source_id :
    camera_config_.camera_type;
  const int image_width = batch.letterbox.src_w > 0 ?
    batch.letterbox.src_w :
    (packet != nullptr ? packet->width() : imgsz_width_);
  const int image_height = batch.letterbox.src_h > 0 ?
    batch.letterbox.src_h :
    (packet != nullptr ? packet->height() : imgsz_height_);
  publish_armors(candidates, source_id, stamp, image_width, image_height);
  const auto publish_done = std::chrono::steady_clock::now();

  if (show_ && packet != nullptr && !packet->empty()) {
    render_preview(*packet, candidates);
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
      "Disabling detector preview after OpenCV display failure: %s",
      exc.what());
  }
}

std::vector<Candidate> ArmorDetectorNode::extract_candidates(
  const DetectionBatch & batch, const FramePacket * packet, double * color_ms) const
{
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

std::string ArmorDetectorNode::resolve_default_model_path() const
{
  const auto cwd_candidate = std::filesystem::current_path() / "model_test" / "bestmerge.engine";
  if (std::filesystem::exists(cwd_candidate)) {
    return cwd_candidate.string();
  }
  return "/home/ultraviolet/CV_Detection/model_test/bestmerge.engine";
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
    "upload=%.2fms inference=%.2fms download=%.2fms parse=%.2fms color=%.2fms publish=%.2fms",
    stage_log_interval_,
    avg_fps,
    stage_totals_.capture_ms / n,
    stage_totals_.preprocess_ms / n,
    stage_totals_.upload_ms / n,
    stage_totals_.inference_ms / n,
    stage_totals_.download_ms / n,
    stage_totals_.parse_ms / n,
    stage_totals_.color_ms / n,
    stage_totals_.publish_ms / n);
  stage_totals_ = StageTotals{};
}

}  // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::ArmorDetectorNode)
