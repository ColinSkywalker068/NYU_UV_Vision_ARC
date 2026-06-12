/// @file armor_detector_node.cpp
/// @brief Implements camera capture, YOLO armor detection, color filtering, and
/// Armor messages.

#include "yolo_cpp/armor_detector_node.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <rclcpp_components/register_node_macro.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "cv_auto_aim_interfaces/msg/armor.hpp"
#include "vision_node_helpers.hpp"

namespace yolo_cpp {

namespace {

constexpr int kDefaultMaxHsvCandidates = 3;
constexpr int kPreviewMaxWidth = 1280;
constexpr int kPreviewMaxHeight = 720;
constexpr char kVisionLogDirName[] = "vision_logs";
constexpr char kTargetFrameDirName[] = "pictures";
constexpr double kPi = 3.14159265358979323846;

std::string json_escape(const std::string &value) {
  std::ostringstream escaped;
  for (const char ch : value) {
    switch (ch) {
    case '"':
      escaped << "\\\"";
      break;
    case '\\':
      escaped << "\\\\";
      break;
    case '\b':
      escaped << "\\b";
      break;
    case '\f':
      escaped << "\\f";
      break;
    case '\n':
      escaped << "\\n";
      break;
    case '\r':
      escaped << "\\r";
      break;
    case '\t':
      escaped << "\\t";
      break;
    default:
      if (static_cast<unsigned char>(ch) < 0x20U) {
        escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<int>(static_cast<unsigned char>(ch)) << std::dec
                << std::setfill(' ');
      } else {
        escaped << ch;
      }
      break;
    }
  }
  return escaped.str();
}

std::optional<int> bayer_to_bgr_code(PixelFormat format) {
  switch (format) {
  case PixelFormat::kBayerRggb8:
    return cv::COLOR_BayerBG2BGR;
  case PixelFormat::kBayerBggr8:
    return cv::COLOR_BayerRG2BGR;
  case PixelFormat::kBayerGrbg8:
    return cv::COLOR_BayerGB2BGR;
  case PixelFormat::kBayerGbrg8:
    return cv::COLOR_BayerGR2BGR;
  default:
    return std::nullopt;
  }
}

cv::Mat make_preview_frame(const FramePacket &packet) {
  cv::Mat preview;
  if (const auto code = bayer_to_bgr_code(packet.pixel_format);
      code.has_value() && packet.frame.type() == CV_8UC1) {
    cv::cvtColor(packet.frame, preview, *code);
  } else if (packet.pixel_format == PixelFormat::kRgb8 &&
             packet.frame.type() == CV_8UC3) {
    cv::cvtColor(packet.frame, preview, cv::COLOR_RGB2BGR);
  } else if (packet.pixel_format == PixelFormat::kYuyv &&
             packet.frame.type() == CV_8UC2) {
    cv::cvtColor(packet.frame, preview, cv::COLOR_YUV2BGR_YUY2);
  } else {
    preview = packet.frame.clone();
  }

  if (packet.flip_180 && !preview.empty()) {
    cv::flip(preview, preview, -1);
  }
  return preview;
}

} // namespace

/// Declare parameters, choose the inference runtime, and start capture
/// processing.
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions &options)
    : rclcpp::Node("armor_detector_node", options) {
  camera_config_ = load_camera_config(*this);

  const std::string default_model = resolve_default_model_path();
  const std::string default_deepstream_config =
      resolve_default_deepstream_config_path();
  this->declare_parameter<std::string>("model_path", default_model);
  this->declare_parameter<std::string>("task", "obb");
  this->declare_parameter<std::string>("type", "yolo");
  this->declare_parameter<std::string>("detector_backend", "native_tensorrt");
  this->declare_parameter<std::string>("deepstream_config_path",
                                       default_deepstream_config);
  this->declare_parameter<bool>("allow_native_tensorrt_fallback", false);
  this->declare_parameter<bool>("half", true);
  this->declare_parameter<bool>("show", false);
  this->declare_parameter<double>("conf", 0.30);
  this->declare_parameter<std::string>("inference_mode", "predict");
  this->declare_parameter<bool>("enable_color_verification", false);
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
  this->declare_parameter<std::vector<double>>("camera_distortion",
                                               std::vector<double>{});
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
  this->declare_parameter<std::string>("preview_target_topic",
                                       "tracker/target");
  this->declare_parameter<bool>("debug_publish_image_raw", false);
  this->declare_parameter<bool>("save_target_frames", false);
  this->declare_parameter<bool>("pre_mux_resize", false);
  this->declare_parameter<bool>("native_tensorrt_cuda_bayer_preprocess", false);
  this->declare_parameter<int>("pre_mux_width", 0);
  this->declare_parameter<int>("pre_mux_height", 0);
  this->declare_parameter<bool>("deepstream_profile_stages", false);
  this->declare_parameter<int>("deepstream_profile_stage_interval", 300);
  this->declare_parameter<std::string>("bayer_conversion_backend", "cuda");
  this->declare_parameter<std::string>("target_frame_save_dir", "");
  this->declare_parameter<int>("target_frame_save_max", 100);
  this->declare_parameter<int>("target_frame_jpeg_quality", 85);
  this->declare_parameter<int>("stage_log_interval", 300);
  this->declare_parameter<double>("preview_max_fps", 60.0);
  this->declare_parameter<bool>("preview_http_enabled", true);
  this->declare_parameter<std::string>("preview_http_bind_address", "0.0.0.0");
  this->declare_parameter<int>("preview_http_port", 8080);
  this->declare_parameter<int>("preview_http_jpeg_quality", 80);

  model_path_ = vision_node_helpers::resolve_model_path(
      this->get_parameter("model_path").as_string());
  task_ = this->get_parameter("task").as_string();
  model_type_ = this->get_parameter("type").as_string();
  detector_backend_kind_ = this->get_parameter("detector_backend").as_string();
  deepstream_config_path_ =
      this->get_parameter("deepstream_config_path").as_string();
  allow_native_tensorrt_fallback_ =
      this->get_parameter("allow_native_tensorrt_fallback").as_bool();
  half_ = this->get_parameter("half").as_bool();
  imgsz_width_ = this->get_parameter("imgsz_width").as_int();
  imgsz_height_ = this->get_parameter("imgsz_height").as_int();
  show_ = this->get_parameter("show").as_bool();
  conf_ = static_cast<float>(this->get_parameter("conf").as_double());
  inference_mode_ = this->get_parameter("inference_mode").as_string();
  enable_color_verification_ =
      this->get_parameter("enable_color_verification").as_bool();
  use_model_color_classes_ =
      this->get_parameter("use_model_color_classes").as_bool();
  target_color_ = this->get_parameter("target_color").as_string();
  use_robot_id_targeting_ =
      this->get_parameter("use_robot_id_targeting").as_bool();
  hfov_rad_ = this->get_parameter("hfov_degrees").as_double() * kPi / 180.0;
  vfov_rad_ = this->get_parameter("vfov_degrees").as_double() * kPi / 180.0;
  enable_3d_pose_ = this->get_parameter("enable_3d_pose").as_bool();
  camera_frame_id_ = this->get_parameter("camera_frame_id").as_string();
  camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
  camera_fx_ = this->get_parameter("camera_fx").as_double();
  camera_fy_ = this->get_parameter("camera_fy").as_double();
  camera_cx_ = this->get_parameter("camera_cx").as_double();
  camera_cy_ = this->get_parameter("camera_cy").as_double();
  camera_distortion_ =
      this->get_parameter("camera_distortion").as_double_array();
  small_armor_width_m_ = this->get_parameter("small_armor_width_m").as_double();
  small_armor_height_m_ =
      this->get_parameter("small_armor_height_m").as_double();
  large_armor_width_m_ = this->get_parameter("large_armor_width_m").as_double();
  large_armor_height_m_ =
      this->get_parameter("large_armor_height_m").as_double();
  large_armor_min_aspect_ratio_ =
      this->get_parameter("large_armor_min_aspect_ratio").as_double();
  enemy_color_red_value_ =
      this->get_parameter("enemy_color_red_value").as_int();
  enemy_color_blue_value_ =
      this->get_parameter("enemy_color_blue_value").as_int();
  max_hsv_candidates_ = this->get_parameter("max_hsv_candidates").as_int();
  runtime_log_dir_ = this->get_parameter("runtime_log_dir").as_string();
  vis_window_name_ = this->get_parameter("vis_window_name").as_string();
  preview_target_topic_ =
      this->get_parameter("preview_target_topic").as_string();
  debug_publish_image_raw_ =
      this->get_parameter("debug_publish_image_raw").as_bool();
  save_target_frames_ = this->get_parameter("save_target_frames").as_bool();
  pre_mux_resize_ = this->get_parameter("pre_mux_resize").as_bool();
  native_tensorrt_cuda_bayer_preprocess_ =
      this->get_parameter("native_tensorrt_cuda_bayer_preprocess").as_bool();
  pre_mux_width_ = this->get_parameter("pre_mux_width").as_int();
  pre_mux_height_ = this->get_parameter("pre_mux_height").as_int();
  deepstream_profile_stages_ =
      this->get_parameter("deepstream_profile_stages").as_bool();
  deepstream_profile_stage_interval_ = std::max(
      1,
      static_cast<int>(
          this->get_parameter("deepstream_profile_stage_interval").as_int()));
  bayer_conversion_backend_ =
      this->get_parameter("bayer_conversion_backend").as_string();
  target_frame_save_dir_ =
      this->get_parameter("target_frame_save_dir").as_string();
  target_frame_save_max_ = std::max(
      0,
      static_cast<int>(this->get_parameter("target_frame_save_max").as_int()));
  target_frame_jpeg_quality_ =
      std::clamp(static_cast<int>(
                     this->get_parameter("target_frame_jpeg_quality").as_int()),
                 1, 100);
  stage_log_interval_ = std::max(
      1, static_cast<int>(this->get_parameter("stage_log_interval").as_int()));
  preview_max_fps_ =
      std::max(1.0, this->get_parameter("preview_max_fps").as_double());
  preview_http_enabled_ = this->get_parameter("preview_http_enabled").as_bool();
  preview_http_bind_address_ =
      this->get_parameter("preview_http_bind_address").as_string();
  preview_http_port_ = std::clamp(
      static_cast<int>(this->get_parameter("preview_http_port").as_int()), 1,
      65535);
  preview_http_jpeg_quality_ =
      std::clamp(static_cast<int>(
                     this->get_parameter("preview_http_jpeg_quality").as_int()),
                 1, 100);

  runtime_log_path_ = (std::filesystem::path(resolve_runtime_log_dir()) /
                       ("vision_runtime_" +
                        vision_node_helpers::timestamp_now_compact() + ".log"))
                          .string();
  runtime_log_stream_.open(runtime_log_path_, std::ios::app);
  if (!runtime_log_stream_.is_open()) {
    RCLCPP_WARN(this->get_logger(),
                "Failed to open runtime vision log file: %s",
                runtime_log_path_.c_str());
  } else {
    RCLCPP_INFO(this->get_logger(), "Runtime logs: %s",
                runtime_log_path_.c_str());
  }
  if (save_target_frames_) {
    try {
      target_frame_save_dir_ = resolve_target_frame_save_dir();
      target_frame_index_path_ =
          (std::filesystem::path(target_frame_save_dir_) /
           ("target_frames_" + vision_node_helpers::timestamp_now_compact() +
            ".csv"))
              .string();
      target_frame_index_stream_.open(target_frame_index_path_, std::ios::app);
      if (!target_frame_index_stream_.is_open()) {
        RCLCPP_WARN(this->get_logger(),
                    "Failed to open target frame index: %s; disabling target "
                    "frame capture",
                    target_frame_index_path_.c_str());
        save_target_frames_ = false;
      } else {
        target_frame_index_stream_
            << "filename,stamp_ns,source_id,image_width,image_height,track_id,"
            << "has_tracker_target,class_id,confidence,is_enemy,x1,y1,x2,y2,"
            << "candidate_count,enemy_count\n";
        RCLCPP_INFO(this->get_logger(),
                    "Saving up to %d target frames as JPEG quality=%d in %s",
                    target_frame_save_max_, target_frame_jpeg_quality_,
                    target_frame_save_dir_.c_str());
      }
    } catch (const std::exception &exc) {
      RCLCPP_WARN(this->get_logger(),
                  "Failed to initialize target frame capture: %s", exc.what());
      save_target_frames_ = false;
    }
  }

  target_class_ = (target_color_ == "red") ? RED_CLASS : BLUE_CLASS;
  RCLCPP_INFO(this->get_logger(),
              "Default enemy target_color=%s class_id=%d; STM attitude_flags "
              "or robot_id can "
              "override when available",
              target_color_.c_str(), target_class_);

  robot_id_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
      "robot_id", 10,
      std::bind(&ArmorDetectorNode::id_listener_callback, this,
                std::placeholders::_1));
  attitude_flags_subscription_ =
      this->create_subscription<std_msgs::msg::Int32MultiArray>(
          "attitude_flags", 10,
          std::bind(&ArmorDetectorNode::handle_attitude_flags, this,
                    std::placeholders::_1));
  preview_target_subscription_ =
      this->create_subscription<cv_auto_aim_interfaces::msg::Target>(
          preview_target_topic_, 10,
          std::bind(&ArmorDetectorNode::handle_preview_target, this,
                    std::placeholders::_1));
  if (enable_3d_pose_) {
    camera_info_subscription_ =
        this->create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic_, rclcpp::SensorDataQoS(),
            std::bind(&ArmorDetectorNode::handle_camera_info, this,
                      std::placeholders::_1));
    configure_static_camera_model();
  }
  armors_publisher_ =
      this->create_publisher<cv_auto_aim_interfaces::msg::Armors>(
          "detector/armors", 10);
  if (debug_publish_image_raw_) {
    auto qos =
        rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
    debug_image_publisher_ =
        this->create_publisher<sensor_msgs::msg::Image>("image_raw", qos);
  }
  initialize_preview_server();

  if (detector_backend_kind_ == "native_tensorrt" &&
      camera_config_.camera_type == "basler" &&
      native_tensorrt_cuda_bayer_preprocess_) {
    camera_config_.basler_raw_bayer_capture = true;
    RCLCPP_INFO(this->get_logger(),
                "Native TensorRT CUDA Bayer preprocessing enabled; Basler "
                "frames will bypass Pylon BGR conversion");
  }

  if (detector_backend_kind_ == "deepstream") {
    try {
      initialize_deepstream_runtime();
    } catch (const std::exception &exc) {
      if (!allow_native_tensorrt_fallback_) {
        throw;
      }
      RCLCPP_WARN(this->get_logger(),
                  "DeepStream startup failed (%s). Falling back to native "
                  "TensorRT because "
                  "allow_native_tensorrt_fallback=true.",
                  exc.what());
      detector_backend_kind_ = "native_tensorrt";
      initialize_native_pipeline();
    } catch (...) {
      if (!allow_native_tensorrt_fallback_) {
        throw std::runtime_error("DeepStream startup failed with a non-std "
                                 "exception, likely from the "
                                 "Basler/Pylon GStreamer source");
      }
      RCLCPP_WARN(
          this->get_logger(),
          "DeepStream startup failed with a non-std exception, likely from the "
          "Basler/Pylon GStreamer source. Falling back to native TensorRT "
          "because "
          "allow_native_tensorrt_fallback=true.");
      detector_backend_kind_ = "native_tensorrt";
      initialize_native_pipeline();
    }
  } else {
    initialize_native_pipeline();
  }

  RCLCPP_INFO(this->get_logger(),
              "Initialized ArmorDetectorNode camera=%s backend=%s fps=%.2f",
              camera_config_.camera_type.c_str(),
              deepstream_runtime_ != nullptr
                  ? deepstream_runtime_->backend_name().c_str()
                  : detector_backend_->backend_name().c_str(),
              fps_);
  log_runtime_note("INFO", "Initializing Modular YOLOv8 Node...");
  log_runtime_note("INFO", "Detector Initialized.");
}

ArmorDetectorNode::~ArmorDetectorNode() {
  timer_.reset();
  if (preview_server_) {
    preview_server_->stop();
    preview_server_.reset();
  }
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
  if (target_frame_index_stream_.is_open()) {
    target_frame_index_stream_.close();
  }
}

void ArmorDetectorNode::id_listener_callback(
    const std_msgs::msg::Int32::SharedPtr msg) {
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

void ArmorDetectorNode::handle_attitude_flags(
    const std_msgs::msg::Int32MultiArray::SharedPtr msg) {
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
    const char *color_name = target_class_ == RED_CLASS ? "RED" : "BLUE";
    log_runtime_note(
        "INFO", "attitude_flags enemy_color=" + std::to_string(enemy_color) +
                    " -> Targeting " + color_name + ".");
  }
}

void ArmorDetectorNode::handle_preview_target(
    const cv_auto_aim_interfaces::msg::Target::SharedPtr msg) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  preview_has_target_ = msg->has_target;
  preview_target_track_id_ = msg->track_id;
  preview_target_bbox_xyxy_ = msg->bbox_xyxy;
  preview_target_image_width_ = msg->image_width;
  preview_target_image_height_ = msg->image_height;
}

void ArmorDetectorNode::handle_camera_info(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  if (!enable_3d_pose_) {
    return;
  }

  std::lock_guard<std::mutex> lock(state_mutex_);
  camera_calibration_.model.camera_matrix =
      cv::Matx33d(msg->k[0], msg->k[1], msg->k[2], msg->k[3], msg->k[4],
                  msg->k[5], msg->k[6], msg->k[7], msg->k[8]);
  camera_calibration_.model.distortion_coefficients = msg->d;
  camera_calibration_.available = true;
  camera_calibration_.from_camera_info = true;
  camera_calibration_.image_width = static_cast<int>(msg->width);
  camera_calibration_.image_height = static_cast<int>(msg->height);
  pnp_solver_.emplace(camera_calibration_.model, small_armor_width_m_,
                      small_armor_height_m_, large_armor_width_m_,
                      large_armor_height_m_);
}

void ArmorDetectorNode::configure_static_camera_model() {
  if (camera_fx_ <= 0.0 || camera_fy_ <= 0.0) {
    return;
  }

  camera_calibration_.model.camera_matrix = cv::Matx33d(
      camera_fx_, 0.0,
      camera_cx_ > 0.0 ? camera_cx_ : static_cast<double>(imgsz_width_) / 2.0,
      0.0, camera_fy_,
      camera_cy_ > 0.0 ? camera_cy_ : static_cast<double>(imgsz_height_) / 2.0,
      0.0, 0.0, 1.0);
  camera_calibration_.model.distortion_coefficients = camera_distortion_;
  camera_calibration_.available = true;
  camera_calibration_.from_camera_info = false;
  camera_calibration_.image_width = imgsz_width_;
  camera_calibration_.image_height = imgsz_height_;
  pnp_solver_.emplace(camera_calibration_.model, small_armor_width_m_,
                      small_armor_height_m_, large_armor_width_m_,
                      large_armor_height_m_);
}

PnPCameraModel
ArmorDetectorNode::make_fov_camera_model(int image_width,
                                         int image_height) const {
  const double fx =
      static_cast<double>(image_width) / (2.0 * std::tan(hfov_rad_ / 2.0));
  const double fy =
      static_cast<double>(image_height) / (2.0 * std::tan(vfov_rad_ / 2.0));
  PnPCameraModel model;
  model.camera_matrix =
      cv::Matx33d(fx, 0.0, static_cast<double>(image_width) / 2.0, 0.0, fy,
                  static_cast<double>(image_height) / 2.0, 0.0, 0.0, 1.0);
  model.distortion_coefficients = {};
  return model;
}

std::optional<geometry_msgs::msg::Pose>
ArmorDetectorNode::solve_candidate_pose(const Candidate &candidate,
                                        int image_width, int image_height) {
  if (!enable_3d_pose_ || !candidate.has_corners) {
    return std::nullopt;
  }

  const double aspect =
      static_cast<double>(std::max(1, candidate.bbox_source.width)) /
      static_cast<double>(std::max(1, candidate.bbox_source.height));
  const int armor_type = aspect >= large_armor_min_aspect_ratio_
                             ? cv_auto_aim_interfaces::msg::Armor::ARMOR_LARGE
                             : cv_auto_aim_interfaces::msg::Armor::ARMOR_SMALL;

  geometry_msgs::msg::Pose pose;
  if (pnp_solver_.has_value()) {
    if (pnp_solver_->solve(candidate.corners_source, armor_type, pose)) {
      return pose;
    }
    return std::nullopt;
  }

  const auto fov_model = make_fov_camera_model(image_width, image_height);
  ArmorPnPSolver solver(fov_model, small_armor_width_m_, small_armor_height_m_,
                        large_armor_width_m_, large_armor_height_m_);
  if (solver.solve(candidate.corners_source, armor_type, pose)) {
    return pose;
  }
  return std::nullopt;
}

void ArmorDetectorNode::initialize_native_pipeline() {
  detector_backend_ = create_detector_backend(
      detector_backend_kind_, model_path_, task_, model_type_, half_, conf_);
  detector_backend_->prepare(this->get_logger());

  camera_backend_ = create_camera_backend(camera_config_, this->get_logger());
  camera_backend_->initialize();
  fps_ = camera_backend_->actual_fps() > 0.0 ? camera_backend_->actual_fps()
                                             : 60.0;
  timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(1.0, fps_)),
      std::bind(&ArmorDetectorNode::timer_callback, this));
}

void ArmorDetectorNode::initialize_deepstream_runtime() {
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
  config.source_width = camera_config_.basler_use_full_sensor
                            ? camera_config_.basler_source_width
                            : imgsz_width_;
  config.source_height = camera_config_.basler_use_full_sensor
                             ? camera_config_.basler_source_height
                             : imgsz_height_;
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
  config.bayer_conversion_backend = bayer_conversion_backend_;
  config.flip_source = camera_config_.basler_flip;
  config.conf_threshold = conf_;
  config.half = half_;
  config.enable_color_verification =
      enable_color_verification_ && !use_model_color_classes_;
  config.debug_publish_image_raw =
      debug_publish_image_raw_ || save_target_frames_;
  config.preview_frame = show_;

  deepstream_runtime_ = std::make_unique<DeepStreamRuntime>(
      this->get_logger(), config, [this](DeepStreamRuntime::Result &&result) {
        this->handle_deepstream_result(std::move(result));
      });
  fps_ = deepstream_runtime_->source_actual_fps() > 0.0
             ? deepstream_runtime_->source_actual_fps()
             : static_cast<double>(std::max(1, camera_config_.basler_fps));
}

void ArmorDetectorNode::timer_callback() {
  const auto capture_start = std::chrono::steady_clock::now();
  FramePacket packet{};
  if (!camera_backend_->capture(packet)) {
    return;
  }
  const auto capture_done = std::chrono::steady_clock::now();

  DetectorStageTimings detector_timings{};
  const auto batch =
      detector_backend_->infer(packet, this->get_logger(), &detector_timings);
  process_detection_cycle(
      batch, &packet, detector_timings,
      std::chrono::duration<double, std::milli>(capture_done - capture_start)
          .count());
}

void ArmorDetectorNode::handle_deepstream_result(
    DeepStreamRuntime::Result result) {
  const FramePacket *packet =
      result.source_packet.has_value() ? &(*result.source_packet) : nullptr;
  process_detection_cycle(result.batch, packet, result.timings,
                          result.capture_ms);
}

void ArmorDetectorNode::process_detection_cycle(
    const DetectionBatch &batch, const FramePacket *packet,
    const DetectorStageTimings &detector_timings, double capture_ms) {
  // This is the detector's central data boundary: model outputs become
  // candidates, optional debug images, runtime telemetry, and ROS interface
  // messages.
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
  preview_fps_ = preview_fps_ <= 0.0
                     ? instant_fps
                     : (0.9 * preview_fps_) + (0.1 * instant_fps);

  double color_ms = 0.0;
  const auto publish_start = std::chrono::steady_clock::now();
  auto candidates = extract_candidates(batch, packet, &color_ms);
  const auto detected_opponent = std::find_if(
      candidates.begin(), candidates.end(),
      [](const Candidate &candidate) { return candidate.is_enemy; });
  const auto enemy_count = static_cast<std::uint64_t>(std::count_if(
      candidates.begin(), candidates.end(),
      [](const Candidate &candidate) { return candidate.is_enemy; }));
  const auto class0_count = static_cast<std::uint64_t>(std::count_if(
      candidates.begin(), candidates.end(),
      [](const Candidate &candidate) { return candidate.class_id == 0; }));
  const auto class1_count = static_cast<std::uint64_t>(std::count_if(
      candidates.begin(), candidates.end(),
      [](const Candidate &candidate) { return candidate.class_id == 1; }));
  const auto other_class_count = static_cast<std::uint64_t>(candidates.size()) -
                                 class0_count - class1_count;
  if (packet != nullptr && !packet->empty()) {
    save_target_frame_if_needed(*packet, candidates, stamp);
  }
  const std::string source_id = packet != nullptr && !packet->source_id.empty()
                                    ? packet->source_id
                                    : camera_config_.camera_type;
  const std::string frame_id =
      camera_frame_id_.empty() ? source_id : camera_frame_id_;
  const int image_width =
      batch.letterbox.src_w > 0
          ? batch.letterbox.src_w
          : (packet != nullptr ? packet->width() : imgsz_width_);
  const int image_height =
      batch.letterbox.src_h > 0
          ? batch.letterbox.src_h
          : (packet != nullptr ? packet->height() : imgsz_height_);
  publish_armors(candidates, frame_id, stamp, image_width, image_height);
  const auto publish_done = std::chrono::steady_clock::now();
  const double publish_ms =
      std::chrono::duration<double, std::milli>(publish_done - publish_start)
          .count();
  const double latency_ms =
      capture_ms + detector_timings.preprocess_ms + detector_timings.upload_ms +
      detector_timings.inference_ms + detector_timings.download_ms +
      detector_timings.parse_ms + color_ms + publish_ms;
  const auto latest_preview_stats = build_preview_stats_json(
      source_id, image_width, image_height, candidates, latency_ms);

  if (show_ && packet != nullptr && !packet->empty()) {
    const auto now = std::chrono::steady_clock::now();
    const double min_interval_ms = 1000.0 / std::max(1.0, preview_max_fps_);
    const bool should_render_preview =
        !have_last_preview_render_time_ ||
        std::chrono::duration<double, std::milli>(now -
                                                  last_preview_render_time_)
                .count() >= min_interval_ms;
    if (should_render_preview) {
      render_preview(*packet, candidates, batch.letterbox,
                     latest_preview_stats);
      last_preview_render_time_ = now;
      have_last_preview_render_time_ = true;
    }
  }

  log_runtime_fps_once_per_second();
  if (detected_opponent != candidates.end()) {
    log_runtime_opponent_event(*detected_opponent, true, latency_ms);
  }

  ++frame_count_;
  stage_raw_detections_ += batch.detections.size();
  stage_candidates_ += candidates.size();
  stage_enemy_candidates_ += enemy_count;
  stage_class0_candidates_ += class0_count;
  stage_class1_candidates_ += class1_count;
  stage_other_class_candidates_ += other_class_count;
  stage_totals_.capture_ms += capture_ms;
  stage_totals_.preprocess_ms += detector_timings.preprocess_ms;
  stage_totals_.upload_ms += detector_timings.upload_ms;
  stage_totals_.inference_ms += detector_timings.inference_ms;
  stage_totals_.download_ms += detector_timings.download_ms;
  stage_totals_.parse_ms += detector_timings.parse_ms;
  stage_totals_.color_ms += color_ms;
  stage_totals_.publish_ms += publish_ms;

  if (frame_count_ % stage_log_interval_ == 0) {
    log_stage_timing();
  }
}

void ArmorDetectorNode::publish_armors(const std::vector<Candidate> &candidates,
                                       const std::string &source_id,
                                       const rclcpp::Time &stamp,
                                       int image_width, int image_height) {
  cv_auto_aim_interfaces::msg::Armors msg;
  msg.header.stamp = stamp;
  msg.header.frame_id = source_id;
  msg.source_id = source_id;
  msg.armors.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    cv_auto_aim_interfaces::msg::Armor armor;
    armor.bbox_xyxy = {candidate.bbox_source.x, candidate.bbox_source.y,
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
    const double aspect =
        static_cast<double>(std::max(1, candidate.bbox_source.width)) /
        static_cast<double>(std::max(1, candidate.bbox_source.height));
    armor.armor_type = aspect >= large_armor_min_aspect_ratio_
                           ? cv_auto_aim_interfaces::msg::Armor::ARMOR_LARGE
                           : cv_auto_aim_interfaces::msg::Armor::ARMOR_SMALL;
    armor.armor_number = -1;
    const double principal_x =
        camera_calibration_.available
            ? camera_calibration_.model.camera_matrix(0, 2)
            : static_cast<double>(image_width) / 2.0;
    const double principal_y =
        camera_calibration_.available
            ? camera_calibration_.model.camera_matrix(1, 2)
            : static_cast<double>(image_height) / 2.0;
    armor.distance_to_image_center = static_cast<float>(std::hypot(
        static_cast<double>(candidate.center_source.x) - principal_x,
        static_cast<double>(candidate.center_source.y) - principal_y));
    if (const auto pose =
            solve_candidate_pose(candidate, image_width, image_height)) {
      armor.has_pose = true;
      armor.pose_camera = *pose;
    }
    msg.armors.push_back(std::move(armor));
  }
  armors_publisher_->publish(std::move(msg));
}

void ArmorDetectorNode::publish_debug_image(const FramePacket &packet,
                                            const rclcpp::Time &stamp) {
  if (!debug_image_publisher_) {
    return;
  }
  auto msg = vision_node_helpers::frame_to_image_message(packet, stamp);
  if (msg) {
    debug_image_publisher_->publish(std::move(msg));
  }
}

void ArmorDetectorNode::render_preview(const FramePacket &packet,
                                       const std::vector<Candidate> &candidates,
                                       const LetterboxMetadata &source_geometry,
                                       const std::string &stats_json) {
  if (preview_failed_ && preview_server_ == nullptr) {
    return;
  }

  try {
    cv::Mat preview = make_preview_frame(packet);
    const int source_width =
        source_geometry.src_w > 0 ? source_geometry.src_w : preview.cols;
    const int source_height =
        source_geometry.src_h > 0 ? source_geometry.src_h : preview.rows;
    const double scale_x =
        static_cast<double>(preview.cols) / static_cast<double>(source_width);
    const double scale_y =
        static_cast<double>(preview.rows) / static_cast<double>(source_height);
    const cv::Rect image_rect(0, 0, preview.cols, preview.rows);
    for (const auto &candidate : candidates) {
      const cv::Scalar color =
          candidate.is_enemy ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
      const cv::Rect preview_rect = vision_node_helpers::scale_rect_to_preview(
          candidate.bbox_source, scale_x, scale_y, image_rect);
      if (preview_rect.empty()) {
        continue;
      }
      cv::rectangle(preview, preview_rect, color, 2);
      const std::string label = std::to_string(candidate.class_id) + " " +
                                cv::format("%.2f", candidate.confidence);
      cv::putText(preview, label,
                  cv::Point(preview_rect.x, std::max(16, preview_rect.y - 4)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
    }

    if (preview_has_target_) {
      const int x1 = preview_target_bbox_xyxy_[0];
      const int y1 = preview_target_bbox_xyxy_[1];
      const int x2 = preview_target_bbox_xyxy_[2];
      const int y2 = preview_target_bbox_xyxy_[3];
      const int target_source_width =
          preview_target_image_width_ > 0 ? preview_target_image_width_ : source_width;
      const int target_source_height =
          preview_target_image_height_ > 0 ? preview_target_image_height_ : source_height;
      const double target_scale_x =
          static_cast<double>(preview.cols) / static_cast<double>(target_source_width);
      const double target_scale_y =
          static_cast<double>(preview.rows) / static_cast<double>(target_source_height);
      const cv::Rect target_rect = vision_node_helpers::scale_rect_to_preview(
          cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1)), target_scale_x,
          target_scale_y, image_rect);
      if (target_rect.width > 0 && target_rect.height > 0) {
        const cv::Scalar target_color(0, 0, 255);
        cv::rectangle(preview, target_rect, target_color, 4);
        cv::circle(preview,
                   cv::Point(target_rect.x + target_rect.width / 2,
                             target_rect.y + target_rect.height / 2),
                   5, target_color, -1, cv::LINE_AA);
        const std::string label =
            preview_target_track_id_ >= 0
                ? "TARGET #" + std::to_string(preview_target_track_id_)
                : "TARGET";
        const cv::Point label_origin(target_rect.x,
                                     std::max(22, target_rect.y - 8));
        cv::putText(preview, label, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    target_color, 2, cv::LINE_AA);
      }
    }

    cv::putText(preview, "FPS " + cv::format("%.1f", preview_fps_),
                cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    const double scale =
        std::min(1.0, std::min(static_cast<double>(kPreviewMaxWidth) /
                                   std::max(1, preview.cols),
                               static_cast<double>(kPreviewMaxHeight) /
                                   std::max(1, preview.rows)));
    if (scale < 1.0) {
      cv::resize(preview, preview, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    if (preview_server_) {
      preview_server_->update_frame(preview, stats_json,
                                    preview_http_jpeg_quality_);
    }

    if (!preview_failed_) {
      try {
        cv::imshow(vis_window_name_, preview);
        cv::waitKey(1);
      } catch (const cv::Exception &exc) {
        preview_failed_ = true;
        RCLCPP_WARN(
            this->get_logger(),
            "Disabling local OpenCV preview window after display failure: %s",
            exc.what());
      }
    }
  } catch (const cv::Exception &exc) {
    preview_failed_ = true;
    RCLCPP_WARN(this->get_logger(),
                "Disabling detector preview after render failure: %s",
                exc.what());
  }
}

void ArmorDetectorNode::initialize_preview_server() {
  if (!show_ || !preview_http_enabled_) {
    return;
  }

  try {
    MjpegPreviewServer::Config config;
    config.bind_address = preview_http_bind_address_;
    config.port = static_cast<std::uint16_t>(preview_http_port_);
    config.title = "CV Detection Preview";
    preview_server_ = std::make_unique<MjpegPreviewServer>();
    preview_server_->start(config);
    RCLCPP_INFO(
        this->get_logger(),
        "Browser preview enabled: http://%s:%d/ (open from another device as "
        "http://<robot-host>:%d/)",
        preview_http_bind_address_.c_str(), preview_http_port_,
        preview_http_port_);
  } catch (const std::exception &exc) {
    preview_server_.reset();
    RCLCPP_WARN(this->get_logger(), "Browser preview disabled: %s", exc.what());
  }
}

std::string ArmorDetectorNode::build_preview_stats_json(
    const std::string &source_id, int image_width, int image_height,
    const std::vector<Candidate> &candidates, double latency_ms) const {
  const auto enemy_count = std::count_if(
      candidates.begin(), candidates.end(),
      [](const Candidate &candidate) { return candidate.is_enemy; });
  const auto class0_count = std::count_if(
      candidates.begin(), candidates.end(),
      [](const Candidate &candidate) { return candidate.class_id == 0; });
  const auto class1_count = std::count_if(
      candidates.begin(), candidates.end(),
      [](const Candidate &candidate) { return candidate.class_id == 1; });
  const std::string backend =
      deepstream_runtime_ != nullptr
          ? deepstream_runtime_->backend_name()
          : (detector_backend_ != nullptr ? detector_backend_->backend_name()
                                          : detector_backend_kind_);

  std::ostringstream json;
  json << std::fixed << std::setprecision(3);
  json << "{"
       << "\"source_id\":\"" << json_escape(source_id) << "\","
       << "\"camera\":\"" << json_escape(camera_config_.camera_type) << "\","
       << "\"backend\":\"" << json_escape(backend) << "\","
       << "\"width\":" << image_width << ',' << "\"height\":" << image_height
       << ',' << "\"fps\":" << preview_fps_ << ','
       << "\"frame_count\":" << frame_count_ << ','
       << "\"candidates\":" << candidates.size() << ','
       << "\"enemy_candidates\":" << enemy_count << ','
       << "\"class0_candidates\":" << class0_count << ','
       << "\"class1_candidates\":" << class1_count << ','
       << "\"target_color\":\"" << json_escape(target_color_) << "\","
       << "\"target_class\":" << target_class_ << ','
       << "\"has_target\":" << (preview_has_target_ ? "true" : "false") << ','
       << "\"target_track_id\":" << preview_target_track_id_ << ','
       << "\"latency_ms\":" << latency_ms << ',' << "\"show\":true"
       << "}";
  return json.str();
}

void ArmorDetectorNode::save_target_frame_if_needed(
    const FramePacket &packet, const std::vector<Candidate> &candidates,
    const rclcpp::Time &stamp) {
  if (!save_target_frames_ || target_frame_save_max_ <= 0 ||
      target_frame_save_count_ >= target_frame_save_max_ ||
      !target_frame_index_stream_.is_open()) {
    return;
  }

  const cv::Rect image_rect(0, 0, packet.width(), packet.height());
  bool has_tracker_target = false;
  int track_id = -1;
  cv::Rect target_rect;
  const Candidate *target_candidate = nullptr;

  if (preview_has_target_) {
    const int x1 = preview_target_bbox_xyxy_[0];
    const int y1 = preview_target_bbox_xyxy_[1];
    const int x2 = preview_target_bbox_xyxy_[2];
    const int y2 = preview_target_bbox_xyxy_[3];
    const int target_source_width =
        preview_target_image_width_ > 0 ? preview_target_image_width_ : packet.width();
    const int target_source_height =
        preview_target_image_height_ > 0 ? preview_target_image_height_ : packet.height();
    const double target_scale_x =
        static_cast<double>(packet.width()) / static_cast<double>(target_source_width);
    const double target_scale_y =
        static_cast<double>(packet.height()) / static_cast<double>(target_source_height);
    target_rect = vision_node_helpers::scale_rect_to_preview(
        cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1)), target_scale_x,
        target_scale_y, image_rect);
    if (!target_rect.empty()) {
      has_tracker_target = true;
      track_id = preview_target_track_id_;
      double best_iou = 0.0;
      for (const auto &candidate : candidates) {
        const auto clipped = candidate.bbox_source & image_rect;
        const double iou = vision_node_helpers::rect_iou(target_rect, clipped);
        if (iou > best_iou) {
          best_iou = iou;
          target_candidate = &candidate;
        }
      }
      if (target_candidate != nullptr) {
        target_rect = target_candidate->bbox_source & image_rect;
      }
    }
  }

  if (target_candidate == nullptr) {
    auto best_enemy = candidates.end();
    for (auto it = candidates.begin(); it != candidates.end(); ++it) {
      if (!it->is_enemy) {
        continue;
      }
      if (best_enemy == candidates.end() ||
          it->confidence > best_enemy->confidence) {
        best_enemy = it;
      }
    }
    if (best_enemy != candidates.end()) {
      target_candidate = &(*best_enemy);
      target_rect = target_candidate->bbox_source & image_rect;
    }
  }

  if (target_rect.empty()) {
    return;
  }

  const int image_index = target_frame_save_count_ + 1;
  std::ostringstream filename;
  filename << "target_" << std::setw(6) << std::setfill('0') << image_index
           << '_' << stamp.nanoseconds() << ".jpg";
  const auto image_path =
      std::filesystem::path(target_frame_save_dir_) / filename.str();

  try {
    const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY,
                                  target_frame_jpeg_quality_};
    if (!cv::imwrite(image_path.string(), packet.frame, params)) {
      RCLCPP_WARN(
          this->get_logger(),
          "Failed to write target frame %s; disabling target frame capture",
          image_path.string().c_str());
      save_target_frames_ = false;
      return;
    }
  } catch (const std::exception &exc) {
    RCLCPP_WARN(
        this->get_logger(),
        "Failed to write target frame %s: %s; disabling target frame capture",
        image_path.string().c_str(), exc.what());
    save_target_frames_ = false;
    return;
  }

  ++target_frame_save_count_;
  const auto enemy_count = std::count_if(
      candidates.begin(), candidates.end(),
      [](const Candidate &candidate) { return candidate.is_enemy; });
  target_frame_index_stream_
      << filename.str() << ',' << stamp.nanoseconds() << ',' << packet.source_id
      << ',' << packet.width() << ',' << packet.height() << ',' << track_id
      << ',' << (has_tracker_target ? 1 : 0) << ','
      << (target_candidate != nullptr ? target_candidate->class_id : -1) << ','
      << (target_candidate != nullptr ? target_candidate->confidence : 0.0F)
      << ','
      << (target_candidate != nullptr && target_candidate->is_enemy ? 1 : 0)
      << ',' << target_rect.x << ',' << target_rect.y << ','
      << target_rect.x + target_rect.width << ','
      << target_rect.y + target_rect.height << ',' << candidates.size() << ','
      << enemy_count << '\n';
  target_frame_index_stream_.flush();

  if (target_frame_save_count_ == target_frame_save_max_) {
    RCLCPP_INFO(this->get_logger(),
                "Target frame capture reached max=%d; no more images will be "
                "saved this run",
                target_frame_save_max_);
  }
}

std::vector<Candidate>
ArmorDetectorNode::extract_candidates(const DetectionBatch &batch,
                                      const FramePacket *packet,
                                      double *color_ms) const {
  // Rebuild source coordinates before color checks so downstream nodes never
  // need to know the model letterbox transform.
  std::vector<Candidate> candidates;
  candidates.reserve(batch.detections.size());
  for (const auto &detection : batch.detections) {
    const auto source_rect =
        unletterbox_rect(detection.bbox_model, batch.letterbox);
    const cv::Point2f center_source(
        static_cast<float>(source_rect.x) +
            static_cast<float>(source_rect.width) / 2.0F,
        static_cast<float>(source_rect.y) +
            static_cast<float>(source_rect.height) / 2.0F);

    candidates.push_back(Candidate{detection.bbox_model,
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
      auto &candidate = candidates.back();
      for (std::size_t i = 0; i < detection.corners_model.size(); ++i) {
        candidate.corners_source[i] =
            unletterbox_point(detection.corners_model[i], batch.letterbox);
      }
    }
  }

  if (use_model_color_classes_) {
    for (auto &candidate : candidates) {
      candidate.is_enemy =
          class_id_matches_target_color(candidate.class_id, target_class_);
    }
    return candidates;
  }

  if (!enable_color_verification_ || candidates.empty() || packet == nullptr ||
      packet->empty()) {
    return candidates;
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &lhs, const Candidate &rhs) {
              return lhs.confidence > rhs.confidence;
            });
  const int candidate_count = static_cast<int>(candidates.size());
  const auto check_count = static_cast<std::size_t>(
      std::min<int>(max_hsv_candidates_, candidate_count));

  const auto color_start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    auto &candidate = candidates[i];
    if (i >= check_count) {
      candidate.is_enemy = false;
      continue;
    }

    const auto clipped = candidate.bbox_source &
                         cv::Rect(0, 0, packet->width(), packet->height());
    if (clipped.width <= 0 || clipped.height <= 0) {
      candidate.is_enemy = false;
      continue;
    }
    const auto roi = packet->frame(clipped);
    candidate.is_enemy = check_enemy(roi, target_class_).first;
  }
  const auto color_done = std::chrono::steady_clock::now();
  if (color_ms != nullptr) {
    *color_ms =
        std::chrono::duration<double, std::milli>(color_done - color_start)
            .count();
  }

  return candidates;
}

std::string ArmorDetectorNode::resolve_default_model_path() const {
  return vision_node_helpers::resolve_model_path("");
}

std::string ArmorDetectorNode::resolve_default_deepstream_config_path() const {
  const auto cwd_candidate = std::filesystem::current_path() / "yolo_cpp" /
                             "config" / "deepstream_bestmerge_nvinfer.txt";
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

std::string ArmorDetectorNode::resolve_runtime_log_dir() const {
  if (!runtime_log_dir_.empty()) {
    std::filesystem::create_directories(runtime_log_dir_);
    return runtime_log_dir_;
  }
  if (const char *env_dir = std::getenv("VISION_LOG_DIR")) {
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

std::string ArmorDetectorNode::resolve_target_frame_save_dir() const {
  if (!target_frame_save_dir_.empty()) {
    std::filesystem::create_directories(target_frame_save_dir_);
    return target_frame_save_dir_;
  }

  const auto log_dir = std::filesystem::path(resolve_runtime_log_dir());
  auto parent = log_dir.parent_path();
  if (parent.empty()) {
    parent = std::filesystem::current_path();
  }
  const auto picture_dir = parent / kTargetFrameDirName;
  std::filesystem::create_directories(picture_dir);
  return picture_dir.string();
}

void ArmorDetectorNode::write_runtime_log_line(const std::string &line) {
  if (!runtime_log_stream_.is_open()) {
    return;
  }
  runtime_log_stream_ << "ts="
                      << vision_node_helpers::timestamp_now_iso8601_ms() << ' '
                      << line << '\n';
  runtime_log_stream_.flush();
}

void ArmorDetectorNode::log_runtime_note(const std::string &level,
                                         const std::string &message) {
  write_runtime_log_line("level=" + level + " msg=" + message);
}

void ArmorDetectorNode::log_runtime_fps_once_per_second() {
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
  line << std::fixed << std::setprecision(2) << "fps_scope="
       << (deepstream_runtime_ != nullptr ? "deepstream_result" : "frame")
       << ' ' << "fps=" << preview_fps_;
  write_runtime_log_line(line.str());
}

void ArmorDetectorNode::log_runtime_opponent_event(const Candidate &target,
                                                   bool color_pass,
                                                   double latency_ms) {
  const int x1 = target.bbox_source.x;
  const int y1 = target.bbox_source.y;
  const int x2 = target.bbox_source.x + target.bbox_source.width;
  const int y2 = target.bbox_source.y + target.bbox_source.height;

  std::ostringstream line;
  line << std::fixed << std::setprecision(2) << "fps_scope="
       << (deepstream_runtime_ != nullptr ? "deepstream_result" : "frame")
       << ' ' << "fps=" << preview_fps_ << ' ' << "bbox=" << x1 << ',' << y1
       << ',' << x2 << ',' << y2 << ' ';
  line << std::setprecision(3) << "conf=" << target.confidence << ' '
       << "color_pass=" << (color_pass ? 1 : 0) << ' ';
  line << "latency_scope="
       << (deepstream_runtime_ != nullptr ? "post_infer" : "capture_to_publish")
       << ' ';
  line << std::setprecision(2) << "latency_ms=" << latency_ms;
  write_runtime_log_line(line.str());
}

void ArmorDetectorNode::log_stage_timing() {
  const double n = static_cast<double>(stage_log_interval_);
  double avg_fps = preview_fps_;
  const auto now = std::chrono::steady_clock::now();
  if (have_last_stage_log_wall_time_) {
    const double elapsed_s =
        std::chrono::duration<double>(now - last_stage_log_wall_time_).count();
    if (elapsed_s > 0.0) {
      avg_fps = n / elapsed_s;
    }
  }
  last_stage_log_wall_time_ = now;
  have_last_stage_log_wall_time_ = true;
  if (deepstream_runtime_ != nullptr) {
    RCLCPP_INFO(this->get_logger(),
                "Detector DeepStream result timing mean over %d results: "
                "result_fps=%.1f parse=%.2fms "
                "color=%.2fms publish=%.2fms raw_det=%.2f candidates=%.2f "
                "enemy=%.2f class0=%.2f "
                "class1=%.2f class_other=%.2f "
                "note=preprocess/upload/inference/download_unavailable",
                stage_log_interval_, avg_fps, stage_totals_.parse_ms / n,
                stage_totals_.color_ms / n, stage_totals_.publish_ms / n,
                static_cast<double>(stage_raw_detections_) / n,
                static_cast<double>(stage_candidates_) / n,
                static_cast<double>(stage_enemy_candidates_) / n,
                static_cast<double>(stage_class0_candidates_) / n,
                static_cast<double>(stage_class1_candidates_) / n,
                static_cast<double>(stage_other_class_candidates_) / n);
    stage_totals_ = StageTotals{};
    stage_raw_detections_ = 0;
    stage_candidates_ = 0;
    stage_enemy_candidates_ = 0;
    stage_class0_candidates_ = 0;
    stage_class1_candidates_ = 0;
    stage_other_class_candidates_ = 0;
    return;
  }

  RCLCPP_INFO(this->get_logger(),
              "Detector timing mean over %d frames: fps=%.1f capture=%.2fms "
              "preprocess=%.2fms "
              "upload=%.2fms inference=%.2fms download=%.2fms parse=%.2fms "
              "color=%.2fms publish=%.2fms "
              "raw_det=%.2f candidates=%.2f enemy=%.2f class0=%.2f class1=%.2f "
              "class_other=%.2f",
              stage_log_interval_, avg_fps, stage_totals_.capture_ms / n,
              stage_totals_.preprocess_ms / n, stage_totals_.upload_ms / n,
              stage_totals_.inference_ms / n, stage_totals_.download_ms / n,
              stage_totals_.parse_ms / n, stage_totals_.color_ms / n,
              stage_totals_.publish_ms / n,
              static_cast<double>(stage_raw_detections_) / n,
              static_cast<double>(stage_candidates_) / n,
              static_cast<double>(stage_enemy_candidates_) / n,
              static_cast<double>(stage_class0_candidates_) / n,
              static_cast<double>(stage_class1_candidates_) / n,
              static_cast<double>(stage_other_class_candidates_) / n);
  stage_totals_ = StageTotals{};
  stage_raw_detections_ = 0;
  stage_candidates_ = 0;
  stage_enemy_candidates_ = 0;
  stage_class0_candidates_ = 0;
  stage_class1_candidates_ = 0;
  stage_other_class_candidates_ = 0;
}

} // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::ArmorDetectorNode)
