#include "yolo_cpp/yolo_node.hpp"

#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <optional>
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

}  // namespace

YoloNode::YoloNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("yolo_modular_node", options),
  detector_("", "", "", true, 0.25F)
{
  const std::string default_model = resolve_default_model_path();

  this->declare_parameter<std::string>("model_path", default_model);
  this->declare_parameter<std::string>("task", "obb");
  this->declare_parameter<std::string>("type", "yolo");
  this->declare_parameter<bool>("half", true);
  this->declare_parameter<int>("imgsz_width", 640);
  this->declare_parameter<int>("imgsz_height", 640);
  this->declare_parameter<bool>("show", false);
  this->declare_parameter<double>("conf", 0.45);
  this->declare_parameter<int>("tracker_timeout", 15);
  this->declare_parameter<double>("prediction_lookahead_steps", 0.5);
  this->declare_parameter<std::string>("inference_mode", "predict");
  this->declare_parameter<bool>("persist_tracking", true);
  this->declare_parameter<bool>("use_kalman", false);
  this->declare_parameter<bool>("enable_color_verification", true);
  this->declare_parameter<bool>("use_model_color_classes", true);
  this->declare_parameter<std::string>("target_color", "blue");
  this->declare_parameter<bool>("use_robot_id_targeting", false);
  this->declare_parameter<int>("max_hsv_candidates", kDefaultMaxHsvCandidates);
  this->declare_parameter<int>("max_coast_frames", kDefaultMaxCoastFrames);
  this->declare_parameter<double>("hfov_degrees", 79.2);
  this->declare_parameter<double>("vfov_degrees", 63.78);
  this->declare_parameter<std::string>("vis_window_name", "YOLO Modular Node");

  model_path_ = this->get_parameter("model_path").as_string();
  task_ = this->get_parameter("task").as_string();
  model_type_ = this->get_parameter("type").as_string();
  half_ = this->get_parameter("half").as_bool();
  imgsz_width_ = this->get_parameter("imgsz_width").as_int();
  imgsz_height_ = this->get_parameter("imgsz_height").as_int();
  show_ = this->get_parameter("show").as_bool();
  conf_ = static_cast<float>(this->get_parameter("conf").as_double());
  tracker_timeout_ = this->get_parameter("tracker_timeout").as_int();
  lookahead_steps_ = static_cast<float>(this->get_parameter("prediction_lookahead_steps").as_double());
  inference_mode_ = this->get_parameter("inference_mode").as_string();
  persist_tracking_ = this->get_parameter("persist_tracking").as_bool();
  use_kalman_ = this->get_parameter("use_kalman").as_bool();
  enable_color_verification_ = this->get_parameter("enable_color_verification").as_bool();
  use_model_color_classes_ = this->get_parameter("use_model_color_classes").as_bool();
  target_color_ = this->get_parameter("target_color").as_string();
  use_robot_id_targeting_ = this->get_parameter("use_robot_id_targeting").as_bool();
  max_hsv_candidates_ = this->get_parameter("max_hsv_candidates").as_int();
  max_coast_frames_ = this->get_parameter("max_coast_frames").as_int();
  hfov_rad_ = static_cast<float>(
    this->get_parameter("hfov_degrees").as_double() * kPi / 180.0);
  vfov_rad_ = static_cast<float>(
    this->get_parameter("vfov_degrees").as_double() * kPi / 180.0);
  vis_window_name_ = this->get_parameter("vis_window_name").as_string();

  target_class_ = (target_color_ == "red") ? RED_CLASS : BLUE_CLASS;
  detector_ = DeepStreamDetector(model_path_, task_, model_type_, half_, conf_);

  auto image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
  image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
    "image_raw", image_qos, std::bind(&YoloNode::image_callback, this, std::placeholders::_1));
  robot_id_subscription_ = this->create_subscription<std_msgs::msg::Int32>(
    "robot_id", 10, std::bind(&YoloNode::id_listener_callback, this, std::placeholders::_1));
  bbox_publisher_ =
    this->create_publisher<std_msgs::msg::Int32MultiArray>("bounding_box", 10);
  override_publisher_ =
    this->create_publisher<std_msgs::msg::Bool>("override", 10);
  angle_publisher_ =
    this->create_publisher<std_msgs::msg::Float32MultiArray>("bounding_box_angles", 10);
  obb_angle_publisher_ =
    this->create_publisher<std_msgs::msg::Float32>("obb_angle", 10);

  RCLCPP_INFO(
    this->get_logger(),
    "Initialized C++ YOLO node with backend=%s, inference_mode=%s, use_kalman=%s",
    detector_.backend_name().c_str(),
    inference_mode_.c_str(),
    use_kalman_ ? "true" : "false");
}

void YoloNode::id_listener_callback(const std_msgs::msg::Int32::SharedPtr msg)
{
  if (!use_robot_id_targeting_) {
    return;
  }
  if (is_red_team(msg->data)) {
    target_class_ = BLUE_CLASS;
  } else if (is_blue_team(msg->data)) {
    target_class_ = RED_CLASS;
  }
}

void YoloNode::image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  try {
    const cv::Mat frame = decode_frame(*msg);
    if (frame.empty()) {
      return;
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

    auto lb = letterbox_frame(frame, imgsz_width_, imgsz_height_);
    auto detections = detector_.infer(lb.frame, this->get_logger());
    auto candidates = rank_candidates(extract_candidates(detections, lb));

    bool published = false;
    if (use_kalman_) {
      published = run_kalman_pipeline(candidates, dt, lb);
    } else {
      auto target = select_target(candidates);
      if (target) {
        publish_target(*target, lb);
        published = true;
      }
    }

    if (!published) {
      publish_no_target(lb.src_w > 0 ? lb.src_w : frame.cols, lb.src_h > 0 ? lb.src_h : frame.rows);
    }
  } catch (const std::exception & exc) {
    RCLCPP_ERROR(this->get_logger(), "Error in image_callback: %s", exc.what());
  }
}

cv::Mat YoloNode::decode_frame(const sensor_msgs::msg::Image & msg)
{
  if (msg.encoding != sensor_msgs::image_encodings::BGR8) {
    RCLCPP_WARN(
      this->get_logger(), "Unsupported image encoding '%s'; expected bgr8",
      msg.encoding.c_str());
    return {};
  }
  if (msg.data.empty()) {
    return {};
  }
  return cv::Mat(
    static_cast<int>(msg.height),
    static_cast<int>(msg.width),
    CV_8UC3,
    const_cast<unsigned char *>(msg.data.data()),
    static_cast<std::size_t>(msg.step));
}

std::vector<Candidate> YoloNode::extract_candidates(
  const std::vector<Detection> & detections, const LetterboxResult & lb) const
{
  std::vector<Candidate> candidates;
  candidates.reserve(detections.size());

  for (std::size_t i = 0; i < detections.size(); ++i) {
    const auto & detection = detections[i];
    auto track_id = detection.track_id;
    if (use_kalman_ && !track_id.has_value()) {
      if (!candidates.empty()) {
        continue;
      }
      track_id = 0;
    }
    const auto source_rect = unletterbox_rect(detection.bbox_model, lb);
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
  } else if (enable_color_verification_ && !candidates.empty()) {
    std::sort(
      candidates.begin(), candidates.end(),
      [](const Candidate & lhs, const Candidate & rhs) {return lhs.confidence > rhs.confidence;});
    const auto check_count =
      static_cast<std::size_t>(std::min<int>(max_hsv_candidates_, static_cast<int>(candidates.size())));
    for (std::size_t i = 0; i < candidates.size(); ++i) {
      auto & candidate = candidates[i];
      if (i >= check_count) {
        candidate.is_enemy = false;
        continue;
      }

      const auto clipped = candidate.bbox_model & cv::Rect(0, 0, lb.frame.cols, lb.frame.rows);
      if (clipped.width <= 0 || clipped.height <= 0) {
        candidate.is_enemy = false;
        continue;
      }
      const auto roi = lb.frame(clipped);
      candidate.is_enemy = check_enemy(roi, target_class_).first;
    }
  }

  return candidates;
}

std::vector<Candidate> YoloNode::rank_candidates(std::vector<Candidate> candidates) const
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

std::optional<Candidate> YoloNode::select_target(const std::vector<Candidate> & candidates) const
{
  for (const auto & candidate : candidates) {
    if (candidate.is_enemy) {
      return candidate;
    }
  }
  return std::nullopt;
}

void YoloNode::publish_target(const Candidate & target, const LetterboxResult & lb)
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

void YoloNode::publish_no_target(int src_w, int src_h)
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

bool YoloNode::publish_kalman_target(
  double xf, double yf, double width, double height, const LetterboxResult & lb,
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

bool YoloNode::run_kalman_pipeline(
  const std::vector<Candidate> & candidates, double dt, const LetterboxResult & lb)
{
  std::vector<int> current_ids;

  for (const auto & candidate : candidates) {
    if (!candidate.is_enemy || !candidate.track_id.has_value()) {
      continue;
    }
    const int track_id = *candidate.track_id;
    current_ids.push_back(track_id);
    const double cx = static_cast<double>(candidate.bbox_model.x) +
      static_cast<double>(candidate.bbox_model.width) / 2.0;
    const double cy = static_cast<double>(candidate.bbox_model.y) +
      static_cast<double>(candidate.bbox_model.height) / 2.0;
    const double width = candidate.bbox_model.width;
    const double height = candidate.bbox_model.height;

    auto it = active_trackers_.find(track_id);
    if (it == active_trackers_.end()) {
      auto inserted = active_trackers_.emplace(
        track_id, TrackerWrapper(ArmorPlateTracker(cx, cy, width, height)));
      it = inserted.first;
    } else {
      it->second.tracker.predict(dt);
      it->second.tracker.update(cx, cy, width, height);
    }
    it->second.lost_frames = 0;
    it->second.obb_angle = candidate.obb_angle;
  }

  bool published = false;
  std::vector<int> to_remove;
  for (auto & [track_id, wrapper] : active_trackers_) {
    if (std::find(current_ids.begin(), current_ids.end(), track_id) == current_ids.end()) {
      ++wrapper.lost_frames;
    }
    if (wrapper.lost_frames >= tracker_timeout_) {
      to_remove.push_back(track_id);
      continue;
    }
    if (!published && wrapper.lost_frames <= max_coast_frames_) {
      const auto & state = wrapper.tracker.state();
      const double t_look = static_cast<double>(lookahead_steps_) * dt;
      const double xf = state[0] + (state[2] * t_look) + (0.5 * state[4] * t_look * t_look);
      const double yf = state[1] + (state[3] * t_look) + (0.5 * state[5] * t_look * t_look);
      published = publish_kalman_target(
        xf, yf, wrapper.tracker.width(), wrapper.tracker.height(), lb, wrapper.obb_angle);
    }
  }

  for (const auto track_id : to_remove) {
    active_trackers_.erase(track_id);
  }
  return published;
}

std::string YoloNode::resolve_default_model_path() const
{
  const auto cwd_candidate =
    std::filesystem::current_path() / "model_test" / "best-roboflow-nitish-obb.engine";
  if (std::filesystem::exists(cwd_candidate)) {
    return cwd_candidate.string();
  }
  return "/home/ultraviolet/CV_Detection/model_test/best-roboflow-nitish-obb.engine";
}

}  // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::YoloNode)
