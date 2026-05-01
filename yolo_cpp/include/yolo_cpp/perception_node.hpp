#ifndef YOLO_CPP__PERCEPTION_NODE_HPP_
#define YOLO_CPP__PERCEPTION_NODE_HPP_

#include "yolo_cpp/camera_backend.hpp"
#include "yolo_cpp/deepstream_runtime.hpp"
#include "yolo_cpp/detector_backend.hpp"
#include "yolo_cpp/tracker.hpp"
#include "yolo_cpp/utils.hpp"
#include "yolo_cpp/visibility_control.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <chrono>

namespace yolo_cpp
{

class YOLO_CPP_PUBLIC PerceptionNode : public rclcpp::Node
{
public:
  explicit PerceptionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~PerceptionNode() override;

private:
  struct StageTotals
  {
    double capture_ms{0.0};
    double preprocess_ms{0.0};
    double upload_ms{0.0};
    double inference_ms{0.0};
    double download_ms{0.0};
    double parse_ms{0.0};
    double tracking_ms{0.0};
    double color_ms{0.0};
    double publish_ms{0.0};
  };

  void id_listener_callback(const std_msgs::msg::Int32::SharedPtr msg);
  void handle_attitude_flags(const std_msgs::msg::Int32MultiArray::SharedPtr msg);
  void timer_callback();
  void handle_deepstream_result(DeepStreamRuntime::Result result);
  void initialize_native_pipeline();
  void initialize_deepstream_runtime();
  void process_detection_cycle(
    const DetectionBatch & batch,
    const FramePacket * packet,
    const DetectorStageTimings & detector_timings,
    double capture_ms);
  void publish_debug_image(const FramePacket & packet);
  void render_preview(
    const FramePacket & packet,
    const std::vector<Candidate> & candidates,
    const std::optional<Candidate> & target);
  std::vector<Candidate> extract_candidates(
    const DetectionBatch & batch, const FramePacket * packet, double * color_ms) const;
  std::vector<Candidate> rank_candidates(std::vector<Candidate> candidates) const;
  std::optional<Candidate> select_target(const std::vector<Candidate> & candidates) const;
  void publish_target(const Candidate & target, const LetterboxMetadata & lb);
  void publish_no_target(int src_w, int src_h);
  bool publish_kalman_target(
    double xf, double yf, double width, double height, const LetterboxMetadata & lb,
    const std::optional<float> & obb_angle);
  bool run_kalman_pipeline(
    const std::vector<Candidate> & candidates, double dt, const LetterboxMetadata & lb);
  std::string resolve_default_model_path() const;
  std::string resolve_default_deepstream_config_path() const;
  void log_stage_timing();

  std::shared_ptr<CameraBackend> camera_backend_;
  std::unique_ptr<DetectorBackend> detector_backend_;
  std::unique_ptr<DeepStreamRuntime> deepstream_runtime_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr robot_id_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr attitude_flags_subscription_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr bbox_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr override_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr angle_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr obb_angle_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_publisher_;

  std_msgs::msg::Int32MultiArray bbox_msg_;
  std_msgs::msg::Bool override_msg_;
  std_msgs::msg::Float32MultiArray angle_msg_;
  std_msgs::msg::Float32 obb_angle_msg_;

  std::map<int, TrackerWrapper> active_trackers_;
  rclcpp::Time last_frame_time_;
  bool have_last_frame_time_{false};
  std::chrono::steady_clock::time_point last_stage_log_wall_time_{};
  bool have_last_stage_log_wall_time_{false};
  std::mutex state_mutex_;

  CameraConfig camera_config_;
  std::string model_path_;
  std::string task_;
  std::string model_type_;
  std::string detector_backend_kind_;
  std::string deepstream_config_path_;
  std::string inference_mode_;
  std::string target_color_;
  std::string vis_window_name_;
  bool allow_native_tensorrt_fallback_{false};
  bool half_{true};
  bool show_{false};
  bool persist_tracking_{true};
  bool use_kalman_{false};
  bool enable_color_verification_{true};
  bool use_model_color_classes_{true};
  bool use_robot_id_targeting_{false};
  bool debug_publish_image_raw_{false};
  bool preview_failed_{false};
  bool stm_color_locked_{false};
  int imgsz_width_{640};
  int imgsz_height_{640};
  int tracker_timeout_{15};
  int max_hsv_candidates_{3};
  int max_coast_frames_{3};
  int stage_log_interval_{300};
  int frame_count_{0};
  int enemy_color_red_value_{0};
  int enemy_color_blue_value_{1};
  float conf_{0.30F};
  float lookahead_steps_{0.5F};
  float hfov_rad_{0.0F};
  float vfov_rad_{0.0F};
  int target_class_{BLUE_CLASS};
  double fps_{60.0};
  double preview_fps_{0.0};
  StageTotals stage_totals_;
};

}  // namespace yolo_cpp

#endif
