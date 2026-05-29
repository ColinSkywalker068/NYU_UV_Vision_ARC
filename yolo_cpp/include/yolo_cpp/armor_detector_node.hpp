#ifndef YOLO_CPP__ARMOR_DETECTOR_NODE_HPP_
#define YOLO_CPP__ARMOR_DETECTOR_NODE_HPP_

#include "cv_auto_aim_interfaces/msg/armors.hpp"
#include "cv_auto_aim_interfaces/msg/target.hpp"
#include "yolo_cpp/camera_backend.hpp"
#include "yolo_cpp/deepstream_runtime.hpp"
#include "yolo_cpp/detector_backend.hpp"
#include "yolo_cpp/pnp_solver.hpp"
#include "yolo_cpp/utils.hpp"
#include "yolo_cpp/visibility_control.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace yolo_cpp
{

/// Captures camera frames, runs armor detection, validates color, and publishes candidates.
class YOLO_CPP_PUBLIC ArmorDetectorNode : public rclcpp::Node
{
public:
  /// Create the detector node and initialize the configured camera/inference pipeline.
  explicit ArmorDetectorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

  /// Release camera, detector, DeepStream, and log resources.
  ~ArmorDetectorNode() override;

private:
  struct StageTotals
  {
    double capture_ms{0.0};
    double preprocess_ms{0.0};
    double upload_ms{0.0};
    double inference_ms{0.0};
    double download_ms{0.0};
    double parse_ms{0.0};
    double color_ms{0.0};
    double publish_ms{0.0};
  };

  struct CameraCalibration
  {
    PnPCameraModel model;
    bool available{false};
    bool from_camera_info{false};
    int image_width{0};
    int image_height{0};
  };

  /// Update enemy-color targeting from the robot id topic.
  void id_listener_callback(const std_msgs::msg::Int32::SharedPtr msg);

  /// Update controller-provided targeting flags such as enemy color lock.
  void handle_attitude_flags(const std_msgs::msg::Int32MultiArray::SharedPtr msg);

  /// Cache the selected target for the optional OpenCV preview overlay.
  void handle_preview_target(const cv_auto_aim_interfaces::msg::Target::SharedPtr msg);

  /// Cache calibrated camera intrinsics for PnP pose solving.
  void handle_camera_info(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  /// Capture and process one frame in the native synchronous pipeline.
  void timer_callback();

  /// Process one asynchronous DeepStream result.
  void handle_deepstream_result(DeepStreamRuntime::Result result);

  /// Create the native camera and detector backends.
  void initialize_native_pipeline();

  /// Create and start the DeepStream runtime.
  void initialize_deepstream_runtime();

  /// Convert detections into candidates, publish messages, previews, and timing logs.
  void process_detection_cycle(
    const DetectionBatch & batch,
    const FramePacket * packet,
    const DetectorStageTimings & detector_timings,
    double capture_ms);
  /// Publish a cv_auto_aim_interfaces/Armors message for detector output.
  void publish_armors(
    const std::vector<Candidate> & candidates,
    const std::string & source_id,
    const rclcpp::Time & stamp,
    int image_width,
    int image_height);
  /// Solve a camera-frame pose for a candidate when corners and intrinsics are available.
  std::optional<geometry_msgs::msg::Pose> solve_candidate_pose(
    const Candidate & candidate, int image_width, int image_height);
  /// Build PnP calibration from static parameters when CameraInfo is unavailable.
  void configure_static_camera_model();

  /// Create a pinhole camera model from configured horizontal/vertical FOV.
  PnPCameraModel make_fov_camera_model(int image_width, int image_height) const;

  /// Publish a ROS debug image for downstream visualization.
  void publish_debug_image(const FramePacket & packet, const rclcpp::Time & stamp);

  /// Render the optional OpenCV preview window.
  void render_preview(const FramePacket & packet, const std::vector<Candidate> & candidates);

  /// Save a bounded set of target frames for offline visual inspection and heatmaps.
  void save_target_frame_if_needed(
    const FramePacket & packet,
    const std::vector<Candidate> & candidates,
    const rclcpp::Time & stamp);

  /// Apply letterbox reversal, color filtering, and candidate shaping.
  std::vector<Candidate> extract_candidates(
    const DetectionBatch & batch, const FramePacket * packet, double * color_ms) const;
  /// Resolve the default model path from package/runtime conventions.
  std::string resolve_default_model_path() const;

  /// Resolve the default DeepStream nvinfer config path.
  std::string resolve_default_deepstream_config_path() const;

  /// Resolve the directory used for runtime CSV-style logs.
  std::string resolve_runtime_log_dir() const;

  /// Resolve the directory used for optional target-frame captures.
  std::string resolve_target_frame_save_dir() const;

  /// Append one line to the runtime log when logging is configured.
  void write_runtime_log_line(const std::string & line);

  /// Write a structured runtime note to ROS logs and the runtime log file.
  void log_runtime_note(const std::string & level, const std::string & message);

  /// Emit an FPS log once per wall-clock second.
  void log_runtime_fps_once_per_second();

  /// Log a target/color-verification event for offline tuning.
  void log_runtime_opponent_event(const Candidate & target, bool color_pass, double latency_ms);

  /// Periodically log accumulated stage timings.
  void log_stage_timing();

  std::shared_ptr<CameraBackend> camera_backend_;
  std::unique_ptr<DetectorBackend> detector_backend_;
  std::unique_ptr<DeepStreamRuntime> deepstream_runtime_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr robot_id_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr attitude_flags_subscription_;
  rclcpp::Subscription<cv_auto_aim_interfaces::msg::Target>::SharedPtr preview_target_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscription_;
  rclcpp::Publisher<cv_auto_aim_interfaces::msg::Armors>::SharedPtr armors_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_publisher_;

  rclcpp::Time last_frame_time_;
  bool have_last_frame_time_{false};
  std::chrono::steady_clock::time_point last_stage_log_wall_time_{};
  bool have_last_stage_log_wall_time_{false};
  std::chrono::steady_clock::time_point last_runtime_fps_log_time_{};
  bool have_last_runtime_fps_log_time_{false};
  std::mutex state_mutex_;

  CameraConfig camera_config_;
  std::string model_path_;
  std::string task_;
  std::string model_type_;
  std::string detector_backend_kind_;
  std::string deepstream_config_path_;
  std::string inference_mode_;
  std::string runtime_log_dir_;
  std::string runtime_log_path_;
  std::string target_frame_save_dir_;
  std::string target_frame_index_path_;
  std::string target_color_;
  std::string preview_target_topic_;
  std::string vis_window_name_;
  std::string camera_frame_id_;
  std::string camera_info_topic_;
  std::string bayer_conversion_backend_;
  bool allow_native_tensorrt_fallback_{false};
  bool half_{true};
  bool show_{false};
  bool enable_color_verification_{true};
  bool use_model_color_classes_{true};
  bool use_robot_id_targeting_{false};
  bool debug_publish_image_raw_{false};
  bool save_target_frames_{false};
  bool pre_mux_resize_{false};
  bool deepstream_profile_stages_{false};
  bool enable_3d_pose_{true};
  bool preview_failed_{false};
  bool preview_has_target_{false};
  bool stm_color_locked_{false};
  int imgsz_width_{640};
  int imgsz_height_{640};
  int pre_mux_width_{0};
  int pre_mux_height_{0};
  int deepstream_profile_stage_interval_{300};
  int max_hsv_candidates_{3};
  int stage_log_interval_{300};
  int frame_count_{0};
  int target_frame_save_count_{0};
  int target_frame_save_max_{100};
  int target_frame_jpeg_quality_{85};
  std::uint64_t stage_raw_detections_{0};
  std::uint64_t stage_candidates_{0};
  std::uint64_t stage_enemy_candidates_{0};
  std::uint64_t stage_class0_candidates_{0};
  std::uint64_t stage_class1_candidates_{0};
  std::uint64_t stage_other_class_candidates_{0};
  int enemy_color_red_value_{0};
  int enemy_color_blue_value_{1};
  int preview_target_track_id_{-1};
  std::array<int32_t, 4> preview_target_bbox_xyxy_{0, 0, 0, 0};
  float conf_{0.30F};
  int target_class_{BLUE_CLASS};
  double fps_{60.0};
  double hfov_rad_{0.0};
  double vfov_rad_{0.0};
  double camera_fx_{0.0};
  double camera_fy_{0.0};
  double camera_cx_{0.0};
  double camera_cy_{0.0};
  double small_armor_width_m_{0.135};
  double small_armor_height_m_{0.055};
  double large_armor_width_m_{0.225};
  double large_armor_height_m_{0.055};
  double large_armor_min_aspect_ratio_{3.2};
  double preview_fps_{0.0};
  CameraCalibration camera_calibration_;
  std::optional<ArmorPnPSolver> pnp_solver_;
  std::vector<double> camera_distortion_;
  StageTotals stage_totals_;
  std::ofstream runtime_log_stream_;
  std::ofstream target_frame_index_stream_;
};

}  // namespace yolo_cpp

#endif
