#ifndef YOLO_CPP__UTILS_HPP_
#define YOLO_CPP__UTILS_HPP_

#include <array>
#include <opencv2/core.hpp>
#include <optional>
#include <tuple>
#include <vector>

namespace yolo_cpp
{

constexpr int RED_CLASS = 1;
constexpr int BLUE_CLASS = 0;

/// Return true when robot_id belongs to the red alliance convention.
bool is_red_team(int robot_id);

/// Return true when robot_id belongs to the blue alliance convention.
bool is_blue_team(int robot_id);

/// Return true when a detection class id matches the currently selected enemy color.
bool class_id_matches_target_color(int detection_class_id, int target_class);

/// Result of resizing an image with letterbox padding.
struct LetterboxResult
{
  cv::Mat frame;
  double scale{1.0};
  int pad_x{0};
  int pad_y{0};
  int src_w{0};
  int src_h{0};
  int dst_w{0};
  int dst_h{0};
};

using LetterboxMetadata = LetterboxResult;
using LetterboxFrameResult = LetterboxResult;

/// Detector candidate transformed into source-image coordinates.
struct Candidate
{
  cv::Rect bbox_model;
  cv::Rect bbox_source;
  cv::Point2f center_source;
  float confidence{0.0F};
  int class_id{0};
  std::optional<int> track_id;
  bool is_enemy{true};
  std::optional<float> obb_angle;
  bool has_corners{false};
  std::array<cv::Point2f, 4> corners_source{};
};

/// Resize and pad frame to target dimensions while preserving aspect ratio.
LetterboxFrameResult letterbox_frame(const cv::Mat& frame, int target_w, int target_h);

/// Build letterbox metadata without materializing a resized frame.
LetterboxMetadata make_letterbox_metadata(int src_w, int src_h, int target_w, int target_h);

/// Map a rectangle from letterboxed model coordinates back to source coordinates.
cv::Rect unletterbox_rect(const cv::Rect& rect, const LetterboxMetadata& lb);

/// Map a point from letterboxed model coordinates back to source coordinates.
cv::Point2f unletterbox_point(const cv::Point2f& point, const LetterboxMetadata& lb);

/// Convert image-space target offset into yaw/pitch angles using camera FOV.
std::pair<float, float> calculate_angles(
  const cv::Rect& bbox, int img_w, int img_h, float hfov_rad, float vfov_rad);

/// Scale relative yaw/pitch errors to make the aim output more aggressive.
std::pair<float, float> apply_angle_gain(float yaw_rad, float pitch_rad, float gain);

/// Estimate whether an ROI matches the configured enemy color using HSV thresholds.
std::pair<bool, int> check_enemy(const cv::Mat& roi_image, int target_class);

}  // namespace yolo_cpp

#endif
