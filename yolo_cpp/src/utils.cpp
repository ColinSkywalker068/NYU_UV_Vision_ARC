/// @file utils.cpp
/// @brief Implements color/team helpers, letterbox transforms, angle math, and HSV checks.

#include "yolo_cpp/utils.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace yolo_cpp
{

namespace
{

constexpr std::array<int, 11> kRedTeam{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
constexpr std::array<int, 11> kBlueTeam{101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111};

/// Choose a saturation threshold that is stricter for dimmer ROIs.
int compute_min_saturation(double mean_value_channel)
{
  if (mean_value_channel >= 200.0) {
    return 55;
  }
  if (mean_value_channel >= 170.0) {
    return 80;
  }
  return 120;
}

/// Choose a minimum value threshold from the ROI brightness.
int compute_min_value(double mean_value_channel)
{
  if (mean_value_channel >= 200.0) {
    return 80;
  }
  return 70;
}

/// Return true when a robot id appears in a fixed team-id array.
template <typename ArrayT>
bool contains_id(const ArrayT& ids, int robot_id)
{
  for (const auto id : ids) {
    if (id == robot_id) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool is_red_team(int robot_id)
{
  return contains_id(kRedTeam, robot_id);
}

bool is_blue_team(int robot_id)
{
  return contains_id(kBlueTeam, robot_id);
}

bool class_id_matches_target_color(int detection_class_id, int target_class)
{
  if (target_class == BLUE_CLASS) {
    return detection_class_id == BLUE_CLASS;
  }
  if (target_class == RED_CLASS) {
    return detection_class_id == RED_CLASS;
  }
  return false;
}

LetterboxMetadata make_letterbox_metadata(int src_w, int src_h, int target_w, int target_h)
{
  if (src_h == target_h && src_w == target_w) {
    return LetterboxMetadata{cv::Mat(), 1.0, 0, 0, src_w, src_h, target_w, target_h};
  }

  const double scale = std::min(static_cast<double>(target_w) / static_cast<double>(src_w),
    static_cast<double>(target_h) / static_cast<double>(src_h));
  const int new_w = static_cast<int>(src_w * scale);
  const int new_h = static_cast<int>(src_h * scale);
  const int pad_w = target_w - new_w;
  const int pad_h = target_h - new_h;
  const int top = pad_h / 2;
  const int left = pad_w / 2;

  return LetterboxMetadata{cv::Mat(), scale, left, top, src_w, src_h, target_w, target_h};
}

LetterboxFrameResult letterbox_frame(const cv::Mat& frame, int target_w, int target_h)
{
  const auto metadata = make_letterbox_metadata(frame.cols, frame.rows, target_w, target_h);

  if (metadata.src_h == target_h && metadata.src_w == target_w) {
    LetterboxFrameResult result = metadata;
    result.frame = frame;
    return result;
  }

  const int new_w = static_cast<int>(frame.cols * metadata.scale);
  const int new_h = static_cast<int>(frame.rows * metadata.scale);
  cv::Mat resized;
  cv::resize(frame, resized, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);

  const int pad_w = target_w - new_w;
  const int pad_h = target_h - new_h;
  const int top = pad_h / 2;
  const int bottom = pad_h - top;
  const int left = pad_w / 2;
  const int right = pad_w - left;

  cv::Mat padded;
  cv::copyMakeBorder(
    resized, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

  LetterboxFrameResult result = metadata;
  result.frame = padded;
  return result;
}

cv::Rect unletterbox_rect(const cv::Rect& rect, const LetterboxResult& lb)
{
  if (lb.scale <= 0.0) {
    return rect;
  }

  const int x = std::max(0, static_cast<int>((rect.x - lb.pad_x) / lb.scale));
  const int y = std::max(0, static_cast<int>((rect.y - lb.pad_y) / lb.scale));
  const int x2 = std::min(lb.src_w, static_cast<int>((rect.x + rect.width - lb.pad_x) / lb.scale));
  const int y2 = std::min(lb.src_h, static_cast<int>((rect.y + rect.height - lb.pad_y) / lb.scale));
  return cv::Rect(x, y, std::max(0, x2 - x), std::max(0, y2 - y));
}

cv::Point2f unletterbox_point(const cv::Point2f& point, const LetterboxMetadata& lb)
{
  if (lb.scale <= 0.0) {
    return point;
  }

  const float x = static_cast<float>((static_cast<double>(point.x) - lb.pad_x) / lb.scale);
  const float y = static_cast<float>((static_cast<double>(point.y) - lb.pad_y) / lb.scale);
  return cv::Point2f(std::clamp(x, 0.0F, static_cast<float>(std::max(0, lb.src_w - 1))),
    std::clamp(y, 0.0F, static_cast<float>(std::max(0, lb.src_h - 1))));
}

std::pair<float, float> calculate_angles(
  const cv::Rect& bbox, int img_w, int img_h, float hfov_rad, float vfov_rad)
{
  const float center_x = static_cast<float>(bbox.x) + static_cast<float>(bbox.width) / 2.0F;
  const float center_y = static_cast<float>(bbox.y) + static_cast<float>(bbox.height) / 2.0F;

  const float pixel_offset_x =
    (center_x - static_cast<float>(img_w) / 2.0F) / (static_cast<float>(img_w) / 2.0F);
  const float pixel_offset_y =
    (center_y - static_cast<float>(img_h) / 2.0F) / (static_cast<float>(img_h) / 2.0F);

  const float yaw = -std::atan(pixel_offset_x * std::tan(hfov_rad / 2.0F));
  const float pitch = std::atan(pixel_offset_y * std::tan(vfov_rad / 2.0F));
  return {yaw, pitch};
}

std::pair<bool, int> check_enemy(const cv::Mat& roi_image, int target_class)
{
  // HSV masks are a fallback when model classes are not trusted for team color.
  if (roi_image.empty()) {
    return {false, 0};
  }

  cv::Mat hsv_roi;
  cv::cvtColor(roi_image, hsv_roi, cv::COLOR_BGR2HSV);
  cv::GaussianBlur(hsv_roi, hsv_roi, cv::Size(3, 3), 0.0);

  const cv::Scalar hsv_mean = cv::mean(hsv_roi);
  const int min_saturation = compute_min_saturation(hsv_mean[2]);
  const int min_value = compute_min_value(hsv_mean[2]);

  cv::Mat final_mask = cv::Mat::zeros(hsv_roi.rows, hsv_roi.cols, CV_8UC1);
  if (target_class == BLUE_CLASS) {
    cv::inRange(
      hsv_roi, cv::Scalar(85, min_saturation, min_value), cv::Scalar(135, 255, 255), final_mask);
  } else {
    cv::Mat low_red_mask;
    cv::Mat high_red_mask;
    cv::inRange(
      hsv_roi, cv::Scalar(0, min_saturation, min_value), cv::Scalar(12, 255, 255), low_red_mask);
    cv::inRange(hsv_roi, cv::Scalar(168, min_saturation, min_value), cv::Scalar(180, 255, 255),
      high_red_mask);
    cv::bitwise_or(low_red_mask, high_red_mask, final_mask);
  }

  cv::morphologyEx(final_mask, final_mask, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 1);
  cv::morphologyEx(final_mask, final_mask, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 1);

  const int color_area = cv::countNonZero(final_mask);
  const double roi_area = static_cast<double>(roi_image.rows) * static_cast<double>(roi_image.cols);
  const double min_ratio = hsv_mean[2] >= 200.0 ? 0.006 : 0.01;
  return {static_cast<double>(color_area) > roi_area * min_ratio, color_area};
}

}  // namespace yolo_cpp
