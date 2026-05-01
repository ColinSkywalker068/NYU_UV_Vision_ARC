#ifndef YOLO_CPP__UTILS_HPP_
#define YOLO_CPP__UTILS_HPP_

#include <opencv2/core.hpp>

#include <array>
#include <optional>
#include <tuple>
#include <vector>

namespace yolo_cpp
{

constexpr int RED_CLASS = 1;
constexpr int BLUE_CLASS = 0;

bool is_red_team(int robot_id);
bool is_blue_team(int robot_id);

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
};

LetterboxFrameResult letterbox_frame(const cv::Mat & frame, int target_w, int target_h);
LetterboxMetadata make_letterbox_metadata(int src_w, int src_h, int target_w, int target_h);
cv::Rect unletterbox_rect(const cv::Rect & rect, const LetterboxMetadata & lb);
std::pair<float, float> calculate_angles(
  const cv::Rect & bbox, int img_w, int img_h, float hfov_rad, float vfov_rad);
std::pair<bool, int> check_enemy(const cv::Mat & roi_image, int target_class);

}  // namespace yolo_cpp

#endif
