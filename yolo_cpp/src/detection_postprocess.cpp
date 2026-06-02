/// @file detection_postprocess.cpp
/// @brief Parses YOLO/TensorRT outputs, metadata, corners, and NMS into detections.

#include "yolo_cpp/detection_postprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace yolo_cpp
{

namespace
{

constexpr float kNmsIouThreshold = 0.45F;
constexpr float kPi = 3.14159265358979323846F;

/// Clamp a value into an inclusive range.
template<typename T>
T clamp_value(T value, T lo, T hi)
{
  return std::max(lo, std::min(value, hi));
}

/// Internal candidate representation before conversion to public Detection messages.
struct ParsedCandidate
{
  float cx{0.0F};
  float cy{0.0F};
  float width{0.0F};
  float height{0.0F};
  float angle{0.0F};
  float confidence{0.0F};
  int class_id{0};
};

int resolve_class_count_from_shape(
  int channels,
  int configured_num_classes,
  bool task_obb,
  const rclcpp::Logger * logger)
{
  int num_classes = std::max(1, configured_num_classes);
  const int fixed_channels = task_obb ? 5 : 4;
  const int inferred_num_classes = channels - fixed_channels;
  if (configured_num_classes <= 1 && inferred_num_classes > num_classes) {
    num_classes = inferred_num_classes;
    if (logger != nullptr) {
      RCLCPP_WARN(
        *logger,
        "Model metadata did not provide class count; inferred %d classes from YOLO output shape",
        num_classes);
    }
  }
  return num_classes;
}

/// Normalize YOLO OBB width/height/angle to OpenCV's rotated rectangle convention.
cv::RotatedRect regularize_rbox(const ParsedCandidate & candidate, float * normalized_angle)
{
  float width = candidate.width;
  float height = candidate.height;
  float angle = candidate.angle;
  if (width <= height) {
    std::swap(width, height);
    angle = std::fmod(angle + (kPi / 2.0F), kPi);
  }
  if (normalized_angle != nullptr) {
    *normalized_angle = angle;
  }
  return cv::RotatedRect(
    cv::Point2f(candidate.cx, candidate.cy),
    cv::Size2f(width, height),
    angle * 180.0F / kPi);
}

/// Compute IoU for oriented boxes using OpenCV polygon intersection.
float rotated_iou(const ParsedCandidate & lhs, const ParsedCandidate & rhs)
{
  float lhs_angle = 0.0F;
  float rhs_angle = 0.0F;
  const auto lhs_rect = regularize_rbox(lhs, &lhs_angle);
  const auto rhs_rect = regularize_rbox(rhs, &rhs_angle);
  std::vector<cv::Point2f> intersection;
  const int relation = cv::rotatedRectangleIntersection(lhs_rect, rhs_rect, intersection);
  if (relation == cv::INTERSECT_NONE || intersection.empty()) {
    return 0.0F;
  }
  const float intersection_area = std::abs(cv::contourArea(intersection));
  const float union_area = lhs_rect.size.area() + rhs_rect.size.area() - intersection_area;
  if (union_area <= 0.0F) {
    return 0.0F;
  }
  return intersection_area / union_area;
}

/// Run class-aware non-maximum suppression for oriented boxes.
std::vector<ParsedCandidate> rotated_nms(
  std::vector<ParsedCandidate> candidates, float iou_threshold, int max_det)
{
  std::sort(
    candidates.begin(), candidates.end(),
    [](const ParsedCandidate & lhs, const ParsedCandidate & rhs) {
      return lhs.confidence > rhs.confidence;
    });

  std::vector<ParsedCandidate> kept;
  kept.reserve(std::min<int>(max_det, static_cast<int>(candidates.size())));
  for (const auto & candidate : candidates) {
    bool suppressed = false;
    for (const auto & existing : kept) {
      if (existing.class_id != candidate.class_id) {
        continue;
      }
      if (rotated_iou(existing, candidate) >= iou_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(candidate);
      if (static_cast<int>(kept.size()) >= max_det) {
        break;
      }
    }
  }
  return kept;
}

/// Compute IoU for center-format axis-aligned boxes.
float axis_aligned_iou(const ParsedCandidate & lhs, const ParsedCandidate & rhs)
{
  const float lhs_x1 = lhs.cx - (lhs.width * 0.5F);
  const float lhs_y1 = lhs.cy - (lhs.height * 0.5F);
  const float lhs_x2 = lhs.cx + (lhs.width * 0.5F);
  const float lhs_y2 = lhs.cy + (lhs.height * 0.5F);
  const float rhs_x1 = rhs.cx - (rhs.width * 0.5F);
  const float rhs_y1 = rhs.cy - (rhs.height * 0.5F);
  const float rhs_x2 = rhs.cx + (rhs.width * 0.5F);
  const float rhs_y2 = rhs.cy + (rhs.height * 0.5F);

  const float inter_x1 = std::max(lhs_x1, rhs_x1);
  const float inter_y1 = std::max(lhs_y1, rhs_y1);
  const float inter_x2 = std::min(lhs_x2, rhs_x2);
  const float inter_y2 = std::min(lhs_y2, rhs_y2);
  const float inter_w = std::max(0.0F, inter_x2 - inter_x1);
  const float inter_h = std::max(0.0F, inter_y2 - inter_y1);
  const float inter_area = inter_w * inter_h;
  const float union_area = (lhs.width * lhs.height) + (rhs.width * rhs.height) - inter_area;
  if (union_area <= 0.0F) {
    return 0.0F;
  }
  return inter_area / union_area;
}

/// Run class-aware non-maximum suppression for axis-aligned boxes.
std::vector<ParsedCandidate> axis_aligned_nms(
  std::vector<ParsedCandidate> candidates, float iou_threshold, int max_det)
{
  std::sort(
    candidates.begin(), candidates.end(),
    [](const ParsedCandidate & lhs, const ParsedCandidate & rhs) {
      return lhs.confidence > rhs.confidence;
    });

  std::vector<ParsedCandidate> kept;
  kept.reserve(std::min<int>(max_det, static_cast<int>(candidates.size())));
  for (const auto & candidate : candidates) {
    bool suppressed = false;
    for (const auto & existing : kept) {
      if (existing.class_id != candidate.class_id) {
        continue;
      }
      if (axis_aligned_iou(existing, candidate) >= iou_threshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.push_back(candidate);
      if (static_cast<int>(kept.size()) >= max_det) {
        break;
      }
    }
  }
  return kept;
}

std::array<cv::Point2f, 4> order_armor_corners(const cv::Point2f points[4])
{
  std::array<cv::Point2f, 4> ordered{};
  std::array<cv::Point2f, 4> sorted{points[0], points[1], points[2], points[3]};
  std::sort(
    sorted.begin(), sorted.end(),
    [](const cv::Point2f & lhs, const cv::Point2f & rhs) {
      if (lhs.y != rhs.y) {
        return lhs.y < rhs.y;
      }
      return lhs.x < rhs.x;
    });

  std::array<cv::Point2f, 2> top{sorted[0], sorted[1]};
  std::array<cv::Point2f, 2> bottom{sorted[2], sorted[3]};
  std::sort(top.begin(), top.end(), [](const cv::Point2f & lhs, const cv::Point2f & rhs) {
    return lhs.x < rhs.x;
  });
  std::sort(bottom.begin(), bottom.end(), [](const cv::Point2f & lhs, const cv::Point2f & rhs) {
    return lhs.x < rhs.x;
  });

  ordered[0] = bottom[0];
  ordered[1] = top[0];
  ordered[2] = top[1];
  ordered[3] = bottom[1];
  return ordered;
}

std::array<cv::Point2f, 4> axis_aligned_corners(float x1, float y1, float x2, float y2)
{
  return {
    cv::Point2f(x1, y2),
    cv::Point2f(x1, y1),
    cv::Point2f(x2, y1),
    cv::Point2f(x2, y2)};
}

bool looks_like_end2end_shape(
  const std::vector<int64_t> & shape,
  int num_classes,
  bool task_obb,
  bool end2end_metadata,
  std::optional<bool> export_includes_nms)
{
  if (shape.size() != 3U) {
    return false;
  }
  if (export_includes_nms.has_value()) {
    return *export_includes_nms;
  }
  if (end2end_metadata) {
    return true;
  }

  const int expected_attrs = task_obb ? 7 : 6;
  const int raw_attrs = 4 + std::max(1, num_classes) + (task_obb ? 1 : 0);
  const int dim1 = static_cast<int>(shape[1]);
  const int dim2 = static_cast<int>(shape[2]);
  const bool matches_raw_shape = (dim1 == raw_attrs) || (dim2 == raw_attrs);
  const bool matches_end2end_shape =
    (dim1 > expected_attrs && dim2 == expected_attrs) ||
    (dim2 > expected_attrs && dim1 == expected_attrs);

  if (matches_raw_shape && !matches_end2end_shape) {
    return false;
  }
  if (matches_end2end_shape && !matches_raw_shape) {
    return true;
  }

  // Ambiguous shapes like [1, 7, N] for 2-class OBB models should default to
  // raw output parsing unless the engine metadata explicitly says export NMS was enabled.
  return false;
}

std::vector<Detection> parse_end2end_output_tensor(
  const float * tensor_data,
  const std::vector<int64_t> & shape,
  int img_w,
  int img_h,
  float conf_threshold,
  bool task_obb,
  const rclcpp::Logger * logger)
{
  std::vector<Detection> detections;
  if (shape.size() != 3U) {
    return detections;
  }

  const int expected_attrs = task_obb ? 7 : 6;
  int num_detections = 0;
  bool attrs_last = false;
  if (static_cast<int>(shape[2]) == expected_attrs) {
    num_detections = static_cast<int>(shape[1]);
    attrs_last = true;
  } else if (static_cast<int>(shape[1]) == expected_attrs) {
    num_detections = static_cast<int>(shape[2]);
    attrs_last = false;
  } else {
    if (logger != nullptr) {
      RCLCPP_WARN(
        *logger,
        "Unexpected end2end YOLO output shape [%ld, %ld, %ld]",
        shape[0], shape[1], shape[2]);
    }
    return detections;
  }

  detections.reserve(static_cast<std::size_t>(num_detections));
  for (int i = 0; i < num_detections; ++i) {
    const auto at = [&](int attr) -> float {
        if (attrs_last) {
          return tensor_data[static_cast<std::size_t>((i * expected_attrs) + attr)];
        }
        return tensor_data[static_cast<std::size_t>((attr * num_detections) + i)];
      };

    const float confidence = at(4);
    if (!std::isfinite(confidence) || confidence < conf_threshold) {
      continue;
    }

    const float x1f = clamp_value(at(0), 0.0F, static_cast<float>(img_w - 1));
    const float y1f = clamp_value(at(1), 0.0F, static_cast<float>(img_h - 1));
    const float x2f = clamp_value(at(2), 0.0F, static_cast<float>(img_w - 1));
    const float y2f = clamp_value(at(3), 0.0F, static_cast<float>(img_h - 1));
    const int x1 = static_cast<int>(std::floor(std::min(x1f, x2f)));
    const int y1 = static_cast<int>(std::floor(std::min(y1f, y2f)));
    const int x2 = static_cast<int>(std::ceil(std::max(x1f, x2f)));
    const int y2 = static_cast<int>(std::ceil(std::max(y1f, y2f)));

    Detection detection{};
    detection.bbox_model = cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
    detection.confidence = confidence;
    detection.class_id = std::max(0, static_cast<int>(std::lround(at(5))));
    detection.has_corners = true;
    detection.corners_model = axis_aligned_corners(
      std::min(x1f, x2f),
      std::min(y1f, y2f),
      std::max(x1f, x2f),
      std::max(y1f, y2f));
    if (task_obb) {
      detection.obb_angle = at(6);
    }
    detections.push_back(detection);
  }

  return detections;
}

}  // namespace

std::vector<Detection> parse_yolo_output_tensor(
  const float * tensor_data,
  const std::vector<int64_t> & shape,
  int img_w,
  int img_h,
  int num_classes,
  float conf_threshold,
  bool task_obb,
  bool end2end,
  std::optional<bool> export_includes_nms,
  const rclcpp::Logger * logger)
{
  std::vector<Detection> detections;
  if (tensor_data == nullptr) {
    if (logger != nullptr) {
      RCLCPP_WARN(*logger, "Missing YOLO output tensor data");
    }
    return detections;
  }

  if (shape.size() != 3U) {
    if (logger != nullptr) {
      RCLCPP_WARN(*logger, "Unexpected YOLO output rank: %zu", shape.size());
    }
    return detections;
  }

  if (looks_like_end2end_shape(
      shape, num_classes, task_obb, end2end, export_includes_nms))
  {
    return parse_end2end_output_tensor(
      tensor_data, shape, img_w, img_h, conf_threshold, task_obb, logger);
  }

  const int channels = static_cast<int>(shape[1]);
  const int boxes = static_cast<int>(shape[2]);
  const int nc = resolve_class_count_from_shape(channels, num_classes, task_obb, logger);
  const int extras = channels - 4 - nc;
  if (extras < 0) {
    if (logger != nullptr) {
      RCLCPP_WARN(*logger, "Unexpected YOLO output shape for parser");
    }
    return detections;
  }

  std::vector<ParsedCandidate> parsed;
  parsed.reserve(static_cast<std::size_t>(boxes));
  for (int i = 0; i < boxes; ++i) {
    const auto at = [&](int channel) -> float {
        return tensor_data[static_cast<std::size_t>(channel * boxes + i)];
      };
    float best_conf = -1.0F;
    int best_cls = 0;
    for (int cls = 0; cls < nc; ++cls) {
      const float conf = at(4 + cls);
      if (conf > best_conf) {
        best_conf = conf;
        best_cls = cls;
      }
    }
    if (best_conf < conf_threshold) {
      continue;
    }

    ParsedCandidate candidate{};
    candidate.cx = at(0);
    candidate.cy = at(1);
    candidate.width = at(2);
    candidate.height = at(3);
    candidate.angle = (extras > 0) ? at(4 + nc) : 0.0F;
    candidate.confidence = best_conf;
    candidate.class_id = best_cls;
    parsed.push_back(candidate);
  }

  const auto kept = task_obb ?
    rotated_nms(std::move(parsed), kNmsIouThreshold, 300) :
    axis_aligned_nms(std::move(parsed), kNmsIouThreshold, 300);
  detections.reserve(kept.size());
  for (const auto & candidate : kept) {
    if (!task_obb) {
      const int x1 = static_cast<int>(std::floor(clamp_value(
        candidate.cx - (candidate.width * 0.5F), 0.0F, static_cast<float>(img_w - 1))));
      const int y1 = static_cast<int>(std::floor(clamp_value(
        candidate.cy - (candidate.height * 0.5F), 0.0F, static_cast<float>(img_h - 1))));
      const int x2 = static_cast<int>(std::ceil(clamp_value(
        candidate.cx + (candidate.width * 0.5F), 0.0F, static_cast<float>(img_w - 1))));
      const int y2 = static_cast<int>(std::ceil(clamp_value(
        candidate.cy + (candidate.height * 0.5F), 0.0F, static_cast<float>(img_h - 1))));

      Detection detection{};
      detection.bbox_model = cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
      detection.confidence = candidate.confidence;
      detection.class_id = candidate.class_id;
      detection.has_corners = true;
      detection.corners_model = axis_aligned_corners(
        candidate.cx - (candidate.width * 0.5F),
        candidate.cy - (candidate.height * 0.5F),
        candidate.cx + (candidate.width * 0.5F),
        candidate.cy + (candidate.height * 0.5F));
      detections.push_back(detection);
      continue;
    }

    float normalized_angle = 0.0F;
    const auto rect = regularize_rbox(candidate, &normalized_angle);
    cv::Point2f corners[4];
    rect.points(corners);
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();
    for (const auto & point : corners) {
      min_x = std::min(min_x, point.x);
      min_y = std::min(min_y, point.y);
      max_x = std::max(max_x, point.x);
      max_y = std::max(max_y, point.y);
    }

    const int x1 = static_cast<int>(
      std::floor(clamp_value(min_x, 0.0F, static_cast<float>(img_w - 1))));
    const int y1 = static_cast<int>(
      std::floor(clamp_value(min_y, 0.0F, static_cast<float>(img_h - 1))));
    const int x2 = static_cast<int>(
      std::ceil(clamp_value(max_x, 0.0F, static_cast<float>(img_w - 1))));
    const int y2 = static_cast<int>(
      std::ceil(clamp_value(max_y, 0.0F, static_cast<float>(img_h - 1))));

    Detection detection{};
    detection.bbox_model = cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
    detection.confidence = candidate.confidence;
    detection.class_id = candidate.class_id;
    detection.has_corners = true;
    detection.corners_model = order_armor_corners(corners);
    if (task_obb) {
      detection.obb_angle = normalized_angle;
    }
    detections.push_back(detection);
  }

  return detections;
}

}  // namespace yolo_cpp
