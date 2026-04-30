#include "yolov8_ros_cpp/detection_postprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace yolov8_ros_cpp
{

namespace
{

constexpr float kNmsIouThreshold = 0.45F;
constexpr float kPi = 3.14159265358979323846F;

template<typename T>
T clamp_value(T value, T lo, T hi)
{
  return std::max(lo, std::min(value, hi));
}

std::string read_file(const std::string & path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Failed to open model file: " + path);
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

std::optional<std::string> extract_engine_metadata(const std::string & data)
{
  if (data.size() < sizeof(std::uint32_t)) {
    return std::nullopt;
  }
  std::uint32_t metadata_len = 0;
  std::memcpy(&metadata_len, data.data(), sizeof(metadata_len));
  const std::size_t json_offset = sizeof(metadata_len);
  if (metadata_len == 0U || json_offset + metadata_len > data.size()) {
    return std::nullopt;
  }
  const std::string metadata = data.substr(json_offset, metadata_len);
  if (metadata.empty() || metadata.front() != '{' || metadata.back() != '}') {
    return std::nullopt;
  }
  return metadata;
}

std::optional<std::string> extract_json_string_value(
  const std::string & json, const std::string & key)
{
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto colon = json.find(':', key_pos + needle.size());
  const auto first_quote = json.find('"', colon + 1);
  if (colon == std::string::npos || first_quote == std::string::npos) {
    return std::nullopt;
  }
  const auto second_quote = json.find('"', first_quote + 1);
  if (second_quote == std::string::npos) {
    return std::nullopt;
  }
  return json.substr(first_quote + 1, second_quote - first_quote - 1);
}

std::unordered_map<int, std::string> extract_names(const std::string & json)
{
  std::unordered_map<int, std::string> names;
  const auto names_key = json.find("\"names\"");
  if (names_key == std::string::npos) {
    return names;
  }
  const auto block_begin = json.find('{', names_key);
  if (block_begin == std::string::npos) {
    return names;
  }
  int depth = 0;
  bool in_string = false;
  bool escaping = false;
  std::size_t block_end = std::string::npos;
  for (std::size_t i = block_begin; i < json.size(); ++i) {
    const char c = json[i];
    if (escaping) {
      escaping = false;
      continue;
    }
    if (c == '\\') {
      escaping = true;
      continue;
    }
    if (c == '"') {
      in_string = !in_string;
      continue;
    }
    if (in_string) {
      continue;
    }
    if (c == '{') {
      ++depth;
    } else if (c == '}') {
      --depth;
      if (depth == 0) {
        block_end = i;
        break;
      }
    }
  }
  if (block_end == std::string::npos) {
    return names;
  }

  const std::string block = json.substr(block_begin + 1, block_end - block_begin - 1);
  std::size_t pos = 0;
  while (true) {
    const auto key_q1 = block.find('"', pos);
    if (key_q1 == std::string::npos) {
      break;
    }
    const auto key_q2 = block.find('"', key_q1 + 1);
    const auto val_q1 = block.find('"', key_q2 + 1);
    const auto val_q2 = block.find('"', val_q1 + 1);
    if (
      key_q2 == std::string::npos || val_q1 == std::string::npos ||
      val_q2 == std::string::npos)
    {
      break;
    }
    names[std::stoi(block.substr(key_q1 + 1, key_q2 - key_q1 - 1))] =
      block.substr(val_q1 + 1, val_q2 - val_q1 - 1);
    pos = val_q2 + 1;
  }
  return names;
}

std::optional<bool> extract_json_bool_value(const std::string & json, const std::string & key)
{
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  const auto value_pos = json.find_first_not_of(" \t\r\n", colon + 1);
  if (value_pos == std::string::npos) {
    return std::nullopt;
  }
  if (json.compare(value_pos, 4, "true") == 0) {
    return true;
  }
  if (json.compare(value_pos, 5, "false") == 0) {
    return false;
  }
  return std::nullopt;
}

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

bool looks_like_end2end_shape(
  const std::vector<int64_t> & shape, bool task_obb, bool end2end_metadata)
{
  if (shape.size() != 3U) {
    return false;
  }
  const int expected_attrs = task_obb ? 7 : 6;
  const int dim1 = static_cast<int>(shape[1]);
  const int dim2 = static_cast<int>(shape[2]);
  if (end2end_metadata) {
    return true;
  }
  return (dim1 > expected_attrs && dim2 == expected_attrs) ||
         (dim2 > expected_attrs && dim1 == expected_attrs);
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
    if (task_obb) {
      detection.obb_angle = at(6);
    }
    detections.push_back(detection);
  }

  return detections;
}

}  // namespace

EngineMetadata read_engine_metadata_from_file(const std::string & model_path)
{
  EngineMetadata metadata{};
  const auto file_data = read_file(model_path);
  const auto json = extract_engine_metadata(file_data);
  if (!json.has_value()) {
    return metadata;
  }
  metadata.class_names = extract_names(*json);
  metadata.task = extract_json_string_value(*json, "task");
  metadata.version = extract_json_string_value(*json, "version");
  metadata.end2end = extract_json_bool_value(*json, "end2end").value_or(false);
  metadata.int8_export = extract_json_bool_value(*json, "int8").value_or(false);
  return metadata;
}

std::vector<Detection> parse_yolo_output_tensor(
  const float * tensor_data,
  const std::vector<int64_t> & shape,
  int img_w,
  int img_h,
  int num_classes,
  float conf_threshold,
  bool task_obb,
  bool end2end,
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

  if (looks_like_end2end_shape(shape, task_obb, end2end)) {
    return parse_end2end_output_tensor(
      tensor_data, shape, img_w, img_h, conf_threshold, task_obb, logger);
  }

  const int channels = static_cast<int>(shape[1]);
  const int boxes = static_cast<int>(shape[2]);
  const int nc = std::max(1, num_classes);
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
    if (task_obb) {
      detection.obb_angle = normalized_angle;
    }
    detections.push_back(detection);
  }

  return detections;
}

}  // namespace yolov8_ros_cpp
