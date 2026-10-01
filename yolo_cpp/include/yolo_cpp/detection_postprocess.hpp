#ifndef YOLO_CPP__DETECTION_POSTPROCESS_HPP_
#define YOLO_CPP__DETECTION_POSTPROCESS_HPP_

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "yolo_cpp/detector_backend.hpp"

namespace yolo_cpp
{

struct EngineMetadata
{
  /// Class id to human-readable class name map parsed from serialized engine metadata.
  std::unordered_map<int, std::string> class_names;
  /// YOLO export task, such as "detect" or "obb", when embedded in the engine.
  std::optional<std::string> task;
  /// Exporter or engine version string when present in the plan.
  std::optional<std::string> version;
  /// True when the exported model already includes final detection heads/NMS.
  bool end2end{false};
  /// True when the engine was exported for INT8 execution.
  bool int8_export{false};
  /// Whether the serialized export includes NMS, if the metadata can prove it.
  std::optional<bool> export_includes_nms;
};

/// Read YOLO/TensorRT metadata embedded next to or inside an engine file.
///
/// Args:
///   model_path: Path to the model or engine file.
///
/// Returns:
///   Parsed metadata. Missing fields remain unset so callers can fall back to
///   launch parameters or model defaults.
EngineMetadata read_engine_metadata_from_file(const std::string& model_path);

/// Read a TensorRT plan file as an opaque string for metadata scanning.
///
/// Args:
///   model_path: Path to the serialized TensorRT engine.
///
/// Returns:
///   Raw plan contents. An empty string indicates the file could not be read.
std::string read_engine_plan_from_file(const std::string& model_path);

/// Convert a raw YOLO output tensor into source-image detections.
///
/// Args:
///   tensor_data: Contiguous output tensor data from TensorRT.
///   shape: Tensor shape in exporter order.
///   img_w: Source image width before letterboxing.
///   img_h: Source image height before letterboxing.
///   num_classes: Number of class logits in each prediction.
///   conf_threshold: Minimum confidence kept after score decoding.
///   task_obb: True when the tensor carries oriented-box angle data.
///   end2end: True when the engine emits post-NMS detections.
///   export_includes_nms: Optional override for NMS behavior detected at export.
///   logger: Optional ROS logger used for one-off shape diagnostics.
///
/// Returns:
///   Detections in model-space coordinates, with OBB fields populated when
///   available.
std::vector<Detection> parse_yolo_output_tensor(const float* tensor_data,
  const std::vector<int64_t>& shape, int img_w, int img_h, int num_classes, float conf_threshold,
  bool task_obb, bool end2end, std::optional<bool> export_includes_nms = std::nullopt,
  const rclcpp::Logger* logger = nullptr);

// Explicit contract for ARUW's pinned 2026 16-class, four-corner model.
std::vector<Detection> parse_aruw_pose_tensor(const float * data,
  const std::vector<int64_t> & shape, int width, int height, float confidence);
LetterboxMetadata aruw_resize_metadata(int source_width, int source_height);
cv::Mat preprocess_aruw_bgr(const cv::Mat & bgr);

}  // namespace yolo_cpp

#endif
