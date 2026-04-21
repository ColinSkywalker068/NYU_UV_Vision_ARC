#include "yolov8_ros_cpp/deepstream_detector.hpp"

#include <opencv2/imgproc.hpp>
#include <rclcpp/logger.hpp>
#include <rclcpp/logging.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

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

std::uint16_t float_to_half_bits(float value)
{
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    mantissa |= 0x800000U;
    const std::uint32_t shifted = mantissa >> static_cast<std::uint32_t>(14 - exponent);
    return static_cast<std::uint16_t>(sign | ((shifted + 1U) >> 1));
  }

  if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }

  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13));
}

float half_bits_to_float(std::uint16_t value)
{
  const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000U)) << 16;
  const std::uint32_t exponent = (value >> 10) & 0x1FU;
  const std::uint32_t mantissa = value & 0x03FFU;
  std::uint32_t bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      std::uint32_t mant = mantissa;
      std::int32_t exp = -14;
      while ((mant & 0x0400U) == 0U) {
        mant <<= 1U;
        --exp;
      }
      mant &= 0x03FFU;
      bits = sign |
        (static_cast<std::uint32_t>(exp + 127) << 23) |
        (mant << 13);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    bits = sign |
      ((exponent + 112U) << 23) |
      (mantissa << 13);
  }

  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
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

std::optional<std::string> extract_json_string_value(const std::string & json, const std::string & key)
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

  std::string block = json.substr(block_begin + 1, block_end - block_begin - 1);
  std::size_t pos = 0;
  while (true) {
    const auto key_q1 = block.find('"', pos);
    if (key_q1 == std::string::npos) {
      break;
    }
    const auto key_q2 = block.find('"', key_q1 + 1);
    const auto val_q1 = block.find('"', key_q2 + 1);
    const auto val_q2 = block.find('"', val_q1 + 1);
    if (key_q2 == std::string::npos || val_q1 == std::string::npos || val_q2 == std::string::npos) {
      break;
    }
    names[std::stoi(block.substr(key_q1 + 1, key_q2 - key_q1 - 1))] =
      block.substr(val_q1 + 1, val_q2 - val_q1 - 1);
    pos = val_q2 + 1;
  }
  return names;
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

#ifdef HAVE_TENSORRT
template<typename T>
struct TrtDeleter
{
  void operator()(T * ptr) const
  {
    delete ptr;
  }
};

class TrtLogger : public nvinfer1::ILogger
{
public:
  explicit TrtLogger(rclcpp::Logger logger) : logger_(logger) {}

  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      RCLCPP_ERROR(logger_, "TensorRT: %s", msg);
    } else if (severity == Severity::kWARNING) {
      RCLCPP_WARN(logger_, "TensorRT: %s", msg);
    } else {
      RCLCPP_INFO(logger_, "TensorRT: %s", msg);
    }
  }

private:
  rclcpp::Logger logger_;
};

struct DeepStreamDetector::Impl
{
  explicit Impl(rclcpp::Logger logger_in) : logger(std::move(logger_in)), trt_logger(logger) {}

  rclcpp::Logger logger;
  TrtLogger trt_logger;
  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context;
  cudaStream_t stream{};
  std::string input_name;
  std::string output_name;
  int input_index{-1};
  int output_index{-1};
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  std::vector<int64_t> runtime_input_shape;
  std::vector<int64_t> runtime_output_shape;
  std::vector<float> host_input;
  std::vector<float> host_output;
  std::vector<std::uint16_t> host_input_half;
  std::vector<std::uint16_t> host_output_half;
  void * device_input{nullptr};
  void * device_output{nullptr};
  nvinfer1::DataType input_dtype{nvinfer1::DataType::kFLOAT};
  nvinfer1::DataType output_dtype{nvinfer1::DataType::kFLOAT};
  int img_w{640};
  int img_h{640};
  int nc{1};
  bool end2end{false};
  bool task_obb{false};

  ~Impl()
  {
    if (device_input != nullptr) {
      cudaFree(device_input);
    }
    if (device_output != nullptr) {
      cudaFree(device_output);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }
};

std::vector<int64_t> dims_to_vector(const nvinfer1::Dims & dims)
{
  std::vector<int64_t> result;
  result.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; ++i) {
    result.push_back(static_cast<int64_t>(dims.d[i]));
  }
  return result;
}

int64_t volume_of(const std::vector<int64_t> & shape)
{
  int64_t volume = 1;
  for (const auto dim : shape) {
    if (dim <= 0) {
      return -1;
    }
    volume *= dim;
  }
  return volume;
}

std::vector<int64_t> resolve_runtime_shape(
  const std::vector<int64_t> & shape, int img_h, int img_w)
{
  auto resolved = shape;
  if (resolved.size() >= 4) {
    if (resolved[0] < 0) {
      resolved[0] = 1;
    }
    if (resolved[resolved.size() - 2] < 0) {
      resolved[resolved.size() - 2] = img_h;
    }
    if (resolved[resolved.size() - 1] < 0) {
      resolved[resolved.size() - 1] = img_w;
    }
  }
  return resolved;
}

void throw_if_cuda_failed(cudaError_t status, const char * what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}
#endif

}  // namespace

DeepStreamDetector::DeepStreamDetector(
  std::string model_path, std::string task, std::string model_type, bool half,
  float conf_threshold)
: model_path_(std::move(model_path)),
  task_(std::move(task)),
  model_type_(std::move(model_type)),
  half_(half),
  conf_threshold_(conf_threshold)
{
}

void DeepStreamDetector::ensure_initialized(const rclcpp::Logger & logger)
{
  if (initialized_) {
    return;
  }

  class_names_.clear();
  try {
    const auto file_data = read_file(model_path_);
    if (const auto metadata = extract_engine_metadata(file_data)) {
      class_names_ = extract_names(*metadata);
      if (const auto task = extract_json_string_value(*metadata, "task")) {
        task_ = *task;
      }
      if (class_names_.empty()) {
        class_names_.emplace(0, "plate");
      }
    }
  } catch (const std::exception & exc) {
    RCLCPP_WARN(logger, "Failed reading engine metadata: %s", exc.what());
  }

#ifdef HAVE_TENSORRT
  impl_ = std::make_unique<Impl>(logger);
  impl_->task_obb = (task_ == "obb");
  impl_->nc = class_names_.empty() ? 1 : static_cast<int>(class_names_.size());

  const std::string engine_bytes = read_file(model_path_);
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->trt_logger));
  if (!impl_->runtime) {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }
  impl_->engine.reset(impl_->runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
  if (!impl_->engine) {
    throw std::runtime_error("Failed to deserialize TensorRT engine");
  }
  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) {
    throw std::runtime_error("Failed to create TensorRT execution context");
  }
  throw_if_cuda_failed(cudaStreamCreate(&impl_->stream), "cudaStreamCreate failed");

  const int nb = impl_->engine->getNbBindings();
  for (int i = 0; i < nb; ++i) {
    const char * name = impl_->engine->getBindingName(i);
    const auto dims = dims_to_vector(impl_->engine->getBindingDimensions(i));
    if (impl_->engine->bindingIsInput(i)) {
      impl_->input_index = i;
      impl_->input_name = name;
      impl_->input_shape = dims;
      if (dims.size() >= 4) {
        if (dims[dims.size() - 2] > 0) {
          impl_->img_h = static_cast<int>(dims[dims.size() - 2]);
        }
        if (dims[dims.size() - 1] > 0) {
          impl_->img_w = static_cast<int>(dims[dims.size() - 1]);
        }
      }
      impl_->input_dtype = impl_->engine->getBindingDataType(i);
    } else {
      impl_->output_index = i;
      impl_->output_name = name;
      impl_->output_shape = dims;
      impl_->output_dtype = impl_->engine->getBindingDataType(i);
    }
  }

  if (impl_->input_index < 0 || impl_->output_index < 0) {
    throw std::runtime_error("TensorRT engine missing input/output bindings");
  }

  impl_->runtime_input_shape = resolve_runtime_shape(impl_->input_shape, impl_->img_h, impl_->img_w);
  if (volume_of(impl_->runtime_input_shape) <= 0) {
    throw std::runtime_error("Unable to resolve TensorRT input dimensions");
  }

  nvinfer1::Dims input_dims{};
  input_dims.nbDims = static_cast<int>(impl_->runtime_input_shape.size());
  for (int i = 0; i < input_dims.nbDims; ++i) {
    input_dims.d[i] = static_cast<int>(impl_->runtime_input_shape[static_cast<std::size_t>(i)]);
  }
  if (!impl_->context->setBindingDimensions(impl_->input_index, input_dims)) {
    throw std::runtime_error("Failed to set TensorRT binding dimensions");
  }

  impl_->runtime_output_shape = dims_to_vector(
    impl_->context->getBindingDimensions(impl_->output_index));
  if (volume_of(impl_->runtime_output_shape) <= 0) {
    throw std::runtime_error("Unable to resolve TensorRT output dimensions");
  }

  if (
    impl_->input_dtype != nvinfer1::DataType::kFLOAT &&
    impl_->input_dtype != nvinfer1::DataType::kHALF)
  {
    throw std::runtime_error("Only FP32 and FP16 TensorRT input bindings are supported right now");
  }
  if (
    impl_->output_dtype != nvinfer1::DataType::kFLOAT &&
    impl_->output_dtype != nvinfer1::DataType::kHALF)
  {
    throw std::runtime_error("Only FP32 and FP16 TensorRT output bindings are supported right now");
  }

  const auto input_volume = static_cast<std::size_t>(volume_of(impl_->runtime_input_shape));
  const auto output_volume = static_cast<std::size_t>(volume_of(impl_->runtime_output_shape));
  impl_->host_input.resize(input_volume);
  impl_->host_output.resize(static_cast<std::size_t>(volume_of(impl_->runtime_output_shape)));
  if (impl_->input_dtype == nvinfer1::DataType::kHALF) {
    impl_->host_input_half.resize(input_volume);
  }
  if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    impl_->host_output_half.resize(output_volume);
  }
  throw_if_cuda_failed(
    cudaMalloc(
      &impl_->device_input,
      input_volume * ((impl_->input_dtype == nvinfer1::DataType::kHALF) ? sizeof(std::uint16_t) : sizeof(float))),
    "cudaMalloc input buffer failed");
  throw_if_cuda_failed(
    cudaMalloc(
      &impl_->device_output,
      output_volume * ((impl_->output_dtype == nvinfer1::DataType::kHALF) ? sizeof(std::uint16_t) : sizeof(float))),
    "cudaMalloc output buffer failed");
  backend_name_ = "tensorrt_native";
#else
  if (!warned_unavailable_) {
    RCLCPP_WARN(
      logger,
      "TensorRT headers/libs were not available at build time. Detector will stay in stub mode.");
    warned_unavailable_ = true;
  }
#endif

  initialized_ = true;
}

std::vector<Detection> DeepStreamDetector::infer(const cv::Mat & frame, const rclcpp::Logger & logger)
{
  ensure_initialized(logger);

#ifndef HAVE_TENSORRT
  if (!warned_unavailable_) {
    RCLCPP_WARN(logger, "TensorRT inference unavailable in this build; detector returning no detections.");
    warned_unavailable_ = true;
  }
  return {};
#else
  if (!impl_) {
    return {};
  }

  cv::Mat rgb;
  cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
  int cursor = 0;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < rgb.rows; ++y) {
      for (int x = 0; x < rgb.cols; ++x) {
        impl_->host_input[static_cast<std::size_t>(cursor++)] =
          static_cast<float>(rgb.at<cv::Vec3b>(y, x)[c]) / 255.0F;
      }
    }
  }

  const void * input_ptr = impl_->host_input.data();
  std::size_t input_bytes = impl_->host_input.size() * sizeof(float);
  if (impl_->input_dtype == nvinfer1::DataType::kHALF) {
    std::transform(
      impl_->host_input.begin(), impl_->host_input.end(), impl_->host_input_half.begin(),
      [](float value) {return float_to_half_bits(value);});
    input_ptr = impl_->host_input_half.data();
    input_bytes = impl_->host_input_half.size() * sizeof(std::uint16_t);
  }

  throw_if_cuda_failed(cudaMemcpyAsync(
    impl_->device_input, input_ptr, input_bytes, cudaMemcpyHostToDevice, impl_->stream),
    "TensorRT input upload failed");

  std::vector<void *> bindings(static_cast<std::size_t>(impl_->engine->getNbBindings()), nullptr);
  bindings[static_cast<std::size_t>(impl_->input_index)] = impl_->device_input;
  bindings[static_cast<std::size_t>(impl_->output_index)] = impl_->device_output;
  if (!impl_->context->enqueueV2(bindings, impl_->stream, nullptr)) {
    throw std::runtime_error("TensorRT enqueue failed");
  }

  void * output_ptr = impl_->host_output.data();
  std::size_t output_bytes = impl_->host_output.size() * sizeof(float);
  if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    output_ptr = impl_->host_output_half.data();
    output_bytes = impl_->host_output_half.size() * sizeof(std::uint16_t);
  }

  throw_if_cuda_failed(cudaMemcpyAsync(
    output_ptr, impl_->device_output, output_bytes, cudaMemcpyDeviceToHost, impl_->stream),
    "TensorRT output download failed");
  throw_if_cuda_failed(cudaStreamSynchronize(impl_->stream), "TensorRT stream sync failed");
  if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    std::transform(
      impl_->host_output_half.begin(), impl_->host_output_half.end(), impl_->host_output.begin(),
      [](std::uint16_t value) {return half_bits_to_float(value);});
  }

  std::vector<ParsedCandidate> parsed;
  const auto & shape = impl_->runtime_output_shape;
  if (shape.size() != 3) {
    RCLCPP_WARN(logger, "Unexpected TensorRT output rank: %zu", shape.size());
    return {};
  }

  const int channels = static_cast<int>(shape[1]);
  const int boxes = static_cast<int>(shape[2]);
  const int nc = std::max(1, impl_->nc);
  const int extras = channels - 4 - nc;
  if (extras < 0) {
    RCLCPP_WARN(logger, "Unexpected TensorRT output shape for YOLO parser");
    return {};
  }

  for (int i = 0; i < boxes; ++i) {
    const auto at = [&](int channel) -> float {
        return impl_->host_output[static_cast<std::size_t>(channel * boxes + i)];
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
    if (best_conf < conf_threshold_) {
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

  const auto kept = rotated_nms(std::move(parsed), kNmsIouThreshold, 300);
  std::vector<Detection> detections;
  detections.reserve(kept.size());
  for (const auto & candidate : kept) {
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
    const int x1 = static_cast<int>(std::floor(clamp_value(min_x, 0.0F, static_cast<float>(impl_->img_w - 1))));
    const int y1 = static_cast<int>(std::floor(clamp_value(min_y, 0.0F, static_cast<float>(impl_->img_h - 1))));
    const int x2 = static_cast<int>(std::ceil(clamp_value(max_x, 0.0F, static_cast<float>(impl_->img_w - 1))));
    const int y2 = static_cast<int>(std::ceil(clamp_value(max_y, 0.0F, static_cast<float>(impl_->img_h - 1))));

    Detection detection{};
    detection.bbox_model = cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
    detection.confidence = candidate.confidence;
    detection.class_id = candidate.class_id;
    if (impl_->task_obb) {
      detection.obb_angle = normalized_angle;
    }
    detections.push_back(detection);
  }
  return detections;
#endif
}

const std::string & DeepStreamDetector::backend_name() const
{
  return backend_name_;
}

}  // namespace yolov8_ros_cpp
