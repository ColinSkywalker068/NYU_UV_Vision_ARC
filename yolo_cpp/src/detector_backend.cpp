/// @file detector_backend.cpp
/// @brief Implements native TensorRT detector loading, preprocessing,
/// inference, and fallback.

#include "yolo_cpp/detector_backend.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <rclcpp/logging.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "yolo_cpp/detection_postprocess.hpp"

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

#ifdef HAVE_YOLO_CUDA_BAYER_RESIZE
#include "cuda_bayer_resize.hpp"
#endif

namespace yolo_cpp
{

namespace
{

/// Clamp a value into an inclusive range.
template <typename T>
T clamp_value(T value, T lo, T hi)
{
  return std::max(lo, std::min(value, hi));
}

enum BayerPattern
{
  kBayerRggb = 0,
  kBayerBggr = 1,
  kBayerGrbg = 2,
  kBayerGbrg = 3,
};

bool is_bayer_format(PixelFormat format)
{
  return format == PixelFormat::kBayerRggb8 || format == PixelFormat::kBayerBggr8
         || format == PixelFormat::kBayerGrbg8 || format == PixelFormat::kBayerGbrg8;
}

int bayer_pattern_from_pixel_format(PixelFormat format)
{
  switch (format) {
    case PixelFormat::kBayerBggr8:
      return kBayerBggr;
    case PixelFormat::kBayerGrbg8:
      return kBayerGrbg;
    case PixelFormat::kBayerGbrg8:
      return kBayerGbrg;
    case PixelFormat::kBayerRggb8:
    default:
      return kBayerRggb;
  }
}

/// Convert float32 to IEEE 754 half bits for FP16 input buffers.
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

  return static_cast<std::uint16_t>(
    sign | (static_cast<std::uint32_t>(exponent) << 10) | (mantissa >> 13));
}

/// Convert IEEE 754 half bits back to float32.
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
      bits = sign | (static_cast<std::uint32_t>(exp + 127) << 23) | (mant << 13);
    }
  } else if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
  }

  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

void fill_planes(float* host_input, int plane, float value)
{
  for (int c = 0; c < 3; ++c) {
    std::fill(host_input + (c * plane), host_input + ((c + 1) * plane), value);
  }
}

void fill_planes(std::uint16_t* host_input, int plane, std::uint16_t value)
{
  for (int c = 0; c < 3; ++c) {
    std::fill(host_input + (c * plane), host_input + ((c + 1) * plane), value);
  }
}

template <typename OutputT, typename ConvertT>
void pack_resized_bgr_to_chw_rgb(const cv::Mat& resized, OutputT* host_input, int dst_w,
  const LetterboxMetadata& metadata, ConvertT convert)
{
  const int plane = metadata.dst_w * metadata.dst_h;
  for (int y = 0; y < resized.rows; ++y) {
    const auto* row = resized.ptr<cv::Vec3b>(y);
    const int out_y = metadata.pad_y + y;
    for (int x = 0; x < resized.cols; ++x) {
      const int out_x = metadata.pad_x + x;
      const int offset = (out_y * dst_w) + out_x;
      const auto& bgr = row[x];
      host_input[offset] = convert(bgr[2]);
      host_input[plane + offset] = convert(bgr[1]);
      host_input[(2 * plane) + offset] = convert(bgr[0]);
    }
  }
}

/// Letterbox, normalize, and pack a BGR frame into a TensorRT FP32 input
/// buffer.
void preprocess_bgr_frame_float(const FramePacket& packet, float* host_input, int dst_w, int dst_h,
  LetterboxMetadata& metadata, cv::Mat& resized)
{
  if (packet.frame.empty() || packet.frame.type() != CV_8UC3) {
    throw std::runtime_error("Expected non-empty BGR8 frame for TensorRT preprocessing");
  }

  metadata = make_letterbox_metadata(packet.width(), packet.height(), dst_w, dst_h);
  const int resized_w = static_cast<int>(packet.width() * metadata.scale);
  const int resized_h = static_cast<int>(packet.height() * metadata.scale);
  cv::resize(packet.frame, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

  const int plane = dst_w * dst_h;
  fill_planes(host_input, plane, 114.0F / 255.0F);
  pack_resized_bgr_to_chw_rgb(resized, host_input, dst_w, metadata,
    [](std::uint8_t value) { return static_cast<float>(value) / 255.0F; });
}

/// Letterbox, normalize, and pack a BGR frame directly into a TensorRT FP16
/// input buffer.
void preprocess_bgr_frame_half(const FramePacket& packet, std::uint16_t* host_input, int dst_w,
  int dst_h, LetterboxMetadata& metadata, cv::Mat& resized)
{
  if (packet.frame.empty() || packet.frame.type() != CV_8UC3) {
    throw std::runtime_error("Expected non-empty BGR8 frame for TensorRT preprocessing");
  }

  metadata = make_letterbox_metadata(packet.width(), packet.height(), dst_w, dst_h);
  const int resized_w = static_cast<int>(packet.width() * metadata.scale);
  const int resized_h = static_cast<int>(packet.height() * metadata.scale);
  cv::resize(packet.frame, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

  const int plane = dst_w * dst_h;
  fill_planes(host_input, plane, float_to_half_bits(114.0F / 255.0F));
  pack_resized_bgr_to_chw_rgb(resized, host_input, dst_w, metadata,
    [](std::uint8_t value) { return float_to_half_bits(static_cast<float>(value) / 255.0F); });
}

#ifdef HAVE_TENSORRT
/// Deleter for TensorRT objects that are released with delete in this API
/// version.
template <typename T>
struct TrtDeleter
{
  void operator()(T* ptr) const
  {
    delete ptr;
  }
};

/// Bridges TensorRT logger callbacks into ROS logging.
class TrtLogger : public nvinfer1::ILogger
{
public:
  explicit TrtLogger(rclcpp::Logger logger) : logger_(logger) {}

  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      RCLCPP_ERROR(logger_, "TensorRT: %s", msg);
    } else if (severity == Severity::kWARNING) {
      RCLCPP_WARN(logger_, "TensorRT: %s", msg);
    }
  }

private:
  rclcpp::Logger logger_;
};

/// Collect TensorRT's optimized-layer timings and persist each sampled inference as CSV.
/// TensorRT can fuse original YOLO operations, so layer names represent engine blocks.
class TensorRtLayerProfiler final : public nvinfer1::IProfiler
{
public:
  explicit TensorRtLayerProfiler(std::string path) : path_(std::move(path))
  {
    const auto parent = std::filesystem::path(path_).parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    stream_.open(path_, std::ios::out | std::ios::app);
    if (!stream_.is_open()) {
      throw std::runtime_error("Failed to open TensorRT layer profile log: " + path_);
    }
    if (stream_.tellp() == std::streampos(0)) {
      stream_ << "timestamp,frame,layer,latency_ms\n";
      stream_.flush();
    }
  }

  void begin_frame(std::uint64_t frame)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    frame_ = frame;
    samples_.clear();
  }

  void reportLayerTime(const char* layer_name, float ms) noexcept override
  {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      samples_.emplace_back(layer_name == nullptr ? "<unnamed>" : layer_name, ms);
    } catch (...) {
      // TensorRT calls this callback from its execution path; profiling must never stop aiming.
    }
  }

  std::size_t write_frame(const std::string& timestamp)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& sample : samples_) {
      stream_ << timestamp << ',' << frame_ << ',' << csv_escape(sample.first) << ','
              << std::fixed << std::setprecision(6) << sample.second << '\n';
    }
    stream_.flush();
    return samples_.size();
  }

private:
  static std::string csv_escape(const std::string& value)
  {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
      return value;
    }
    std::string escaped{"\""};
    for (const char ch : value) {
      if (ch == '\"') {
        escaped += "\"\"";
      } else {
        escaped += ch;
      }
    }
    return escaped + "\"";
  }

  std::string path_;
  std::ofstream stream_;
  std::mutex mutex_;
  std::uint64_t frame_{0};
  std::vector<std::pair<std::string, float>> samples_;
};

/// Throw a runtime_error when a CUDA API call fails.
void throw_if_cuda_failed(cudaError_t status, const char* what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}
#endif

}  // namespace

/// Optional TensorRT implementation state hidden from builds without TensorRT
/// headers.
struct NativeTensorRTBackend::Impl
{
#ifdef HAVE_TENSORRT
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
  float* host_input{nullptr};
  float* host_output{nullptr};
  std::uint16_t* host_input_half{nullptr};
  std::uint16_t* host_output_half{nullptr};
  std::vector<float> parsed_output;
  std::vector<void*> bindings;
  cv::Mat resized_input;
  void* cuda_bayer_context{nullptr};
  std::size_t input_volume{0};
  std::size_t output_volume{0};
  void* device_input{nullptr};
  void* device_output{nullptr};
  nvinfer1::DataType input_dtype{nvinfer1::DataType::kFLOAT};
  nvinfer1::DataType output_dtype{nvinfer1::DataType::kFLOAT};
  int img_w{640};
  int img_h{640};
  int nc{1};
  bool task_obb{false};
  bool end2end{false};
  bool logged_output_sample{false};
  std::optional<bool> export_includes_nms;
  std::unique_ptr<TensorRtLayerProfiler> layer_profiler;
  std::uint64_t inference_frame_count{0};

  ~Impl()
  {
    if (device_input != nullptr) {
      cudaFree(device_input);
    }
    if (device_output != nullptr) {
      cudaFree(device_output);
    }
    if (host_input != nullptr) {
      cudaFreeHost(host_input);
    }
    if (host_output != nullptr) {
      cudaFreeHost(host_output);
    }
    if (host_input_half != nullptr) {
      cudaFreeHost(host_input_half);
    }
    if (host_output_half != nullptr) {
      cudaFreeHost(host_output_half);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
#ifdef HAVE_YOLO_CUDA_BAYER_RESIZE
    if (cuda_bayer_context != nullptr) {
      yolo_cuda_bayer_resize_destroy(cuda_bayer_context);
    }
#endif
  }
#endif
};

namespace
{

#ifdef HAVE_TENSORRT
/// Convert TensorRT dimensions into a standard vector.
std::vector<int64_t> dims_to_vector(const nvinfer1::Dims& dims)
{
  std::vector<int64_t> result;
  result.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; ++i) {
    result.push_back(static_cast<int64_t>(dims.d[i]));
  }
  return result;
}

std::string format_shape(const std::vector<int64_t>& shape)
{
  std::ostringstream output;
  output << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0U) {
      output << ',';
    }
    output << shape[i];
  }
  output << ']';
  return output.str();
}

std::string format_output_sample(
  const std::vector<float>& values, const std::vector<int64_t>& shape)
{
  if (shape.size() != 3U || shape[0] != 1 || shape[1] <= 0 || shape[2] <= 0) {
    return "unsupported-shape";
  }

  const int dim1 = static_cast<int>(shape[1]);
  const int dim2 = static_cast<int>(shape[2]);
  const bool attributes_last = dim2 <= 16;
  const int attribute_count = attributes_last ? dim2 : dim1;
  const int candidate_count = attributes_last ? dim1 : dim2;
  const int rows_to_log = std::min(candidate_count, 6);
  if (attribute_count > 16) {
    return "unable-to-identify-attribute-axis";
  }

  std::ostringstream output;
  output << (attributes_last ? "rows-last" : "channels-first") << " candidates=" << candidate_count
         << " attrs=" << attribute_count;
  output << std::fixed << std::setprecision(5);
  for (int row = 0; row < rows_to_log; ++row) {
    output << " row" << row << "=[";
    for (int attr = 0; attr < attribute_count; ++attr) {
      if (attr > 0) {
        output << ',';
      }
      const std::size_t index = attributes_last
                                  ? static_cast<std::size_t>((row * attribute_count) + attr)
                                  : static_cast<std::size_t>((attr * candidate_count) + row);
      output << (index < values.size() ? values[index] : std::numeric_limits<float>::quiet_NaN());
    }
    output << ']';
  }
  return output.str();
}

/// Return the element volume for a fully specified tensor shape.
int64_t volume_of(const std::vector<int64_t>& shape)
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

/// Resolve dynamic batch/spatial dimensions for this package's fixed-size
/// inference path.
std::vector<int64_t> resolve_runtime_shape(const std::vector<int64_t>& shape, int img_h, int img_w)
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
#endif

}  // namespace

NativeTensorRTBackend::NativeTensorRTBackend(
  std::string model_path, std::string task, std::string model_type, bool half, float conf_threshold,
  bool layer_profile_enabled, std::string layer_profile_path, int layer_profile_interval)
    : model_path_(std::move(model_path)),
      task_(std::move(task)),
      model_type_(std::move(model_type)),
      half_(half),
      conf_threshold_(conf_threshold),
      layer_profile_enabled_(layer_profile_enabled),
      layer_profile_path_(std::move(layer_profile_path)),
      layer_profile_interval_(std::max(1, layer_profile_interval))
{
}

NativeTensorRTBackend::~NativeTensorRTBackend() = default;

void NativeTensorRTBackend::ensure_initialized(const rclcpp::Logger& logger)
{
  // Metadata is read first so class count, task, and NMS behavior are known
  // before allocation.
  if (initialized_) {
    return;
  }

  class_names_.clear();
  try {
    const auto metadata = read_engine_metadata_from_file(model_path_);
    class_names_ = metadata.class_names;
    if (metadata.task.has_value()) {
      task_ = *metadata.task;
    }
    engine_version_ = metadata.version.value_or("");
    end2end_ = metadata.end2end;
    int8_export_ = metadata.int8_export;
    export_includes_nms_ = metadata.export_includes_nms;
    if (class_names_.empty() && task_ == "obb") {
      class_names_.emplace(0, "blue");
      class_names_.emplace(1, "red");
      RCLCPP_WARN(logger, "Engine metadata has no class names; assuming project OBB classes "
                          "{0: blue, 1: red}");
    } else if (class_names_.empty()) {
      class_names_.emplace(0, "plate");
    }
  } catch (const std::exception& exc) {
    RCLCPP_WARN(logger, "Failed reading engine metadata: %s", exc.what());
  }

#ifdef HAVE_TENSORRT
  impl_ = std::make_unique<Impl>(logger);
  impl_->task_obb = (task_ == "obb");
  impl_->end2end = end2end_;
  impl_->export_includes_nms = export_includes_nms_;
  impl_->nc = class_names_.empty() ? 1 : static_cast<int>(class_names_.size());

  const std::string engine_bytes = read_engine_plan_from_file(model_path_);
  impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->trt_logger));
  if (!impl_->runtime) {
    throw std::runtime_error("Failed to create TensorRT runtime");
  }
  impl_->engine.reset(
    impl_->runtime->deserializeCudaEngine(engine_bytes.data(), engine_bytes.size()));
  if (!impl_->engine) {
    throw std::runtime_error("Failed to deserialize TensorRT engine");
  }
  impl_->context.reset(impl_->engine->createExecutionContext());
  if (!impl_->context) {
    throw std::runtime_error("Failed to create TensorRT execution context");
  }
  throw_if_cuda_failed(cudaStreamCreate(&impl_->stream), "cudaStreamCreate failed");
  if (layer_profile_enabled_) {
    if (layer_profile_path_.empty()) {
      throw std::runtime_error("TensorRT layer profiling is enabled but no log path was provided");
    }
    impl_->layer_profiler = std::make_unique<TensorRtLayerProfiler>(layer_profile_path_);
    RCLCPP_INFO(logger,
      "TensorRT per-layer profiling enabled: path=%s interval=%d frame(s). "
      "Timings are optimized engine layers/blocks in milliseconds.",
      layer_profile_path_.c_str(), layer_profile_interval_);
  }

  const int nb = impl_->engine->getNbBindings();
  for (int i = 0; i < nb; ++i) {
    const char* name = impl_->engine->getBindingName(i);
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

  impl_->runtime_input_shape =
    resolve_runtime_shape(impl_->input_shape, impl_->img_h, impl_->img_w);
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

  impl_->runtime_output_shape =
    dims_to_vector(impl_->context->getBindingDimensions(impl_->output_index));
  if (volume_of(impl_->runtime_output_shape) <= 0) {
    throw std::runtime_error("Unable to resolve TensorRT output dimensions");
  }

  RCLCPP_INFO(logger,
    "TensorRT bindings: input=%s declared=%s runtime=%s output=%s "
    "declared=%s runtime=%s classes=%d",
    impl_->input_name.c_str(), format_shape(impl_->input_shape).c_str(),
    format_shape(impl_->runtime_input_shape).c_str(), impl_->output_name.c_str(),
    format_shape(impl_->output_shape).c_str(), format_shape(impl_->runtime_output_shape).c_str(),
    impl_->nc);

  if (impl_->input_dtype != nvinfer1::DataType::kFLOAT
      && impl_->input_dtype != nvinfer1::DataType::kHALF)
  {
    throw std::runtime_error("TensorRT engine uses unsupported input I/O datatype. "
                             "Ultralytics INT8 exports are supported when bindings stay FP32/FP16, "
                             "but true INT8 input bindings are not yet handled by this runtime.");
  }
  if (impl_->output_dtype != nvinfer1::DataType::kFLOAT
      && impl_->output_dtype != nvinfer1::DataType::kHALF)
  {
    throw std::runtime_error("TensorRT engine uses unsupported output I/O datatype. "
                             "Ultralytics INT8 exports are supported when bindings stay FP32/FP16, "
                             "but true INT8 output bindings are not yet handled by this runtime.");
  }

  impl_->input_volume = static_cast<std::size_t>(volume_of(impl_->runtime_input_shape));
  impl_->output_volume = static_cast<std::size_t>(volume_of(impl_->runtime_output_shape));
  impl_->parsed_output.resize(impl_->output_volume);

  if (impl_->input_dtype == nvinfer1::DataType::kFLOAT) {
    throw_if_cuda_failed(cudaHostAlloc(reinterpret_cast<void**>(&impl_->host_input),
                           impl_->input_volume * sizeof(float), cudaHostAllocPortable),
      "cudaHostAlloc input buffer failed");
  }
  throw_if_cuda_failed(cudaHostAlloc(reinterpret_cast<void**>(&impl_->host_output),
                         impl_->output_volume * sizeof(float), cudaHostAllocPortable),
    "cudaHostAlloc output buffer failed");

  if (impl_->input_dtype == nvinfer1::DataType::kHALF) {
    throw_if_cuda_failed(cudaHostAlloc(reinterpret_cast<void**>(&impl_->host_input_half),
                           impl_->input_volume * sizeof(std::uint16_t), cudaHostAllocPortable),
      "cudaHostAlloc FP16 input buffer failed");
  }
  if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    throw_if_cuda_failed(cudaHostAlloc(reinterpret_cast<void**>(&impl_->host_output_half),
                           impl_->output_volume * sizeof(std::uint16_t), cudaHostAllocPortable),
      "cudaHostAlloc FP16 output buffer failed");
  }

  throw_if_cuda_failed(
    cudaMalloc(&impl_->device_input,
      impl_->input_volume
        * ((impl_->input_dtype == nvinfer1::DataType::kHALF) ? sizeof(std::uint16_t)
                                                             : sizeof(float))),
    "cudaMalloc input buffer failed");
  throw_if_cuda_failed(
    cudaMalloc(&impl_->device_output,
      impl_->output_volume
        * ((impl_->output_dtype == nvinfer1::DataType::kHALF) ? sizeof(std::uint16_t)
                                                              : sizeof(float))),
    "cudaMalloc output buffer failed");
  impl_->bindings.assign(static_cast<std::size_t>(impl_->engine->getNbBindings()), nullptr);
  impl_->bindings[static_cast<std::size_t>(impl_->input_index)] = impl_->device_input;
  impl_->bindings[static_cast<std::size_t>(impl_->output_index)] = impl_->device_output;

  backend_name_ = "native_tensorrt";
  RCLCPP_INFO(logger,
    "TensorRT engine ready: task=%s version=%s export_int8=%s export_nms=%s "
    "end2end=%s input_dtype=%d output_dtype=%d",
    task_.c_str(), engine_version_.empty() ? "unknown" : engine_version_.c_str(),
    int8_export_ ? "true" : "false",
    export_includes_nms_.has_value() ? (*export_includes_nms_ ? "true" : "false") : "unknown",
    end2end_ ? "true" : "false", static_cast<int>(impl_->input_dtype),
    static_cast<int>(impl_->output_dtype));
#else
  (void)warned_unavailable_;
  throw std::runtime_error("Native TensorRT backend requested, but this "
                           "yolo_cpp build did not find TensorRT/CUDA "
                           "headers and libraries. Rebuild after CMake reports "
                           "'TensorRT found. Native TensorRT "
                           "backend enabled.'");
#endif

  initialized_ = true;
}

void NativeTensorRTBackend::prepare(const rclcpp::Logger& logger)
{
  ensure_initialized(logger);
}

DetectionBatch NativeTensorRTBackend::infer(
  const FramePacket& packet, const rclcpp::Logger& logger, DetectorStageTimings* timings)
{
  ensure_initialized(logger);

  DetectionBatch batch{};
  if (timings != nullptr) {
    *timings = DetectorStageTimings{};
  }
  if (packet.empty()) {
    return batch;
  }

#ifndef HAVE_TENSORRT
  if (!warned_unavailable_) {
    RCLCPP_WARN(logger, "TensorRT inference unavailable in this build; "
                        "detector returning no detections.");
    warned_unavailable_ = true;
  }
  batch.letterbox =
    make_letterbox_metadata(packet.width(), packet.height(), packet.width(), packet.height());
  return batch;
#else
  if (!impl_) {
    return batch;
  }

  const auto preprocess_start = std::chrono::steady_clock::now();
  if (is_bayer_format(packet.pixel_format)) {
    if (packet.frame.empty() || packet.frame.type() != CV_8UC1) {
      throw std::runtime_error("Expected non-empty Bayer8 frame for TensorRT CUDA preprocessing");
    }
    batch.letterbox =
      make_letterbox_metadata(packet.width(), packet.height(), impl_->img_w, impl_->img_h);
#ifdef HAVE_YOLO_CUDA_BAYER_RESIZE
    if (impl_->cuda_bayer_context == nullptr) {
      impl_->cuda_bayer_context = yolo_cuda_bayer_resize_create();
      if (impl_->cuda_bayer_context == nullptr) {
        throw std::runtime_error("Failed to create native TensorRT CUDA Bayer preprocess context");
      }
    }
    std::array<char, 256> error_message{};
    const int output_data_type = impl_->input_dtype == nvinfer1::DataType::kHALF ? 1 : 0;
    const bool ok = yolo_cuda_bayer_resize_chw(impl_->cuda_bayer_context, packet.frame.ptr(),
      packet.width(), packet.height(), static_cast<int>(packet.frame.step[0]), impl_->device_input,
      impl_->img_w, impl_->img_h, bayer_pattern_from_pixel_format(packet.pixel_format),
      packet.flip_180 ? 2 : 0, output_data_type, error_message.data(), error_message.size());
    if (!ok) {
      throw std::runtime_error(
        std::string("Native TensorRT CUDA Bayer preprocessing failed: ") + error_message.data());
    }
#else
    throw std::runtime_error(
      "Native TensorRT received raw Bayer input, but this build does not include "
      "HAVE_YOLO_CUDA_BAYER_RESIZE");
#endif
  } else {
    const void* input_ptr = impl_->host_input;
    std::size_t input_bytes = impl_->input_volume * sizeof(float);
    if (impl_->input_dtype == nvinfer1::DataType::kHALF) {
      preprocess_bgr_frame_half(packet, impl_->host_input_half, impl_->img_w, impl_->img_h,
        batch.letterbox, impl_->resized_input);
      input_ptr = impl_->host_input_half;
      input_bytes = impl_->input_volume * sizeof(std::uint16_t);
    } else {
      preprocess_bgr_frame_float(packet, impl_->host_input, impl_->img_w, impl_->img_h,
        batch.letterbox, impl_->resized_input);
    }
    const auto upload_start = std::chrono::steady_clock::now();
    throw_if_cuda_failed(cudaMemcpyAsync(impl_->device_input, input_ptr, input_bytes,
                           cudaMemcpyHostToDevice, impl_->stream),
      "TensorRT input upload failed");
    const auto upload_done = std::chrono::steady_clock::now();
    if (timings != nullptr) {
      timings->upload_ms =
        std::chrono::duration<double, std::milli>(upload_done - upload_start).count();
    }
  }
  const auto preprocess_done = std::chrono::steady_clock::now();

  const auto inference_start = std::chrono::steady_clock::now();
  const std::uint64_t inference_frame = ++impl_->inference_frame_count;
  const bool profile_this_frame = impl_->layer_profiler != nullptr
                                  && ((inference_frame - 1U)
                                      % static_cast<std::uint64_t>(layer_profile_interval_) == 0U);
  if (profile_this_frame) {
    impl_->layer_profiler->begin_frame(inference_frame);
  }
  impl_->context->setProfiler(profile_this_frame ? impl_->layer_profiler.get() : nullptr);
  if (!impl_->context->enqueueV2(impl_->bindings.data(), impl_->stream, nullptr)) {
    throw std::runtime_error("TensorRT enqueue failed");
  }
  const auto inference_done = std::chrono::steady_clock::now();

  void* output_ptr = impl_->host_output;
  std::size_t output_bytes = impl_->output_volume * sizeof(float);
  if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    output_ptr = impl_->host_output_half;
    output_bytes = impl_->output_volume * sizeof(std::uint16_t);
  }
  const auto download_start = std::chrono::steady_clock::now();
  throw_if_cuda_failed(cudaMemcpyAsync(output_ptr, impl_->device_output, output_bytes,
                         cudaMemcpyDeviceToHost, impl_->stream),
    "TensorRT output download failed");
  throw_if_cuda_failed(cudaStreamSynchronize(impl_->stream), "TensorRT stream sync failed");
  const auto download_done = std::chrono::steady_clock::now();
  if (profile_this_frame) {
    const auto layer_count = impl_->layer_profiler->write_frame(
      std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()));
    if (layer_count == 0U) {
      RCLCPP_WARN(logger, "TensorRT layer profiler reported no layers for frame %llu",
        static_cast<unsigned long long>(inference_frame));
    }
  }

  if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    for (std::size_t i = 0; i < impl_->output_volume; ++i) {
      impl_->parsed_output[i] = half_bits_to_float(impl_->host_output_half[i]);
    }
  } else {
    std::memcpy(
      impl_->parsed_output.data(), impl_->host_output, impl_->output_volume * sizeof(float));
  }

  const bool log_output_sample = !impl_->logged_output_sample;
  if (log_output_sample) {
    RCLCPP_INFO(logger, "TensorRT first output sample: shape=%s %s",
      format_shape(impl_->runtime_output_shape).c_str(),
      format_output_sample(impl_->parsed_output, impl_->runtime_output_shape).c_str());
  }

  const auto parse_start = std::chrono::steady_clock::now();
  batch.detections = parse_yolo_output_tensor(impl_->parsed_output.data(),
    impl_->runtime_output_shape, impl_->img_w, impl_->img_h, impl_->nc, conf_threshold_,
    impl_->task_obb, impl_->end2end, impl_->export_includes_nms, &logger);
  const auto parse_done = std::chrono::steady_clock::now();
  if (log_output_sample) {
    impl_->logged_output_sample = true;
    RCLCPP_INFO(logger,
      "TensorRT first parse result: detections=%zu conf_threshold=%.3f "
      "task_obb=%s end2end_metadata=%s export_nms=%s",
      batch.detections.size(), conf_threshold_, impl_->task_obb ? "true" : "false",
      impl_->end2end ? "true" : "false",
      impl_->export_includes_nms.has_value() ? (*impl_->export_includes_nms ? "true" : "false")
                                             : "unknown");
  }

  if (timings != nullptr) {
    timings->preprocess_ms =
      std::chrono::duration<double, std::milli>(preprocess_done - preprocess_start).count();
    timings->inference_ms =
      std::chrono::duration<double, std::milli>(inference_done - inference_start).count();
    timings->download_ms =
      std::chrono::duration<double, std::milli>(download_done - download_start).count();
    timings->parse_ms = std::chrono::duration<double, std::milli>(parse_done - parse_start).count();
  }

  return batch;
#endif
}

const std::string& NativeTensorRTBackend::backend_name() const
{
  return backend_name_;
}

DeepStreamBackend::DeepStreamBackend(
  std::string model_path, std::string task, std::string model_type, bool half, float conf_threshold,
  bool layer_profile_enabled, std::string layer_profile_path, int layer_profile_interval)
    : fallback_(std::move(model_path), std::move(task), std::move(model_type), half, conf_threshold,
        layer_profile_enabled, std::move(layer_profile_path), layer_profile_interval)
{
}

DetectionBatch DeepStreamBackend::infer(
  const FramePacket& packet, const rclcpp::Logger& logger, DetectorStageTimings* timings)
{
  if (!warned_) {
    RCLCPP_WARN(logger, "DeepStream backend is not implemented yet in "
                        "yolo_cpp; falling back to native TensorRT.");
    warned_ = true;
  }
  return fallback_.infer(packet, logger, timings);
}

void DeepStreamBackend::prepare(const rclcpp::Logger& logger)
{
  fallback_.prepare(logger);
}

const std::string& DeepStreamBackend::backend_name() const
{
  return backend_name_;
}

std::unique_ptr<DetectorBackend> create_detector_backend(const std::string& backend_kind,
  const std::string& model_path, const std::string& task, const std::string& model_type, bool half,
  float conf_threshold, bool layer_profile_enabled, const std::string& layer_profile_path,
  int layer_profile_interval)
{
  if (backend_kind == "deepstream") {
    return std::make_unique<DeepStreamBackend>(model_path, task, model_type, half, conf_threshold,
      layer_profile_enabled, layer_profile_path, layer_profile_interval);
  }
  return std::make_unique<NativeTensorRTBackend>(
    model_path, task, model_type, half, conf_threshold, layer_profile_enabled, layer_profile_path,
    layer_profile_interval);
}

}  // namespace yolo_cpp
