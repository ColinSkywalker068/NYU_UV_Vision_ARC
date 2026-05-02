/// @file detector_backend.cpp
/// @brief Implements native TensorRT detector loading, preprocessing, inference, and fallback.

#include "yolo_cpp/detector_backend.hpp"
#include "yolo_cpp/detection_postprocess.hpp"

#include <opencv2/imgproc.hpp>
#include <rclcpp/logging.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

namespace yolo_cpp
{

namespace
{

/// Clamp a value into an inclusive range.
template<typename T>
T clamp_value(T value, T lo, T hi)
{
  return std::max(lo, std::min(value, hi));
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

/// Letterbox, normalize, and pack a BGR frame into a TensorRT input buffer.
void preprocess_bgr_frame(
  const FramePacket & packet, float * host_input, int dst_w, int dst_h,
  LetterboxMetadata & metadata)
{
  if (packet.frame.empty() || packet.frame.type() != CV_8UC3) {
    throw std::runtime_error("Expected non-empty BGR8 frame for TensorRT preprocessing");
  }

  metadata = make_letterbox_metadata(packet.width(), packet.height(), dst_w, dst_h);
  const int resized_w = static_cast<int>(packet.width() * metadata.scale);
  const int resized_h = static_cast<int>(packet.height() * metadata.scale);
  cv::Mat resized;
  cv::resize(packet.frame, resized, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat padded(dst_h, dst_w, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(metadata.pad_x, metadata.pad_y, resized_w, resized_h)));

  cv::Mat rgb;
  cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

  cv::Mat rgb_float;
  rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

  std::array<cv::Mat, 3> channels;
  cv::split(rgb_float, channels.data());

  const int plane = dst_w * dst_h;
  for (int c = 0; c < 3; ++c) {
    std::memcpy(
      host_input + (c * plane),
      channels[static_cast<std::size_t>(c)].ptr<float>(),
      static_cast<std::size_t>(plane) * sizeof(float));
  }
}

#ifdef HAVE_TENSORRT
/// Deleter for TensorRT objects that are released with delete in this API version.
template<typename T>
struct TrtDeleter
{
  void operator()(T * ptr) const
  {
    delete ptr;
  }
};

/// Bridges TensorRT logger callbacks into ROS logging.
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
    }
  }

private:
  rclcpp::Logger logger_;
};

/// Throw a runtime_error when a CUDA API call fails.
void throw_if_cuda_failed(cudaError_t status, const char * what)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}
#endif

}  // namespace

/// Optional TensorRT implementation state hidden from builds without TensorRT headers.
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
  float * host_input{nullptr};
  float * host_output{nullptr};
  std::uint16_t * host_input_half{nullptr};
  std::uint16_t * host_output_half{nullptr};
  std::vector<float> parsed_output;
  std::size_t input_volume{0};
  std::size_t output_volume{0};
  void * device_input{nullptr};
  void * device_output{nullptr};
  nvinfer1::DataType input_dtype{nvinfer1::DataType::kFLOAT};
  nvinfer1::DataType output_dtype{nvinfer1::DataType::kFLOAT};
  int img_w{640};
  int img_h{640};
  int nc{1};
  bool task_obb{false};
  bool end2end{false};
  std::optional<bool> export_includes_nms;

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
  }
#endif
};

namespace
{

#ifdef HAVE_TENSORRT
/// Convert TensorRT dimensions into a standard vector.
std::vector<int64_t> dims_to_vector(const nvinfer1::Dims & dims)
{
  std::vector<int64_t> result;
  result.reserve(dims.nbDims);
  for (int i = 0; i < dims.nbDims; ++i) {
    result.push_back(static_cast<int64_t>(dims.d[i]));
  }
  return result;
}

/// Return the element volume for a fully specified tensor shape.
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

/// Resolve dynamic batch/spatial dimensions for this package's fixed-size inference path.
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
#endif

}  // namespace

NativeTensorRTBackend::NativeTensorRTBackend(
  std::string model_path, std::string task, std::string model_type, bool half,
  float conf_threshold)
: model_path_(std::move(model_path)),
  task_(std::move(task)),
  model_type_(std::move(model_type)),
  half_(half),
  conf_threshold_(conf_threshold)
{
}

NativeTensorRTBackend::~NativeTensorRTBackend() = default;

void NativeTensorRTBackend::ensure_initialized(const rclcpp::Logger & logger)
{
  // Metadata is read first so class count, task, and NMS behavior are known before allocation.
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
    if (class_names_.empty()) {
      class_names_.emplace(0, "plate");
    }
  } catch (const std::exception & exc) {
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

  impl_->runtime_output_shape = dims_to_vector(
    impl_->context->getBindingDimensions(impl_->output_index));
  if (volume_of(impl_->runtime_output_shape) <= 0) {
    throw std::runtime_error("Unable to resolve TensorRT output dimensions");
  }

  if (
    impl_->input_dtype != nvinfer1::DataType::kFLOAT &&
    impl_->input_dtype != nvinfer1::DataType::kHALF)
  {
    throw std::runtime_error(
            "TensorRT engine uses unsupported input I/O datatype. "
            "Ultralytics INT8 exports are supported when bindings stay FP32/FP16, "
            "but true INT8 input bindings are not yet handled by this runtime.");
  }
  if (
    impl_->output_dtype != nvinfer1::DataType::kFLOAT &&
    impl_->output_dtype != nvinfer1::DataType::kHALF)
  {
    throw std::runtime_error(
            "TensorRT engine uses unsupported output I/O datatype. "
            "Ultralytics INT8 exports are supported when bindings stay FP32/FP16, "
            "but true INT8 output bindings are not yet handled by this runtime.");
  }

  impl_->input_volume = static_cast<std::size_t>(volume_of(impl_->runtime_input_shape));
  impl_->output_volume = static_cast<std::size_t>(volume_of(impl_->runtime_output_shape));
  impl_->parsed_output.resize(impl_->output_volume);

  throw_if_cuda_failed(
    cudaHostAlloc(
      reinterpret_cast<void **>(&impl_->host_input), impl_->input_volume * sizeof(float),
      cudaHostAllocPortable),
    "cudaHostAlloc input buffer failed");
  throw_if_cuda_failed(
    cudaHostAlloc(
      reinterpret_cast<void **>(&impl_->host_output), impl_->output_volume * sizeof(float),
      cudaHostAllocPortable),
    "cudaHostAlloc output buffer failed");

  if (impl_->input_dtype == nvinfer1::DataType::kHALF) {
    throw_if_cuda_failed(
      cudaHostAlloc(
        reinterpret_cast<void **>(&impl_->host_input_half),
        impl_->input_volume * sizeof(std::uint16_t),
        cudaHostAllocPortable),
      "cudaHostAlloc FP16 input buffer failed");
  }
  if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    throw_if_cuda_failed(
      cudaHostAlloc(
        reinterpret_cast<void **>(&impl_->host_output_half),
        impl_->output_volume * sizeof(std::uint16_t),
        cudaHostAllocPortable),
      "cudaHostAlloc FP16 output buffer failed");
  }

  throw_if_cuda_failed(
    cudaMalloc(
      &impl_->device_input,
      impl_->input_volume *
      ((impl_->input_dtype == nvinfer1::DataType::kHALF) ? sizeof(std::uint16_t) : sizeof(float))),
    "cudaMalloc input buffer failed");
  throw_if_cuda_failed(
    cudaMalloc(
      &impl_->device_output,
      impl_->output_volume *
      ((impl_->output_dtype == nvinfer1::DataType::kHALF) ? sizeof(std::uint16_t) : sizeof(float))),
    "cudaMalloc output buffer failed");

  backend_name_ = "native_tensorrt";
  RCLCPP_INFO(
    logger,
    "TensorRT engine ready: task=%s version=%s export_int8=%s export_nms=%s end2end=%s input_dtype=%d output_dtype=%d",
    task_.c_str(),
    engine_version_.empty() ? "unknown" : engine_version_.c_str(),
    int8_export_ ? "true" : "false",
    export_includes_nms_.has_value() ? (*export_includes_nms_ ? "true" : "false") : "unknown",
    end2end_ ? "true" : "false",
    static_cast<int>(impl_->input_dtype),
    static_cast<int>(impl_->output_dtype));
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

DetectionBatch NativeTensorRTBackend::infer(
  const FramePacket & packet, const rclcpp::Logger & logger, DetectorStageTimings * timings)
{
  ensure_initialized(logger);

  DetectionBatch batch{};
  if (packet.empty()) {
    return batch;
  }

#ifndef HAVE_TENSORRT
  if (!warned_unavailable_) {
    RCLCPP_WARN(
      logger, "TensorRT inference unavailable in this build; detector returning no detections.");
    warned_unavailable_ = true;
  }
  batch.letterbox = make_letterbox_metadata(packet.width(), packet.height(), packet.width(), packet.height());
  return batch;
#else
  if (!impl_) {
    return batch;
  }

  const auto preprocess_start = std::chrono::steady_clock::now();
  preprocess_bgr_frame(packet, impl_->host_input, impl_->img_w, impl_->img_h, batch.letterbox);
  const auto preprocess_done = std::chrono::steady_clock::now();

  const void * input_ptr = impl_->host_input;
  std::size_t input_bytes = impl_->input_volume * sizeof(float);
  if (impl_->input_dtype == nvinfer1::DataType::kHALF) {
    for (std::size_t i = 0; i < impl_->input_volume; ++i) {
      impl_->host_input_half[i] = float_to_half_bits(impl_->host_input[i]);
    }
    input_ptr = impl_->host_input_half;
    input_bytes = impl_->input_volume * sizeof(std::uint16_t);
  }
  const auto upload_start = std::chrono::steady_clock::now();
  throw_if_cuda_failed(
    cudaMemcpyAsync(
      impl_->device_input, input_ptr, input_bytes, cudaMemcpyHostToDevice, impl_->stream),
    "TensorRT input upload failed");
  const auto upload_done = std::chrono::steady_clock::now();

  std::vector<void *> bindings(static_cast<std::size_t>(impl_->engine->getNbBindings()), nullptr);
  bindings[static_cast<std::size_t>(impl_->input_index)] = impl_->device_input;
  bindings[static_cast<std::size_t>(impl_->output_index)] = impl_->device_output;
  const auto inference_start = std::chrono::steady_clock::now();
  if (!impl_->context->enqueueV2(bindings.data(), impl_->stream, nullptr)) {
    throw std::runtime_error("TensorRT enqueue failed");
  }
  const auto inference_done = std::chrono::steady_clock::now();

  void * output_ptr = impl_->host_output;
  std::size_t output_bytes = impl_->output_volume * sizeof(float);
  if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    output_ptr = impl_->host_output_half;
    output_bytes = impl_->output_volume * sizeof(std::uint16_t);
  }
  const auto download_start = std::chrono::steady_clock::now();
  throw_if_cuda_failed(
    cudaMemcpyAsync(
      output_ptr, impl_->device_output, output_bytes, cudaMemcpyDeviceToHost, impl_->stream),
    "TensorRT output download failed");
  throw_if_cuda_failed(cudaStreamSynchronize(impl_->stream), "TensorRT stream sync failed");
  const auto download_done = std::chrono::steady_clock::now();

  if (impl_->output_dtype == nvinfer1::DataType::kHALF) {
    for (std::size_t i = 0; i < impl_->output_volume; ++i) {
      impl_->parsed_output[i] = half_bits_to_float(impl_->host_output_half[i]);
    }
  } else {
    std::memcpy(
      impl_->parsed_output.data(), impl_->host_output, impl_->output_volume * sizeof(float));
  }

  const auto parse_start = std::chrono::steady_clock::now();
  batch.detections = parse_yolo_output_tensor(
    impl_->parsed_output.data(),
    impl_->runtime_output_shape,
    impl_->img_w,
    impl_->img_h,
    impl_->nc,
    conf_threshold_,
    impl_->task_obb,
    impl_->end2end,
    impl_->export_includes_nms,
    &logger);
  const auto parse_done = std::chrono::steady_clock::now();

  if (timings != nullptr) {
    timings->preprocess_ms = std::chrono::duration<double, std::milli>(
      preprocess_done - preprocess_start).count();
    timings->upload_ms = std::chrono::duration<double, std::milli>(
      upload_done - upload_start).count();
    timings->inference_ms = std::chrono::duration<double, std::milli>(
      inference_done - inference_start).count();
    timings->download_ms = std::chrono::duration<double, std::milli>(
      download_done - download_start).count();
    timings->parse_ms = std::chrono::duration<double, std::milli>(
      parse_done - parse_start).count();
  }

  return batch;
#endif
}

const std::string & NativeTensorRTBackend::backend_name() const
{
  return backend_name_;
}

DeepStreamBackend::DeepStreamBackend(
  std::string model_path, std::string task, std::string model_type, bool half,
  float conf_threshold)
: fallback_(
    std::move(model_path), std::move(task), std::move(model_type), half, conf_threshold)
{
}

DetectionBatch DeepStreamBackend::infer(
  const FramePacket & packet, const rclcpp::Logger & logger, DetectorStageTimings * timings)
{
  if (!warned_) {
    RCLCPP_WARN(
      logger,
      "DeepStream backend is not implemented yet in yolo_cpp; falling back to native TensorRT.");
    warned_ = true;
  }
  return fallback_.infer(packet, logger, timings);
}

const std::string & DeepStreamBackend::backend_name() const
{
  return backend_name_;
}

std::unique_ptr<DetectorBackend> create_detector_backend(
  const std::string & backend_kind, const std::string & model_path, const std::string & task,
  const std::string & model_type, bool half, float conf_threshold)
{
  if (backend_kind == "deepstream") {
    return std::make_unique<DeepStreamBackend>(
      model_path, task, model_type, half, conf_threshold);
  }
  return std::make_unique<NativeTensorRTBackend>(
    model_path, task, model_type, half, conf_threshold);
}

}  // namespace yolo_cpp
