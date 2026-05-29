/// @file fps_benchmark.cpp
/// @brief Standalone throughput harness for isolating camera, TensorRT, and DeepStream FPS limits.

#include "yolo_cpp/camera_backend.hpp"
#include "yolo_cpp/deepstream_runtime.hpp"
#include "yolo_cpp/detector_backend.hpp"
#include "yolo_cpp/frame_packet.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <rclcpp/rclcpp.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef HAVE_PYLON
#include <GenApi/GenApi.h>
#include <pylon/PylonIncludes.h>
#endif

namespace
{

using Clock = std::chrono::steady_clock;

constexpr char kDefaultModelPath[] =
  "/home/ultraviolet/CV_Detection/model_test/"
  "best-roboflow-nitish-obb.onnx_b1_gpu0_fp16.engine";

constexpr char kDefaultDeepStreamConfigPath[] =
  "/home/ultraviolet/CV_Detection/yolo_cpp/config/deepstream_bestmerge_nvinfer.txt";

struct Options
{
  std::string mode{"all"};
  double duration_s{10.0};
  double warmup_s{2.0};
  std::string model_path;
  std::string deepstream_config_path;
  std::string input_image_path;
  std::string csv_path;
  std::string task{"obb"};
  std::string model_type{"yolo"};
  bool half{true};
  float conf_threshold{0.30F};

  yolo_cpp::CameraConfig camera;
};

struct TimedSample
{
  double total_ms{0.0};
  yolo_cpp::DetectorStageTimings timings;
  std::size_t detections{0};
};

struct BenchmarkResult
{
  std::string mode;
  bool ok{false};
  double fps{0.0};
  std::size_t frames{0};
  double duration_s{0.0};
  std::string note;
};

#ifdef HAVE_PYLON
struct PylonLifecycle
{
  PylonLifecycle() {Pylon::PylonInitialize();}
  ~PylonLifecycle() {Pylon::PylonTerminate();}
};
#endif

struct CameraReleaseGuard
{
  explicit CameraReleaseGuard(std::shared_ptr<yolo_cpp::CameraBackend> camera)
  : camera_(std::move(camera)) {}

  ~CameraReleaseGuard()
  {
    if (camera_) {
      camera_->release();
    }
  }

  std::shared_ptr<yolo_cpp::CameraBackend> camera_;
};

std::optional<std::string> getenv_string(const char * name)
{
  const char * value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::string(value);
}

bool parse_bool_value(const std::string & value)
{
  std::string normalized = value;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (
    normalized == "1" || normalized == "true" || normalized == "yes" ||
    normalized == "on")
  {
    return true;
  }
  if (
    normalized == "0" || normalized == "false" || normalized == "no" ||
    normalized == "off")
  {
    return false;
  }
  throw std::invalid_argument("Expected boolean value, got '" + value + "'");
}

std::optional<bool> getenv_bool(const char * name)
{
  if (const auto value = getenv_string(name)) {
    return parse_bool_value(*value);
  }
  return std::nullopt;
}

std::optional<int> getenv_int(const char * name)
{
  if (const auto value = getenv_string(name)) {
    return std::stoi(*value);
  }
  return std::nullopt;
}

std::optional<double> getenv_double(const char * name)
{
  if (const auto value = getenv_string(name)) {
    return std::stod(*value);
  }
  return std::nullopt;
}

std::string resolve_default_model_path()
{
  if (const auto env_path = getenv_string("CV_STM_MODEL_PATH")) {
    return *env_path;
  }

  const auto cwd_candidate =
    std::filesystem::current_path() / "model_test" /
    "best-roboflow-nitish-obb.onnx_b1_gpu0_fp16.engine";
  if (std::filesystem::exists(cwd_candidate)) {
    return cwd_candidate.string();
  }
  return kDefaultModelPath;
}

std::string resolve_default_deepstream_config_path()
{
  const auto cwd_candidate =
    std::filesystem::current_path() / "yolo_cpp" / "config" /
    "deepstream_bestmerge_nvinfer.txt";
  if (std::filesystem::exists(cwd_candidate)) {
    return cwd_candidate.string();
  }
  return kDefaultDeepStreamConfigPath;
}

std::string default_camera_type_from_env()
{
  if (const auto camera_type = getenv_string("CV_STM_CAMERA_TYPE")) {
    return *camera_type;
  }
  if (const auto profile = getenv_string("CV_STM_LAUNCH_PROFILE")) {
    if (*profile == "realsense") {
      return "realsense";
    }
  }
  return "basler";
}

Options default_options_from_env()
{
  Options options;
  options.model_path = resolve_default_model_path();
  options.deepstream_config_path = resolve_default_deepstream_config_path();
  options.camera.camera_type = default_camera_type_from_env();

  if (const auto mode = getenv_string("CV_FPS_BENCH_MODE")) {
    options.mode = *mode;
  }
  if (const auto duration = getenv_double("CV_FPS_BENCH_DURATION_S")) {
    options.duration_s = *duration;
  }
  if (const auto warmup = getenv_double("CV_FPS_BENCH_WARMUP_S")) {
    options.warmup_s = *warmup;
  }
  if (const auto conf = getenv_double("CV_FPS_BENCH_CONF")) {
    options.conf_threshold = static_cast<float>(*conf);
  }

  if (const auto value = getenv_int("CV_STM_BASLER_FPS")) {
    options.camera.basler_fps = *value;
  }
  if (const auto value = getenv_bool("CV_STM_BASLER_UNCAPPED_FPS")) {
    options.camera.basler_uncapped_fps = *value;
  }
  if (const auto value = getenv_string("CV_STM_BASLER_PIXEL_FORMAT")) {
    options.camera.basler_pixel_format = *value;
  }
  if (const auto value = getenv_int("CV_STM_BASLER_SOURCE_WIDTH")) {
    options.camera.basler_source_width = *value;
  }
  if (const auto value = getenv_int("CV_STM_BASLER_SOURCE_HEIGHT")) {
    options.camera.basler_source_height = *value;
  }
  if (const auto value = getenv_int("CV_STM_BASLER_BINNING")) {
    options.camera.basler_binning_x = *value;
    options.camera.basler_binning_y = *value;
  }
  if (const auto value = getenv_int("CV_STM_BASLER_BINNING_X")) {
    options.camera.basler_binning_x = *value;
  }
  if (const auto value = getenv_int("CV_STM_BASLER_BINNING_Y")) {
    options.camera.basler_binning_y = *value;
  }
  if (const auto value = getenv_double("CV_STM_BASLER_EXPOSURE_US")) {
    options.camera.basler_exposure_us = *value;
  }
  if (const auto value = getenv_double("CV_STM_BASLER_GAIN_DB")) {
    options.camera.basler_gain_db = *value;
  }
  if (const auto value = getenv_bool("CV_STM_BASLER_USE_FULL_SENSOR")) {
    options.camera.basler_use_full_sensor = *value;
  }
  // Match basler.launch.py and the STM service path unless explicitly overridden.
  options.camera.basler_flip = true;
  if (const auto value = getenv_bool("CV_STM_BASLER_FLIP")) {
    options.camera.basler_flip = *value;
  }

  if (const auto value = getenv_int("CV_STM_REALSENSE_WIDTH")) {
    options.camera.realsense_width = *value;
  }
  if (const auto value = getenv_int("CV_STM_REALSENSE_HEIGHT")) {
    options.camera.realsense_height = *value;
  }
  if (const auto value = getenv_int("CV_STM_REALSENSE_FPS")) {
    options.camera.realsense_fps = *value;
  }
  if (const auto value = getenv_double("CV_STM_REALSENSE_EXPOSURE")) {
    options.camera.realsense_exposure = *value;
  }
  if (const auto value = getenv_bool("CV_STM_REALSENSE_AUTO_EXPOSURE")) {
    options.camera.realsense_auto_exposure = *value;
  }
  if (const auto value = getenv_bool("CV_STM_REALSENSE_USE_POLL")) {
    options.camera.realsense_use_poll = *value;
  }
  if (const auto value = getenv_bool("CV_STM_REALSENSE_USE_MAX_NATIVE_FPS")) {
    options.camera.realsense_use_max_native_fps = *value;
  }

  return options;
}

std::string require_arg_value(int argc, char ** argv, int & index)
{
  if (index + 1 >= argc) {
    throw std::invalid_argument("Missing value for " + std::string(argv[index]));
  }
  ++index;
  return argv[index];
}

void print_usage(const char * program)
{
  std::cout
    << "Usage: " << program << " [options]\n\n"
    << "Modes:\n"
    << "  --mode all                     Run inference, camera, and DeepStream modes\n"
    << "  --mode inference               Synthetic BGR frames through NativeTensorRTBackend\n"
    << "  --mode camera                  CameraBackend capture only, no model\n"
    << "  --mode deepstream              Basler + DeepStream runtime result cadence\n\n"
    << "  --mode basler_features         Print writable Basler GenICam feature nodes\n\n"
    << "Common options:\n"
    << "  --duration SEC                 Measured seconds per mode (default: 10)\n"
    << "  --warmup SEC                   Warmup seconds before measuring (default: 2)\n"
    << "  --model-path PATH              Engine path, defaults to CV_STM_MODEL_PATH\n"
    << "  --deepstream-config-path PATH  DeepStream nvinfer template path\n"
    << "  --input-image PATH             Optional real image for inference mode\n"
    << "  --csv PATH                     Write per-frame samples for offline analysis\n"
    << "  --imgsz-width N                Model input width (default: 640)\n"
    << "  --imgsz-height N               Model input height (default: 640)\n"
    << "  --conf FLOAT                   Confidence threshold (default: 0.30)\n"
    << "  --half BOOL                    Use FP16 network mode where supported (default: true)\n\n"
    << "Camera options mirror STM env defaults:\n"
    << "  --camera-type basler|realsense\n"
    << "  --basler-fps N                 Requested cap/fallback FPS (default: 90)\n"
    << "  --basler-uncapped-fps BOOL     Disable Basler FPS limit (default: true)\n"
    << "  --basler-pixel-format VALUE    bayer_rggb8, yuy2, mono8, bgr8, rgb8\n"
    << "  --basler-source-width N        Full-sensor width (default: 1920)\n"
    << "  --basler-source-height N       Full-sensor height (default: 1200)\n"
    << "  --basler-binning N             Full-FOV sensor binning, e.g. 2 -> 960x600\n"
    << "  --basler-binning-x N           Horizontal binning factor (default: 1)\n"
    << "  --basler-binning-y N           Vertical binning factor (default: 1)\n"
    << "  --basler-exposure-us FLOAT     Exposure in microseconds (default: 5000)\n"
    << "  --basler-gain-db FLOAT         Gain in dB (default: 5)\n"
    << "  --basler-use-full-sensor BOOL  Use full sensor instead of 640x640 ROI (default: true)\n"
    << "  --basler-flip BOOL             Rotate frames 180 degrees (default: true)\n"
    << "  --realsense-width N            RealSense color width (default: 640)\n"
    << "  --realsense-height N           RealSense color height (default: 480)\n"
    << "  --realsense-fps N              RealSense color FPS (default: 60)\n"
    << "  --realsense-use-poll BOOL      Use poll_for_frames (default: false)\n";
}

Options parse_args(int argc, char ** argv)
{
  Options options = default_options_from_env();

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--mode") {
      options.mode = require_arg_value(argc, argv, i);
    } else if (arg == "--duration") {
      options.duration_s = std::stod(require_arg_value(argc, argv, i));
    } else if (arg == "--warmup") {
      options.warmup_s = std::stod(require_arg_value(argc, argv, i));
    } else if (arg == "--model-path") {
      options.model_path = require_arg_value(argc, argv, i);
    } else if (arg == "--deepstream-config-path") {
      options.deepstream_config_path = require_arg_value(argc, argv, i);
    } else if (arg == "--input-image") {
      options.input_image_path = require_arg_value(argc, argv, i);
    } else if (arg == "--csv") {
      options.csv_path = require_arg_value(argc, argv, i);
    } else if (arg == "--imgsz-width") {
      options.camera.imgsz_width = std::stoi(require_arg_value(argc, argv, i));
    } else if (arg == "--imgsz-height") {
      options.camera.imgsz_height = std::stoi(require_arg_value(argc, argv, i));
    } else if (arg == "--conf") {
      options.conf_threshold = static_cast<float>(std::stof(require_arg_value(argc, argv, i)));
    } else if (arg == "--half") {
      options.half = parse_bool_value(require_arg_value(argc, argv, i));
    } else if (arg == "--camera-type") {
      options.camera.camera_type = require_arg_value(argc, argv, i);
    } else if (arg == "--basler-fps") {
      options.camera.basler_fps = std::stoi(require_arg_value(argc, argv, i));
    } else if (arg == "--basler-uncapped-fps") {
      options.camera.basler_uncapped_fps = parse_bool_value(require_arg_value(argc, argv, i));
    } else if (arg == "--basler-pixel-format") {
      options.camera.basler_pixel_format = require_arg_value(argc, argv, i);
    } else if (arg == "--basler-source-width") {
      options.camera.basler_source_width = std::stoi(require_arg_value(argc, argv, i));
    } else if (arg == "--basler-source-height") {
      options.camera.basler_source_height = std::stoi(require_arg_value(argc, argv, i));
    } else if (arg == "--basler-binning") {
      const int binning = std::stoi(require_arg_value(argc, argv, i));
      options.camera.basler_binning_x = binning;
      options.camera.basler_binning_y = binning;
    } else if (arg == "--basler-binning-x") {
      options.camera.basler_binning_x = std::stoi(require_arg_value(argc, argv, i));
    } else if (arg == "--basler-binning-y") {
      options.camera.basler_binning_y = std::stoi(require_arg_value(argc, argv, i));
    } else if (arg == "--basler-exposure-us") {
      options.camera.basler_exposure_us = std::stod(require_arg_value(argc, argv, i));
    } else if (arg == "--basler-gain-db") {
      options.camera.basler_gain_db = std::stod(require_arg_value(argc, argv, i));
    } else if (arg == "--basler-use-full-sensor") {
      options.camera.basler_use_full_sensor = parse_bool_value(require_arg_value(argc, argv, i));
    } else if (arg == "--basler-flip") {
      options.camera.basler_flip = parse_bool_value(require_arg_value(argc, argv, i));
    } else if (arg == "--realsense-width") {
      options.camera.realsense_width = std::stoi(require_arg_value(argc, argv, i));
    } else if (arg == "--realsense-height") {
      options.camera.realsense_height = std::stoi(require_arg_value(argc, argv, i));
    } else if (arg == "--realsense-fps") {
      options.camera.realsense_fps = std::stoi(require_arg_value(argc, argv, i));
    } else if (arg == "--realsense-use-poll") {
      options.camera.realsense_use_poll = parse_bool_value(require_arg_value(argc, argv, i));
    } else {
      throw std::invalid_argument("Unknown option: " + arg);
    }
  }

  if (options.duration_s <= 0.0) {
    throw std::invalid_argument("--duration must be positive");
  }
  if (options.warmup_s < 0.0) {
    throw std::invalid_argument("--warmup cannot be negative");
  }
  if (options.camera.imgsz_width <= 0 || options.camera.imgsz_height <= 0) {
    throw std::invalid_argument("--imgsz-width and --imgsz-height must be positive");
  }
  options.camera.basler_binning_x = std::max(1, options.camera.basler_binning_x);
  options.camera.basler_binning_y = std::max(1, options.camera.basler_binning_y);
  return options;
}

std::vector<std::string> split_modes(const std::string & mode)
{
  if (mode == "all") {
    return {"inference", "camera", "deepstream"};
  }

  std::vector<std::string> modes;
  std::stringstream stream(mode);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      modes.push_back(item);
    }
  }
  if (modes.empty()) {
    throw std::invalid_argument("No benchmark modes selected");
  }
  return modes;
}

#ifdef HAVE_PYLON
std::string access_text(GenApi::INode * node)
{
  if (node == nullptr || !GenApi::IsAvailable(node)) {
    return "absent";
  }
  std::string access = GenApi::IsReadable(node) ? "read" : "no-read";
  access += GenApi::IsWritable(node) ? "/write" : "/no-write";
  return access;
}

void print_genicam_node(GenApi::INodeMap & node_map, const char * name)
{
  GenApi::INode * raw_node = node_map.GetNode(name);
  std::cout << "  " << name << ": " << access_text(raw_node);
  if (raw_node == nullptr || !GenApi::IsAvailable(raw_node) || !GenApi::IsReadable(raw_node)) {
    std::cout << '\n';
    return;
  }

  try {
    if (GenApi::CEnumerationPtr node(raw_node); node) {
      std::cout << " value=" << node->ToString();
    } else if (GenApi::CIntegerPtr node(raw_node); node) {
      std::cout << " value=" << node->GetValue()
                << " min=" << node->GetMin()
                << " max=" << node->GetMax()
                << " inc=" << node->GetInc();
    } else if (GenApi::CFloatPtr node(raw_node); node) {
      std::cout << " value=" << node->GetValue()
                << " min=" << node->GetMin()
                << " max=" << node->GetMax();
    } else if (GenApi::CBooleanPtr node(raw_node); node) {
      std::cout << " value=" << (node->GetValue() ? "true" : "false");
    } else if (GenApi::CStringPtr node(raw_node); node) {
      std::cout << " value=" << node->GetValue();
    }
  } catch (const std::exception & exc) {
    std::cout << " read_error='" << exc.what() << "'";
  } catch (...) {
    std::cout << " read_error='non-standard exception'";
  }
  std::cout << '\n';
}

bool try_set_enum(GenApi::INodeMap & node_map, const char * name, const char * value)
{
  GenApi::CEnumerationPtr node(node_map.GetNode(name));
  if (!node || !GenApi::IsWritable(node)) {
    return false;
  }
  GenApi::CEnumEntryPtr entry(node->GetEntryByName(value));
  if (!entry || !GenApi::IsReadable(entry)) {
    return false;
  }
  node->SetIntValue(entry->GetValue());
  return true;
}

bool try_set_integer(GenApi::INodeMap & node_map, const char * name, int value)
{
  GenApi::CIntegerPtr node(node_map.GetNode(name));
  if (!node || !GenApi::IsWritable(node)) {
    return false;
  }
  const int64_t min_value = node->GetMin();
  const int64_t max_value = node->GetMax();
  const int64_t inc = std::max<int64_t>(1, node->GetInc());
  int64_t clamped = std::max<int64_t>(min_value, std::min<int64_t>(max_value, value));
  if (inc > 1) {
    clamped = min_value + (((clamped - min_value) / inc) * inc);
  }
  node->SetValue(clamped);
  return true;
}

bool try_set_binning_pair(
  GenApi::INodeMap & node_map,
  int binning_x,
  int binning_y)
{
  const bool set_x = binning_x <= 1 ||
    try_set_integer(node_map, "BinningHorizontal", binning_x);
  const bool set_y = binning_y <= 1 ||
    try_set_integer(node_map, "BinningVertical", binning_y);
  return set_x && set_y;
}
#endif

double elapsed_seconds(Clock::time_point start, Clock::time_point end)
{
  return std::chrono::duration<double>(end - start).count();
}

Clock::duration seconds_duration(double seconds)
{
  return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}

double percentile(std::vector<double> values, double p)
{
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = p * static_cast<double>(values.size() - 1U);
  const auto lower = static_cast<std::size_t>(std::floor(rank));
  const auto upper = std::min<std::size_t>(lower + 1U, values.size() - 1U);
  const double fraction = rank - static_cast<double>(lower);
  return values[lower] + ((values[upper] - values[lower]) * fraction);
}

double average(const std::vector<double> & values)
{
  if (values.empty()) {
    return 0.0;
  }
  double total = 0.0;
  for (const auto value : values) {
    total += value;
  }
  return total / static_cast<double>(values.size());
}

std::vector<double> total_times(const std::vector<TimedSample> & samples)
{
  std::vector<double> values;
  values.reserve(samples.size());
  for (const auto & sample : samples) {
    values.push_back(sample.total_ms);
  }
  return values;
}

void print_sample_summary(const std::string & label, const std::vector<TimedSample> & samples)
{
  const auto totals = total_times(samples);
  std::cout << std::fixed << std::setprecision(3)
            << "  " << label << "_ms avg=" << average(totals)
            << " p50=" << percentile(totals, 0.50)
            << " p95=" << percentile(totals, 0.95)
            << " max=" << (totals.empty() ? 0.0 : *std::max_element(totals.begin(), totals.end()))
            << '\n';
}

void print_detector_stage_summary(const std::vector<TimedSample> & samples)
{
  if (samples.empty()) {
    return;
  }

  yolo_cpp::DetectorStageTimings totals{};
  std::size_t detections = 0;
  for (const auto & sample : samples) {
    totals.preprocess_ms += sample.timings.preprocess_ms;
    totals.upload_ms += sample.timings.upload_ms;
    totals.inference_ms += sample.timings.inference_ms;
    totals.download_ms += sample.timings.download_ms;
    totals.parse_ms += sample.timings.parse_ms;
    detections += sample.detections;
  }
  const double n = static_cast<double>(samples.size());
  std::cout << std::fixed << std::setprecision(3)
            << "  stage_avg_ms preprocess=" << totals.preprocess_ms / n
            << " upload=" << totals.upload_ms / n
            << " infer=" << totals.inference_ms / n
            << " download=" << totals.download_ms / n
            << " parse=" << totals.parse_ms / n
            << " detections_per_frame=" << static_cast<double>(detections) / n
            << '\n';
}

void write_csv(
  const std::string & path,
  const std::string & mode,
  const std::vector<TimedSample> & samples,
  bool append)
{
  if (path.empty()) {
    return;
  }

  std::ofstream stream(path, append ? std::ios::app : std::ios::out);
  if (!stream) {
    throw std::runtime_error("Failed to open CSV output: " + path);
  }
  if (!append) {
    stream << "mode,index,total_ms,preprocess_ms,upload_ms,inference_ms,download_ms,parse_ms,"
           << "detections\n";
  }
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const auto & sample = samples[i];
    stream << mode << ','
           << i << ','
           << sample.total_ms << ','
           << sample.timings.preprocess_ms << ','
           << sample.timings.upload_ms << ','
           << sample.timings.inference_ms << ','
           << sample.timings.download_ms << ','
           << sample.timings.parse_ms << ','
           << sample.detections << '\n';
  }
}

std::pair<int, int> synthetic_frame_size(const Options & options)
{
  if (options.camera.camera_type == "realsense") {
    return {options.camera.realsense_width, options.camera.realsense_height};
  }
  if (options.camera.basler_use_full_sensor) {
    return {basler_output_width(options.camera), basler_output_height(options.camera)};
  }
  return {options.camera.imgsz_width, options.camera.imgsz_height};
}

yolo_cpp::FramePacket make_inference_packet(const Options & options)
{
  yolo_cpp::FramePacket packet;
  cv::Mat image;
  if (!options.input_image_path.empty()) {
    image = cv::imread(options.input_image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("Failed to read --input-image: " + options.input_image_path);
    }
  } else {
    const auto [width, height] = synthetic_frame_size(options);
    image.create(height, width, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
  }

  packet.frame = image;
  packet.pixel_format = yolo_cpp::PixelFormat::kBgr8;
  packet.storage_kind = yolo_cpp::BackingStoreKind::kHost;
  packet.source_id = "synthetic";
  packet.capture_time_ns = static_cast<std::int64_t>(
    std::chrono::duration_cast<std::chrono::nanoseconds>(
      Clock::now().time_since_epoch()).count());
  return packet;
}

void print_header(const Options & options)
{
  const auto [source_width, source_height] = synthetic_frame_size(options);
  std::cout
    << "[fps_benchmark] mode=" << options.mode
    << " duration_s=" << options.duration_s
    << " warmup_s=" << options.warmup_s
    << " model=" << options.model_path
    << "\n[fps_benchmark] model_input=" << options.camera.imgsz_width << 'x'
    << options.camera.imgsz_height
    << " synthetic_source=" << source_width << 'x' << source_height
    << " camera=" << options.camera.camera_type
    << " basler_source=" << options.camera.basler_source_width << 'x'
    << options.camera.basler_source_height
    << " basler_binning=" << options.camera.basler_binning_x << 'x'
    << options.camera.basler_binning_y
    << " basler_fps=" << options.camera.basler_fps
    << " basler_uncapped=" << (options.camera.basler_uncapped_fps ? "true" : "false")
    << " basler_pixel_format=" << options.camera.basler_pixel_format
    << " basler_exposure_us=" << options.camera.basler_exposure_us
    << '\n';
}

BenchmarkResult run_inference_mode(
  const Options & options,
  const rclcpp::Logger & logger,
  bool csv_append)
{
  std::cout << "\n[inference] Native TensorRT synthetic-frame benchmark\n";
  auto backend = yolo_cpp::create_detector_backend(
    "native_tensorrt",
    options.model_path,
    options.task,
    options.model_type,
    options.half,
    options.conf_threshold);
  auto packet = make_inference_packet(options);
  yolo_cpp::DetectorStageTimings init_timings{};
  (void)backend->infer(packet, logger, &init_timings);
  std::cout << "[inference] frame=" << packet.width() << 'x' << packet.height()
            << " backend=" << backend->backend_name() << '\n';

  const auto warmup_end = Clock::now() + std::chrono::duration<double>(options.warmup_s);
  while (Clock::now() < warmup_end) {
    yolo_cpp::DetectorStageTimings timings{};
    (void)backend->infer(packet, logger, &timings);
  }

  std::vector<TimedSample> samples;
  const auto start = Clock::now();
  const auto end = start + std::chrono::duration<double>(options.duration_s);
  while (Clock::now() < end) {
    yolo_cpp::DetectorStageTimings timings{};
    const auto frame_start = Clock::now();
    const auto batch = backend->infer(packet, logger, &timings);
    const auto frame_done = Clock::now();
    samples.push_back(TimedSample{
      std::chrono::duration<double, std::milli>(frame_done - frame_start).count(),
      timings,
      batch.detections.size()});
  }
  const auto done = Clock::now();
  const double measured_s = elapsed_seconds(start, done);
  const double fps = samples.empty() ? 0.0 : static_cast<double>(samples.size()) / measured_s;

  std::cout << std::fixed << std::setprecision(2)
            << "[inference] frames=" << samples.size()
            << " measured_s=" << measured_s
            << " fps=" << fps << '\n';
  print_sample_summary("total", samples);
  print_detector_stage_summary(samples);
  write_csv(options.csv_path, "inference", samples, csv_append);

  return BenchmarkResult{"inference", true, fps, samples.size(), measured_s, ""};
}

BenchmarkResult run_basler_features_mode(const Options & options)
{
  std::cout << "\n[basler_features] Basler GenICam feature probe\n";
#ifndef HAVE_PYLON
  throw std::runtime_error("Basler feature probe unavailable: build missing Pylon SDK");
#else
  PylonLifecycle pylon;
  Pylon::CInstantCamera camera(Pylon::CTlFactory::GetInstance().CreateFirstDevice());
  camera.Open();

  std::cout << "[basler_features] model=" << camera.GetDeviceInfo().GetModelName()
            << " serial=" << camera.GetDeviceInfo().GetSerialNumber() << '\n';
  auto & node_map = camera.GetNodeMap();
  const std::vector<const char *> nodes{
    "BinningSelector",
    "BinningHorizontal",
    "BinningVertical",
    "BinningHorizontalMode",
    "BinningVerticalMode",
    "DecimationHorizontal",
    "DecimationVertical",
    "DecimationHorizontalMode",
    "DecimationVerticalMode",
    "ScalingHorizontal",
    "ScalingVertical",
    "PixelFormat",
    "Width",
    "Height",
    "OffsetX",
    "OffsetY",
    "SensorWidth",
    "SensorHeight",
    "WidthMax",
    "HeightMax"};

  std::cout << "[basler_features] current nodes:\n";
  for (const auto * name : nodes) {
    print_genicam_node(node_map, name);
  }

  const int requested_x = std::max(1, options.camera.basler_binning_x);
  const int requested_y = std::max(1, options.camera.basler_binning_y);
  if (requested_x > 1 || requested_y > 1) {
    std::cout << "[basler_features] trying binning=" << requested_x << 'x' << requested_y
              << '\n';
    bool did_set = false;
    try {
      did_set = try_set_binning_pair(node_map, requested_x, requested_y);
      std::cout << "  default selector: " << (did_set ? "set" : "not writable") << '\n';
    } catch (const std::exception & exc) {
      std::cout << "  default selector error: " << exc.what() << '\n';
    }

    for (const char * selector : {"Sensor", "Region0", "All"}) {
      try {
        if (!try_set_enum(node_map, "BinningSelector", selector)) {
          std::cout << "  selector " << selector << ": unavailable\n";
          continue;
        }
        const bool selector_set = try_set_binning_pair(node_map, requested_x, requested_y);
        std::cout << "  selector " << selector << ": "
                  << (selector_set ? "set" : "not writable") << '\n';
      } catch (const std::exception & exc) {
        std::cout << "  selector " << selector << " error: " << exc.what() << '\n';
      }
    }

    std::cout << "[basler_features] after write attempts:\n";
    for (const auto * name : nodes) {
      print_genicam_node(node_map, name);
    }
  }

  camera.Close();
  return BenchmarkResult{"basler_features", true, 0.0, 0U, 0.0, ""};
#endif
}

BenchmarkResult run_camera_mode(
  const Options & options,
  const rclcpp::Logger & logger,
  bool csv_append)
{
  std::cout << "\n[camera] Camera capture-only benchmark\n";
  auto camera = yolo_cpp::create_camera_backend(options.camera, logger);
  CameraReleaseGuard release_guard(camera);
  camera->initialize();

  const auto warmup_end = Clock::now() + std::chrono::duration<double>(options.warmup_s);
  yolo_cpp::FramePacket packet;
  while (Clock::now() < warmup_end) {
    (void)camera->capture(packet);
  }

  std::vector<TimedSample> samples;
  int first_width = 0;
  int first_height = 0;
  const auto start = Clock::now();
  const auto end = start + std::chrono::duration<double>(options.duration_s);
  while (Clock::now() < end) {
    const auto frame_start = Clock::now();
    if (!camera->capture(packet)) {
      continue;
    }
    const auto frame_done = Clock::now();
    if (first_width == 0 && !packet.empty()) {
      first_width = packet.width();
      first_height = packet.height();
    }
    samples.push_back(TimedSample{
      std::chrono::duration<double, std::milli>(frame_done - frame_start).count(),
      {},
      0U});
  }
  const auto done = Clock::now();
  const double measured_s = elapsed_seconds(start, done);
  const double fps = samples.empty() ? 0.0 : static_cast<double>(samples.size()) / measured_s;

  std::cout << std::fixed << std::setprecision(2)
            << "[camera] frames=" << samples.size()
            << " measured_s=" << measured_s
            << " fps=" << fps
            << " camera_reported_fps=" << camera->actual_fps()
            << " frame=" << first_width << 'x' << first_height
            << " source=" << camera->source_id() << '\n';
  print_sample_summary("capture", samples);
  write_csv(options.csv_path, "camera", samples, csv_append);

  return BenchmarkResult{"camera", true, fps, samples.size(), measured_s, ""};
}

yolo_cpp::DeepStreamRuntime::Config make_deepstream_config(const Options & options)
{
  yolo_cpp::DeepStreamRuntime::Config config;
  config.model_path = options.model_path;
  config.config_template_path = options.deepstream_config_path;
  config.task = options.task;
  config.imgsz_width = options.camera.imgsz_width;
  config.imgsz_height = options.camera.imgsz_height;
  config.source_width = basler_output_width(options.camera);
  config.source_height = basler_output_height(options.camera);
  config.source_fps = options.camera.basler_fps;
  config.source_exposure_us = options.camera.basler_exposure_us;
  config.source_gain_db = options.camera.basler_gain_db;
  config.source_fps_uncapped = options.camera.basler_uncapped_fps;
  config.source_use_full_sensor = options.camera.basler_use_full_sensor;
  config.source_binning_x = options.camera.basler_binning_x;
  config.source_binning_y = options.camera.basler_binning_y;
  config.source_pixel_format = options.camera.basler_pixel_format;
  config.flip_source = options.camera.basler_flip;
  config.conf_threshold = options.conf_threshold;
  config.half = options.half;
  config.enable_color_verification = false;
  config.debug_publish_image_raw = false;
  config.source_element = "pylonsrc";
  config.source_id = "basler";
  return config;
}

BenchmarkResult run_deepstream_mode(
  const Options & options,
  const rclcpp::Logger & logger,
  bool csv_append)
{
  if (options.camera.camera_type != "basler") {
    throw std::runtime_error("DeepStream benchmark mode requires --camera-type basler");
  }

  std::cout << "\n[deepstream] Basler + DeepStream result-cadence benchmark\n";
  std::vector<TimedSample> samples;
  std::mutex mutex;
  Clock::time_point last_callback{};
  Clock::time_point measure_start{};
  Clock::time_point measure_end{};
  std::atomic<bool> timing_ready{false};
  bool have_last_callback = false;
  std::size_t warmup_frames = 0;

  auto runtime = std::make_unique<yolo_cpp::DeepStreamRuntime>(
    logger,
    make_deepstream_config(options),
    [&](yolo_cpp::DeepStreamRuntime::Result && result) {
      if (!timing_ready.load(std::memory_order_acquire)) {
        return;
      }
      const auto now = Clock::now();
      if (now < measure_start) {
        ++warmup_frames;
        return;
      }
      if (now > measure_end) {
        return;
      }

      std::lock_guard<std::mutex> lock(mutex);
      double interval_ms = 0.0;
      if (have_last_callback) {
        interval_ms = std::chrono::duration<double, std::milli>(now - last_callback).count();
      }
      last_callback = now;
      have_last_callback = true;
      samples.push_back(TimedSample{
        interval_ms,
        result.timings,
        result.batch.detections.size()});
    });

  std::cout << "[deepstream] backend=" << runtime->backend_name()
            << " source_reported_fps=" << runtime->source_actual_fps()
            << " deepstream_config=" << options.deepstream_config_path << '\n';
  measure_start = Clock::now() + seconds_duration(options.warmup_s);
  measure_end = measure_start + seconds_duration(options.duration_s);
  timing_ready.store(true, std::memory_order_release);
  std::this_thread::sleep_until(measure_end);
  runtime.reset();

  const double measured_s = options.duration_s;
  const double fps = samples.empty() ? 0.0 : static_cast<double>(samples.size()) / measured_s;
  std::cout << std::fixed << std::setprecision(2)
            << "[deepstream] warmup_frames=" << warmup_frames
            << " frames=" << samples.size()
            << " measured_s=" << measured_s
            << " fps=" << fps << '\n';
  if (samples.size() > 1U) {
    std::vector<TimedSample> interval_samples(samples.begin() + 1, samples.end());
    print_sample_summary("callback_interval", interval_samples);
  }
  print_detector_stage_summary(samples);
  write_csv(options.csv_path, "deepstream", samples, csv_append);

  return BenchmarkResult{"deepstream", true, fps, samples.size(), measured_s, ""};
}

void print_decision_aid(const std::map<std::string, BenchmarkResult> & results)
{
  const auto inference = results.find("inference");
  const auto camera = results.find("camera");
  const auto deepstream = results.find("deepstream");
  if (inference == results.end() && camera == results.end() && deepstream == results.end()) {
    return;
  }

  std::cout << "\n[readout] ";
  if (
    inference != results.end() && camera != results.end() &&
    inference->second.ok && camera->second.ok)
  {
    if (inference->second.fps < camera->second.fps * 0.8) {
      std::cout
        << "inference FPS is below camera-only FPS, "
        << "so the model/GPU path is the lower ceiling. ";
    } else if (camera->second.fps < inference->second.fps * 0.8) {
      std::cout
        << "camera-only FPS is below inference FPS, "
        << "so camera exposure/format/link is the lower ceiling. ";
    } else {
      std::cout
        << "camera-only and inference-only FPS are close, "
        << "so check full-pipeline overhead next. ";
    }
  }
  if (
    deepstream != results.end() && deepstream->second.ok &&
    inference != results.end() && inference->second.ok &&
    camera != results.end() && camera->second.ok)
  {
    const double isolated_ceiling = std::min(inference->second.fps, camera->second.fps);
    if (deepstream->second.fps < isolated_ceiling * 0.8) {
      std::cout
        << "DeepStream combined FPS is well below both isolated ceilings, "
        << "so inspect GStreamer caps, color conversion, and drops.";
    } else {
      std::cout << "DeepStream combined FPS tracks the lower isolated ceiling.";
    }
  }
  std::cout << '\n';
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  const auto logger = rclcpp::get_logger("fps_benchmark");

  int exit_code = 0;
  try {
    const Options options = parse_args(argc, argv);
    print_header(options);

    std::map<std::string, BenchmarkResult> results;
    bool wrote_csv = false;
    for (const auto & mode : split_modes(options.mode)) {
      try {
        if (mode == "inference") {
          results[mode] = run_inference_mode(options, logger, wrote_csv);
          wrote_csv = wrote_csv || !options.csv_path.empty();
        } else if (mode == "camera") {
          results[mode] = run_camera_mode(options, logger, wrote_csv);
          wrote_csv = wrote_csv || !options.csv_path.empty();
        } else if (mode == "deepstream") {
          results[mode] = run_deepstream_mode(options, logger, wrote_csv);
          wrote_csv = wrote_csv || !options.csv_path.empty();
        } else if (mode == "basler_features") {
          results[mode] = run_basler_features_mode(options);
        } else {
          throw std::invalid_argument("Unknown benchmark mode: " + mode);
        }
      } catch (const std::exception & exc) {
        exit_code = 1;
        results[mode] = BenchmarkResult{mode, false, 0.0, 0U, 0.0, exc.what()};
        std::cerr << '[' << mode << "] ERROR: " << exc.what() << '\n';
      } catch (...) {
        exit_code = 1;
        results[mode] = BenchmarkResult{
          mode,
          false,
          0.0,
          0U,
          0.0,
          "non-standard exception"};
        std::cerr << '[' << mode << "] ERROR: non-standard exception\n";
      }
    }

    print_decision_aid(results);
  } catch (const std::exception & exc) {
    std::cerr << "[fps_benchmark] ERROR: " << exc.what() << '\n';
    exit_code = 2;
  }

  rclcpp::shutdown();
  return exit_code;
}
