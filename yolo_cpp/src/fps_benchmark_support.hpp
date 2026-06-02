#ifndef YOLO_CPP_SRC_FPS_BENCHMARK_SUPPORT_HPP_
#define YOLO_CPP_SRC_FPS_BENCHMARK_SUPPORT_HPP_

#include <chrono>
#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "yolo_cpp/camera_backend.hpp"
#include "yolo_cpp/deepstream_runtime.hpp"
#include "yolo_cpp/detector_backend.hpp"
#include "yolo_cpp/frame_packet.hpp"

namespace yolo_cpp::fps_benchmark
{

using Clock = std::chrono::steady_clock;

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
  bool pre_mux_resize{false};
  int pre_mux_width{0};
  int pre_mux_height{0};
  bool deepstream_profile_stages{false};
  int deepstream_profile_stage_interval{300};
  std::string bayer_conversion_backend{"cuda"};

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

struct CameraReleaseGuard
{
  explicit CameraReleaseGuard(std::shared_ptr<yolo_cpp::CameraBackend> camera);
  ~CameraReleaseGuard();

  std::shared_ptr<yolo_cpp::CameraBackend> camera_;
};

bool parse_bool_value(const std::string& value);
Options parse_args(int argc, char** argv);
std::vector<std::string> split_modes(const std::string& mode);

double elapsed_seconds(Clock::time_point start, Clock::time_point end);
Clock::duration seconds_duration(double seconds);

void print_sample_summary(const std::string& label, const std::vector<TimedSample>& samples);
void print_detector_stage_summary(const std::vector<TimedSample>& samples);
void write_csv(const std::string& path, const std::string& mode,
  const std::vector<TimedSample>& samples, bool append);

std::pair<int, int> synthetic_frame_size(const Options& options);
yolo_cpp::FramePacket make_inference_packet(const Options& options);
void print_header(const Options& options);
yolo_cpp::DeepStreamRuntime::Config make_deepstream_config(const Options& options);
void print_decision_aid(const std::map<std::string, BenchmarkResult>& results);

}  // namespace yolo_cpp::fps_benchmark

#endif  // YOLO_CPP_SRC_FPS_BENCHMARK_SUPPORT_HPP_
