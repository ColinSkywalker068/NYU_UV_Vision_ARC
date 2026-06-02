/// @file fps_benchmark.cpp
/// @brief Standalone throughput harness for isolating camera, TensorRT, and DeepStream FPS limits.

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "fps_benchmark_support.hpp"

namespace
{

using yolo_cpp::fps_benchmark::BenchmarkResult;
using yolo_cpp::fps_benchmark::CameraReleaseGuard;
using yolo_cpp::fps_benchmark::Clock;
using yolo_cpp::fps_benchmark::elapsed_seconds;
using yolo_cpp::fps_benchmark::make_deepstream_config;
using yolo_cpp::fps_benchmark::make_inference_packet;
using yolo_cpp::fps_benchmark::Options;
using yolo_cpp::fps_benchmark::parse_args;
using yolo_cpp::fps_benchmark::print_decision_aid;
using yolo_cpp::fps_benchmark::print_detector_stage_summary;
using yolo_cpp::fps_benchmark::print_header;
using yolo_cpp::fps_benchmark::print_sample_summary;
using yolo_cpp::fps_benchmark::seconds_duration;
using yolo_cpp::fps_benchmark::split_modes;
using yolo_cpp::fps_benchmark::TimedSample;
using yolo_cpp::fps_benchmark::write_csv;

BenchmarkResult run_inference_mode(
  const Options& options, const rclcpp::Logger& logger, bool csv_append)
{
  std::cout << "\n[inference] Native TensorRT synthetic-frame benchmark\n";
  auto backend = yolo_cpp::create_detector_backend("native_tensorrt", options.model_path,
    options.task, options.model_type, options.half, options.conf_threshold);
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
    samples.push_back(
      TimedSample{std::chrono::duration<double, std::milli>(frame_done - frame_start).count(),
        timings, batch.detections.size()});
  }
  const auto done = Clock::now();
  const double measured_s = elapsed_seconds(start, done);
  const double fps = samples.empty() ? 0.0 : static_cast<double>(samples.size()) / measured_s;

  std::cout << std::fixed << std::setprecision(2) << "[inference] frames=" << samples.size()
            << " measured_s=" << measured_s << " fps=" << fps << '\n';
  print_sample_summary("total", samples);
  print_detector_stage_summary(samples);
  write_csv(options.csv_path, "inference", samples, csv_append);

  return BenchmarkResult{"inference", true, fps, samples.size(), measured_s, ""};
}

BenchmarkResult run_camera_mode(
  const Options& options, const rclcpp::Logger& logger, bool csv_append)
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
      std::chrono::duration<double, std::milli>(frame_done - frame_start).count(), {}, 0U});
  }
  const auto done = Clock::now();
  const double measured_s = elapsed_seconds(start, done);
  const double fps = samples.empty() ? 0.0 : static_cast<double>(samples.size()) / measured_s;

  std::cout << std::fixed << std::setprecision(2) << "[camera] frames=" << samples.size()
            << " measured_s=" << measured_s << " fps=" << fps
            << " camera_reported_fps=" << camera->actual_fps() << " frame=" << first_width << 'x'
            << first_height << " source=" << camera->source_id() << '\n';
  print_sample_summary("capture", samples);
  write_csv(options.csv_path, "camera", samples, csv_append);

  return BenchmarkResult{"camera", true, fps, samples.size(), measured_s, ""};
}

BenchmarkResult run_deepstream_mode(
  const Options& options, const rclcpp::Logger& logger, bool csv_append)
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
    logger, make_deepstream_config(options), [&](yolo_cpp::DeepStreamRuntime::Result&& result) {
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
      samples.push_back(TimedSample{interval_ms, result.timings, result.batch.detections.size()});
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
  std::cout << std::fixed << std::setprecision(2) << "[deepstream] warmup_frames=" << warmup_frames
            << " frames=" << samples.size() << " measured_s=" << measured_s << " fps=" << fps
            << '\n';
  if (samples.size() > 1U) {
    std::vector<TimedSample> interval_samples(samples.begin() + 1, samples.end());
    print_sample_summary("callback_interval", interval_samples);
  }
  print_detector_stage_summary(samples);
  write_csv(options.csv_path, "deepstream", samples, csv_append);

  return BenchmarkResult{"deepstream", true, fps, samples.size(), measured_s, ""};
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  const auto logger = rclcpp::get_logger("fps_benchmark");

  int exit_code = 0;
  try {
    const Options options = parse_args(argc, argv);
    print_header(options);

    std::map<std::string, BenchmarkResult> results;
    bool wrote_csv = false;
    for (const auto& mode : split_modes(options.mode)) {
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
        } else {
          throw std::invalid_argument("Unknown benchmark mode: " + mode);
        }
      } catch (const std::exception& exc) {
        exit_code = 1;
        results[mode] = BenchmarkResult{mode, false, 0.0, 0U, 0.0, exc.what()};
        std::cerr << '[' << mode << "] ERROR: " << exc.what() << '\n';
      } catch (...) {
        exit_code = 1;
        results[mode] = BenchmarkResult{mode, false, 0.0, 0U, 0.0, "non-standard exception"};
        std::cerr << '[' << mode << "] ERROR: non-standard exception\n";
      }
    }

    print_decision_aid(results);
  } catch (const std::exception& exc) {
    std::cerr << "[fps_benchmark] ERROR: " << exc.what() << '\n';
    exit_code = 2;
  }

  rclcpp::shutdown();
  return exit_code;
}
