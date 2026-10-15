// Observational benchmark of production tracker + aim prediction, without ROS/hardware.
// Run: bash tools/benchmark-moving-following.sh
// All scenarios: direct plate at Z=4m, 46ms capture-to-solver age, 18m/s projectile,
// default 35ms position filter, gravity disabled for the horizontal comparison.
// The reference uses exact current position/velocity with the SAME solver. It is
// a constant-velocity intercept reference, not foreknowledge of future maneuvers.
// Instantaneous +/-2m/s reversal is a stress case, not a physical robot model.
// Noise is deterministic alternating +/-15mm, not measured camera noise. Results
// show response/jitter tradeoffs; they do not prescribe gains or change defaults.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "yolo_cpp/aim_prediction.hpp"
#include "yolo_cpp/tracker.hpp"

namespace
{
constexpr double kRadiansToDegrees = 180.0 / 3.141592653589793;
constexpr double kFrameAgeS = 0.046;

std::int64_t stamp(double time_s)
{
  return 1000000000LL + std::llround(time_s * 1e9);
}

yolo_cpp::TrackObservation observation(double x, double time_s)
{
  yolo_cpp::TrackObservation out;
  out.center = {320, 240};
  out.bbox = {300, 230, 40, 20};
  out.confidence = .9;
  out.image_width = 640;
  out.image_height = 480;
  out.has_pose = true;
  out.position = {x, 0, 4};
  out.measurement_stamp_ns = stamp(time_s);
  return out;
}

yolo_cpp::TrackSnapshot selected(const yolo_cpp::TrackingFrameResult & result)
{
  if (!result.selected_target) {
    throw std::runtime_error("Benchmark lost its synthetic target; inspect tracker changes");
  }
  return *result.selected_target;
}

yolo_cpp::AimPredictionResult solve(const yolo_cpp::TrackSnapshot & track, double time_s)
{
  yolo_cpp::AimPredictionInput input;
  input.position = track.observation.position;
  input.velocity = {track.velocity[0], track.velocity[1], track.velocity[2]};
  input.has_velocity = true;
  input.frame_stamp_ns = stamp(time_s);
  input.measurement_stamp_ns = track.observation.measurement_stamp_ns;
  input.now_ns = input.frame_stamp_ns + std::llround(kFrameAgeS * 1e9);
  yolo_cpp::AimPredictionConfig config;
  config.bullet_speed_mps = 18;
  config.compensate_gravity = false;
  const auto result = yolo_cpp::predict_aim(input, config);
  if (!result.valid) {
    throw std::runtime_error(result.reason);
  }
  return result;
}

double intercept_error(const yolo_cpp::TrackSnapshot & track, double time_s,
  double x, double velocity)
{
  auto reference = track;
  reference.observation.position = {x, 0, 4};
  reference.velocity = {velocity, 0, 0};
  return std::abs(solve(track, time_s).yaw - solve(reference, time_s).yaw) * kRadiansToDegrees;
}

void benchmark(const yolo_cpp::TrackConfig & config, double fps)
{
  // One second at +2m/s, then an abrupt reversal. Report first threshold crossing
  // relative to the capture-time reversal; frame latency/motor delay is not added.
  yolo_cpp::TrackManager reversal(config);
  double zero_s = -1, settle_s = -1, max_error = 0, max_position_error = 0, steady_error = 0;
  unsigned rejects = 0;
  for (int frame = 0; frame < 3 * fps; ++frame) {
    const double time = frame / fps;
    const double x = time <= 1 ? 2 * (time - 1) : -2 * (time - 1);
    const auto result = reversal.update({observation(x, time)}, 1 / fps);
    const auto track = selected(result);
    rejects += result.innovation_rejected_count;
    const double error = intercept_error(track, time, x, time <= 1 ? 2 : -2);
    if (time > .8 && time < 1) {steady_error = std::max(steady_error, error);}
    if (time > 1 && time < 2) {
      if (zero_s < 0 && track.velocity[0] <= 0) {zero_s = time - 1;}
      // +2 to -1.8 completes 95% of the +2 to -2 step.
      if (settle_s < 0 && track.velocity[0] <= -1.8) {settle_s = time - 1;}
      max_error = std::max(max_error, error);
      max_position_error = std::max(max_position_error, std::abs(track.observation.position.x - x));
    }
  }
  std::cout << "scenario=reversal fps=" << fps
            << " velocity_tau_s=" << config.velocity_filter_tau_s
            << " steady_max_intercept_error_deg=" << steady_error
            << " reversal_velocity_zero_ms=" << (zero_s < 0 ? -1 : zero_s * 1000)
            << " reversal_velocity95_ms=" << (settle_s < 0 ? -1 : settle_s * 1000)
            << " reversal_peak_intercept_error_deg=" << max_error
            << " reversal_peak_capture_position_error_m=" << max_position_error
            << " innovation_rejections=" << rejects << '\n';

  // Settled stationary plate starts translating at +2m/s after one second.
  yolo_cpp::TrackManager start(config);
  double rise_s = -1, start_error = 0;
  for (int frame = 0; frame < 2 * fps; ++frame) {
    const double time = frame / fps;
    const double x = std::max(0.0, 2 * (time - 1));
    const auto track = selected(start.update({observation(x, time)}, 1 / fps));
    if (time > 1) {
      if (rise_s < 0 && track.velocity[0] >= 1.8) {rise_s = time - 1;}
      start_error = std::max(start_error, intercept_error(track, time, x, 2));
    }
  }
  std::cout << "scenario=start fps=" << fps
            << " velocity_tau_s=" << config.velocity_filter_tau_s
            << " start_velocity90_ms=" << (rise_s < 0 ? -1 : rise_s * 1000)
            << " start_peak_intercept_error_deg=" << start_error << '\n';

  // Stationary target with alternating pose noise. Skip the first second.
  yolo_cpp::TrackManager stationary(config);
  double squared_error = 0;
  int count = 0;
  for (int frame = 0; frame < 3 * fps; ++frame) {
    const double time = frame / fps;
    const auto track = selected(stationary.update({observation(frame % 2 ? .015 : -.015, time)}, 1 / fps));
    const auto aim = solve(track, time);
    if (time > 1) {squared_error += aim.yaw * aim.yaw; ++count;}
  }
  std::cout << "scenario=stationary_noise fps=" << fps
            << " velocity_tau_s=" << config.velocity_filter_tau_s
            << " stationary_15mm_noise_aim_rms_deg=" << std::sqrt(squared_error / count) * kRadiansToDegrees
            << '\n';
}
}  // namespace

int main()
{
  try {
    std::cout << "# Observational synthetic benchmark; not physical validation or gain recommendations.\n"
              << "# range_z_m=4 frame_age_s=0.046 bullet_speed_mps=18 position_tau_s=0.035; -1 means threshold not reached.\n";
    for (double tau : {.1, .05, .035}) {
      yolo_cpp::TrackConfig config;
      config.position_filter_tau_s = .035;
      config.velocity_filter_tau_s = tau;
      for (double fps : {30., 60., 90., 120.}) {benchmark(config, fps);}
    }
  } catch (const std::exception & error) {
    std::cerr << "Benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
