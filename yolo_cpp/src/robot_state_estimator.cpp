#include "yolo_cpp/robot_state_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace yolo_cpp
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

double norm(const Vector3d & value)
{
  return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

Vector3d add(const Vector3d & lhs, const Vector3d & rhs)
{
  return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

Vector3d subtract(const Vector3d & lhs, const Vector3d & rhs)
{
  return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Vector3d scale(const Vector3d & value, double factor)
{
  return {value.x * factor, value.y * factor, value.z * factor};
}

Vector3d inferred_center(const RobotPlateObservation & observation, double radius)
{
  const double length = std::max(1e-9, norm(observation.outward_normal));
  return subtract(observation.position, scale(observation.outward_normal, radius / length));
}

double observation_phase(const RobotPlateObservation & observation)
{
  return std::atan2(observation.outward_normal.x, -observation.outward_normal.z);
}
}  // namespace

double wrap_angle(double angle) { return std::remainder(angle, 2.0 * kPi); }

FutureArmor select_future_armor(
  const RobotEstimate & state, double prediction_horizon_s, double armor_radius_m, int armor_count)
{
  FutureArmor best;
  double best_facing = -std::numeric_limits<double>::infinity();
  const double horizon = std::max(0.0, prediction_horizon_s);
  const Vector3d center = add(state.center, scale(state.velocity, horizon));
  const double phase = state.phase_rad + state.angular_velocity_rad_s * horizon;
  const int count = std::max(1, armor_count);
  for (int slot = 0; slot < count; ++slot) {
    const double angle = phase + 2.0 * kPi * static_cast<double>(slot) / count;
    const Vector3d normal{std::sin(angle), 0.0, -std::cos(angle)};
    const Vector3d position = add(center, scale(normal, armor_radius_m));
    // The camera is the origin. A visible plate's outward normal points toward
    // it.
    const Vector3d to_camera = scale(position, -1.0);
    const double facing =
      (normal.x * to_camera.x + normal.y * to_camera.y + normal.z * to_camera.z) /
      std::max(1e-9, norm(to_camera));
    if (facing > best_facing) {
      best_facing = facing;
      best = {position, normal, slot};
    }
  }
  return best;
}

RobotStateEstimator::RobotStateEstimator(RobotEstimatorConfig config) : config_(config)
{
  config_.armor_count = std::max(1, config_.armor_count);
  config_.min_updates = std::max(1, config_.min_updates);
  config_.coast_frames = std::max(0, config_.coast_frames);
}

void RobotStateEstimator::reset()
{
  state_ = RobotEstimate();
  robot_geometry_evidence_ = false;
}

void RobotStateEstimator::update(
  const std::vector<RobotPlateObservation> & observations, double timestamp_s)
{
  if (state_.initialized && timestamp_s <= state_.timestamp_s) {
    timestamp_s = state_.timestamp_s + 1e-4;
  }

  RobotEstimate prior = predicted(timestamp_s);
  if (observations.empty()) {
    if (!state_.initialized) {
      return;
    }
    state_ = prior;
    ++state_.coast_frames;
    if (state_.coast_frames > config_.coast_frames) {
      state_.model_confirmed = false;
    }
    return;
  }

  struct Candidate
  {
    const RobotPlateObservation * observation;
    Vector3d center;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(observations.size());
  std::size_t rejected = 0;
  for (const auto & observation : observations) {
    const Vector3d candidate = inferred_center(observation, config_.armor_radius_m);
    if (state_.initialized && norm(subtract(candidate, prior.center)) > config_.center_gate_m) {
      ++rejected;
      continue;
    }
    candidates.push_back({&observation, candidate});
  }
  if (candidates.empty()) {
    if (state_.initialized) {
      state_ = prior;
      ++state_.coast_frames;
      state_.rejected_observations += rejected;
    }
    return;
  }

  const auto anchor_it = std::min_element(
    candidates.begin(), candidates.end(), [&prior](const auto & lhs, const auto & rhs) {
      return norm(subtract(lhs.center, prior.center)) < norm(subtract(rhs.center, prior.center));
    });
  const Vector3d anchor_center = anchor_it->center;
  std::vector<const RobotPlateObservation *> accepted;
  accepted.reserve(candidates.size());
  for (const auto & candidate : candidates) {
    if (norm(subtract(candidate.center, anchor_center)) > config_.plate_cluster_gate_m) {
      ++rejected;
      continue;
    }
    accepted.push_back(candidate.observation);
  }

  Vector3d measured_center;
  for (const auto * observation : accepted) {
    measured_center = add(measured_center, inferred_center(*observation, config_.armor_radius_m));
  }
  measured_center = scale(measured_center, 1.0 / static_cast<double>(accepted.size()));

  if (!state_.initialized) {
    state_.initialized = true;
    state_.center = measured_center;
    state_.phase_rad = observation_phase(*accepted.front());
    state_.timestamp_s = timestamp_s;
    state_.last_observation_s = timestamp_s;
    state_.update_count = 1;
    state_.accepted_observations = accepted.size();
    state_.rejected_observations = rejected;
    return;
  }

  const double dt = std::max(1e-4, timestamp_s - state_.timestamp_s);
  const Vector3d position_residual = subtract(measured_center, prior.center);
  state_ = prior;
  state_.center = add(prior.center, scale(position_residual, config_.position_alpha));
  state_.velocity = add(prior.velocity, scale(position_residual, config_.position_beta / dt));

  double phase_residual_sum = 0.0;
  for (const auto * observation : accepted) {
    const double plate_phase = observation_phase(*observation);
    double best_residual = std::numeric_limits<double>::infinity();
    for (int slot = 0; slot < config_.armor_count; ++slot) {
      const double base_phase =
        plate_phase - 2.0 * kPi * static_cast<double>(slot) / config_.armor_count;
      const double residual = wrap_angle(base_phase - prior.phase_rad);
      if (std::abs(residual) < std::abs(best_residual)) {
        best_residual = residual;
      }
    }
    phase_residual_sum += best_residual;
  }
  const double phase_residual = phase_residual_sum / static_cast<double>(accepted.size());
  state_.phase_rad = wrap_angle(prior.phase_rad + config_.phase_alpha * phase_residual);
  state_.angular_velocity_rad_s = std::clamp(
    prior.angular_velocity_rad_s + config_.phase_beta * phase_residual / dt,
    -config_.max_angular_speed_rad_s, config_.max_angular_speed_rad_s);
  state_.rotating = std::abs(state_.angular_velocity_rad_s) >= config_.rotating_threshold_rad_s;
  state_.timestamp_s = timestamp_s;
  state_.last_observation_s = timestamp_s;
  state_.coast_frames = 0;
  ++state_.update_count;
  state_.accepted_observations += accepted.size();
  state_.rejected_observations += rejected;

  robot_geometry_evidence_ = robot_geometry_evidence_ || accepted.size() >= 2 || state_.rotating;
  state_.model_confirmed = state_.update_count >= config_.min_updates && robot_geometry_evidence_;
}

const RobotEstimate & RobotStateEstimator::state() const { return state_; }

RobotEstimate RobotStateEstimator::predicted(double timestamp_s) const
{
  RobotEstimate result = state_;
  if (!result.initialized || timestamp_s <= result.timestamp_s) {
    return result;
  }
  const double dt = timestamp_s - result.timestamp_s;
  result.center = add(result.center, scale(result.velocity, dt));
  result.phase_rad = wrap_angle(result.phase_rad + result.angular_velocity_rad_s * dt);
  result.timestamp_s = timestamp_s;
  return result;
}

}  // namespace yolo_cpp
