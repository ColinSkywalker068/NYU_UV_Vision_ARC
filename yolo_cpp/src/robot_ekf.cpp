#include "yolo_cpp/robot_ekf.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace yolo_cpp
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
bool finite_observation(const RobotPlateObservation & o)
{
  return std::isfinite(o.position.x) && std::isfinite(o.position.y) &&
         std::isfinite(o.position.z) && std::isfinite(o.outward_normal.x) &&
         std::isfinite(o.outward_normal.z) &&
         std::hypot(o.outward_normal.x, o.outward_normal.z) > 1e-6;
}
double phase_of(const RobotPlateObservation & o)
{
  return std::atan2(o.outward_normal.x, -o.outward_normal.z);
}
}  // namespace

RobotEkf::RobotEkf(RobotEstimatorConfig config) : config_(config) { reset(); }

void RobotEkf::reset()
{
  state_ = {};
  x_ = {};
  covariance_ = Covariance::eye();
  seen_slots_ = 0;
  confident_rotation_updates_ = 0;
  multiple_plate_evidence_ = false;
}

cv::Vec4d RobotEkf::measurement(int slot, Jacobian * h) const
{
  const double alternate = config_.armor_count == 4 ? (slot % 2) : 0;
  const double r = x_[8] + alternate * x_[9];
  const double phase = x_[6] + slot * 2 * kPi / config_.armor_count;
  const double s = std::sin(phase), c = std::cos(phase);
  if (h) {
    *h = Jacobian::zeros();
    (*h)(0, 0) = (*h)(1, 2) = (*h)(2, 4) = (*h)(3, 6) = 1;
    (*h)(0, 6) = r * c;
    (*h)(2, 6) = r * s;
    (*h)(0, 8) = s;
    (*h)(2, 8) = -c;
    (*h)(0, 9) = alternate * s;
    (*h)(2, 9) = -alternate * c;
    (*h)(1, 10) = alternate;
  }
  return {x_[0] + r * s, x_[2] + alternate * x_[10], x_[4] - r * c, wrap_angle(phase)};
}

void RobotEkf::predict(double dt)
{
  auto f = Covariance::eye();
  auto q = Covariance::zeros();
  for (int i : {0, 2, 4, 6}) {
    f(i, i + 1) = dt;
    const double variance = i == 6 ? 100.0 : 9.0;
    q(i, i) = variance * std::pow(dt, 4) / 4;
    q(i, i + 1) = q(i + 1, i) = variance * std::pow(dt, 3) / 2;
    q(i + 1, i + 1) = variance * dt * dt;
  }
  for (int i = 8; i < 11; ++i) {
    q(i, i) = 1e-7 * dt;
  }
  x_ = f * x_;
  x_[6] = wrap_angle(x_[6]);
  covariance_ = f * covariance_ * f.t() + q;
}

void RobotEkf::synchronize(double stamp)
{
  state_.center = {x_[0], x_[2], x_[4]};
  state_.velocity = {x_[1], x_[3], x_[5]};
  state_.phase_rad = x_[6];
  state_.angular_velocity_rad_s = x_[7];
  state_.armor_radius_m = x_[8];
  state_.alternating_radius_m = x_[9];
  state_.alternating_height_m = x_[10];
  state_.phase_stddev_rad = std::sqrt(std::max(0.0, covariance_(6, 6)));
  state_.angular_speed_stddev_rad_s = std::sqrt(std::max(0.0, covariance_(7, 7)));
  state_.timestamp_s = stamp;
  state_.rotating = std::abs(x_[7]) > config_.rotating_threshold_rad_s;
}

void RobotEkf::update(const std::vector<RobotPlateObservation> & observations, double stamp)
{
  if (!std::isfinite(stamp) || (state_.initialized && stamp <= state_.timestamp_s)) {
    return;
  }
  if (state_.initialized && stamp - state_.last_observation_s > 0.4) {
    reset();
  }
  if (!state_.initialized) {
    const auto first = std::find_if(observations.begin(), observations.end(), finite_observation);
    if (first == observations.end()) {
      return;
    }
    const double phase = phase_of(*first);
    x_ = State{
      first->position.x - config_.armor_radius_m * std::sin(phase),
      0,
      first->position.y,
      0,
      first->position.z + config_.armor_radius_m * std::cos(phase),
      0,
      phase,
      0,
      config_.armor_radius_m,
      0,
      0};
    covariance_ = Covariance::zeros();
    const double variances[] = {0.04, 1, 0.04, 1, 0.04, 1, 0.25, 16, 0.0025, 0.0025, 0.01};
    for (int i = 0; i < 11; ++i) {
      covariance_(i, i) = variances[i];
    }
    state_.initialized = true;
    state_.timestamp_s = state_.last_observation_s = stamp;
  } else {
    predict(stamp - state_.timestamp_s);
  }

  std::set<int> accepted_slots;
  std::size_t accepted = 0;
  for (const auto & observation : observations) {
    if (!finite_observation(observation)) {
      ++state_.rejected_observations;
      continue;
    }
    const cv::Vec4d observed(
      observation.position.x, observation.position.y, observation.position.z,
      phase_of(observation));
    const double normal_sigma = observation.normal_reliable ? 0.10 : 0.45;
    const double range = std::hypot(observation.position.x, observation.position.z);
    const double position_sigma = std::clamp(0.006 + 0.001 * range * range, 0.01, 0.2);
    const auto noise = cv::Matx44d::diag(
      {position_sigma * position_sigma, position_sigma * position_sigma,
       position_sigma * position_sigma, normal_sigma * normal_sigma});
    int best_slot = -1;
    double best_distance = 18.5;
    Jacobian best_h;
    cv::Vec4d best_residual;
    cv::Matx44d best_inverse;
    for (int slot = 0; slot < config_.armor_count; ++slot) {
      if (accepted_slots.count(slot)) {
        continue;
      }
      Jacobian h;
      const auto expected = measurement(slot, &h);
      auto residual = observed - expected;
      residual[3] = wrap_angle(residual[3]);
      const double spatial_error = cv::norm(cv::Vec3d(residual[0], residual[1], residual[2]));
      if (spatial_error > config_.center_gate_m || std::abs(residual[3]) > kPi / 2) {
        continue;
      }
      bool invertible = false;
      const auto innovation = h * covariance_ * h.t() + noise;
      const auto inverse = innovation.inv(cv::DECOMP_CHOLESKY, &invertible);
      if (!invertible) {
        continue;
      }
      const double distance = residual.dot(inverse * residual);
      if (std::isfinite(distance) && distance < best_distance) {
        best_distance = distance;
        best_slot = slot;
        best_h = h;
        best_residual = residual;
        best_inverse = inverse;
      }
    }
    if (best_slot < 0) {
      ++state_.rejected_observations;
      continue;
    }
    const auto gain = covariance_ * best_h.t() * best_inverse;
    const State candidate = x_ + gain * best_residual;
    if (
      candidate[8] < 0.08 || candidate[8] > 0.45 || candidate[8] + candidate[9] < 0.08 ||
      candidate[8] + candidate[9] > 0.45 || std::abs(candidate[10]) > 0.3 ||
      std::abs(candidate[7]) > config_.max_angular_speed_rad_s) {
      ++state_.rejected_observations;
      continue;
    }
    x_ = candidate;
    x_[6] = wrap_angle(x_[6]);
    // Joseph form keeps covariance symmetric/positive under round-off.
    const auto correction = Covariance::eye() - gain * best_h;
    covariance_ = correction * covariance_ * correction.t() + gain * noise * gain.t();
    covariance_ = (covariance_ + covariance_.t()) * 0.5;
    accepted_slots.insert(best_slot);
    seen_slots_ |= 1U << best_slot;
    ++accepted;
  }
  synchronize(stamp);
  if (accepted == 0) {
    ++state_.coast_frames;
    if (state_.coast_frames > config_.coast_frames || stamp - state_.last_observation_s > 0.15) {
      state_.model_confirmed = false;
    }
    return;
  }
  state_.accepted_observations += accepted;
  state_.last_observation_s = stamp;
  state_.coast_frames = 0;
  ++state_.update_count;
  multiple_plate_evidence_ = multiple_plate_evidence_ || accepted_slots.size() >= 2;
  const bool significant_rotation =
    std::abs(x_[7]) > config_.rotating_threshold_rad_s + 2 * state_.angular_speed_stddev_rad_s;
  confident_rotation_updates_ = significant_rotation ? confident_rotation_updates_ + 1 : 0;
  state_.model_confirmed = state_.update_count >= config_.min_updates &&
                           (multiple_plate_evidence_ || confident_rotation_updates_ >= 8);
}
}  // namespace yolo_cpp
