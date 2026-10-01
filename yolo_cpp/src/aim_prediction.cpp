#include "yolo_cpp/aim_prediction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace yolo_cpp
{
namespace
{
constexpr double kPi = 3.14159265358979323846;

FutureArmor select_plate(const AimPredictionInput & in, double horizon, double margin)
{
  FutureArmor best;
  FutureArmor previous;
  double best_score = -std::numeric_limits<double>::infinity();
  double previous_score = best_score;
  const auto & r = in.robot;
  const cv::Point3d center(
    r.center.x + r.velocity.x * horizon, r.center.y + r.velocity.y * horizon,
    r.center.z + r.velocity.z * horizon);
  for (int slot = 0; slot < in.armor_count; ++slot) {
    const double phase =
      r.phase_rad + r.angular_velocity_rad_s * horizon + 2 * kPi * slot / in.armor_count;
    const cv::Point3d n(std::sin(phase), 0, -std::cos(phase));
    const bool alternate = in.armor_count == 4 && slot % 2;
    auto p = center + n * (in.armor_radius_m + (alternate ? r.alternating_radius_m : 0));
    if (alternate) {
      p.y += r.alternating_height_m;
    }
    const double score = -n.dot(p) / std::max(1e-9, cv::norm(p));
    const FutureArmor armor{{p.x, p.y, p.z}, {n.x, n.y, n.z}, slot};
    if (score > best_score) {
      best_score = score;
      best = armor;
    }
    if (slot == in.previous_slot) {
      previous_score = score;
      previous = armor;
    }
  }
  // Never retain a back-facing plate merely to avoid a switch.
  return previous_score > 0 && best_score - previous_score < margin ? previous : best;
}
}  // namespace

double minimum_slew_time(double displacement_rad, double max_speed, double max_acceleration)
{
  if (
    !std::isfinite(displacement_rad) || !std::isfinite(max_speed) ||
    !std::isfinite(max_acceleration) || max_speed <= 0 || max_acceleration <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  const double distance = std::abs(displacement_rad);
  const double ramp_distance = max_speed * max_speed / max_acceleration;
  return distance <= ramp_distance ? 2 * std::sqrt(distance / max_acceleration)
                                   : distance / max_speed + max_speed / max_acceleration;
}

AimPredictionResult predict_aim(const AimPredictionInput & in, const AimPredictionConfig & c)
{
  AimPredictionResult out;
  if (
    in.frame_stamp_ns <= 0 || in.measurement_stamp_ns <= 0 ||
    in.measurement_stamp_ns > in.frame_stamp_ns || in.frame_stamp_ns > in.now_ns ||
    !finite_point(in.position) || !finite_point(in.velocity) ||
    !std::isfinite(c.bullet_speed_mps) || c.bullet_speed_mps <= 0 ||
    !std::isfinite(c.transport_delay_s) || c.transport_delay_s < 0 ||
    !std::isfinite(c.actuator_delay_s) || c.actuator_delay_s < 0 ||
    !std::isfinite(c.max_frame_age_s) || c.max_frame_age_s <= 0 ||
    !std::isfinite(c.max_measurement_age_s) || c.max_measurement_age_s <= 0 ||
    !std::isfinite(c.max_prediction_s) || c.max_prediction_s <= 0 ||
    !std::isfinite(c.gravity_mps2) || c.gravity_mps2 < 0 || !std::isfinite(c.plate_switch_margin) ||
    c.plate_switch_margin < 0 || c.plate_switch_margin > 2 || !std::isfinite(c.yaw_speed_rad_s) ||
    c.yaw_speed_rad_s <= 0 || !std::isfinite(c.pitch_speed_rad_s) || c.pitch_speed_rad_s <= 0 ||
    !std::isfinite(c.yaw_accel_rad_s2) || c.yaw_accel_rad_s2 <= 0 ||
    !std::isfinite(c.pitch_accel_rad_s2) || c.pitch_accel_rad_s2 <= 0 ||
    !std::isfinite(in.current_attitude.yaw) || !std::isfinite(in.current_attitude.pitch)) {
    return out;
  }
  out.frame_age_s = (in.now_ns - in.frame_stamp_ns) * 1e-9;
  out.measurement_age_s = (in.now_ns - in.measurement_stamp_ns) * 1e-9;
  if (out.frame_age_s > c.max_frame_age_s || out.measurement_age_s > c.max_measurement_age_s) {
    out.reason = "stale_measurement";
    return out;
  }
  if (
    in.has_robot_state &&
    (in.armor_count < 1 || in.armor_count > 8 || !std::isfinite(in.armor_radius_m) ||
     in.armor_radius_m <= 0 ||
     !finite_point({in.robot.center.x, in.robot.center.y, in.robot.center.z}) ||
     !finite_point({in.robot.velocity.x, in.robot.velocity.y, in.robot.velocity.z}) ||
     !std::isfinite(in.robot.phase_rad) || !std::isfinite(in.robot.angular_velocity_rad_s) ||
     !std::isfinite(in.robot.alternating_radius_m) ||
     !std::isfinite(in.robot.alternating_height_m) ||
     in.armor_radius_m + in.robot.alternating_radius_m <= 0)) {
    return out;
  }

  cv::Point3d p = in.position;
  const double delay = out.frame_age_s + c.transport_delay_s + c.actuator_delay_s;
  for (int iteration = 0; iteration < 5; ++iteration) {
    out.flight_time_s = cv::norm(p) / c.bullet_speed_mps;
    out.horizon_s = delay + out.flight_time_s;
    if (out.horizon_s > c.max_prediction_s) {
      out.reason = "prediction_horizon_exceeded";
      return out;
    }
    if (in.has_robot_state) {
      const auto plate = select_plate(in, out.horizon_s, c.plate_switch_margin);
      p = {plate.position.x, plate.position.y, plate.position.z};
      out.selected_slot = plate.slot;
    } else {
      p = in.position + (in.has_velocity ? in.velocity * out.horizon_s : cv::Point3d());
    }
  }
  if (!finite_point(p) || cv::norm(p) < 0.05) {
    return out;
  }
  out.range_m = cv::norm(p);
  out.flight_time_s = out.range_m / c.bullet_speed_mps;
  const double drop =
    c.compensate_gravity ? 0.5 * c.gravity_mps2 * out.flight_time_s * out.flight_time_s : 0;
  out.yaw = std::atan2(p.x, p.z);
  out.pitch = std::atan2(drop - p.y, std::hypot(p.x, p.z));
  out.valid = std::isfinite(out.yaw) && std::isfinite(out.pitch);
  out.minimum_slew_s = std::max(
    minimum_slew_time(
      angle_difference(out.yaw, in.current_attitude.yaw), c.yaw_speed_rad_s, c.yaw_accel_rad_s2),
    minimum_slew_time(
      out.pitch - in.current_attitude.pitch, c.pitch_speed_rad_s, c.pitch_accel_rad_s2));
  // Diagnostic rest-to-rest bound; does not promise measured motor response.
  out.reachable = out.minimum_slew_s <= c.actuator_delay_s + c.transport_delay_s;
  out.reason = out.valid ? "ok" : "nonfinite_solution";
  return out;
}
}  // namespace yolo_cpp
