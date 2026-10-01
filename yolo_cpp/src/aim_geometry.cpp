#include "yolo_cpp/aim_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace yolo_cpp
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
bool finite_attitude(const AttitudeSample & s)
{
  return s.stamp_ns > 0 && std::isfinite(s.yaw) && std::isfinite(s.pitch) && std::isfinite(s.roll);
}
}  // namespace

double angle_difference(double lhs, double rhs) { return std::remainder(lhs - rhs, 2.0 * kPi); }

bool AttitudeHistory::insert(const AttitudeSample & sample)
{
  if (!finite_attitude(sample)) {
    clear();
    return false;
  }
  // Never splice delayed/out-of-order telemetry into the current time history.
  if (!samples_.empty() && sample.stamp_ns <= samples_.back().stamp_ns) {
    return false;
  }
  samples_.push_back(sample);
  while (samples_.size() > 2 && samples_.front().stamp_ns < sample.stamp_ns - 2000000000LL) {
    samples_.pop_front();
  }
  return true;
}

void AttitudeHistory::clear() { samples_.clear(); }

std::optional<AttitudeSample> AttitudeHistory::latest(std::int64_t now_ns, double max_age_s) const
{
  if (
    samples_.empty() || now_ns < samples_.back().stamp_ns ||
    now_ns - samples_.back().stamp_ns > max_age_s * 1e9) {
    return std::nullopt;
  }
  return samples_.back();
}

std::optional<AttitudeSample> AttitudeHistory::at(
  std::int64_t stamp_ns, std::int64_t now_ns, double edge_tolerance_s, double max_gap_s,
  double max_age_s) const
{
  if (stamp_ns <= 0 || stamp_ns > now_ns || !latest(now_ns, max_age_s)) {
    return std::nullopt;
  }
  const auto edge = [stamp_ns,
                     edge_tolerance_s](const AttitudeSample & s) -> std::optional<AttitudeSample> {
    if (std::abs(static_cast<double>(stamp_ns - s.stamp_ns)) > edge_tolerance_s * 1e9) {
      return std::nullopt;
    }
    auto result = s;
    result.stamp_ns = stamp_ns;
    return result;
  };
  if (stamp_ns <= samples_.front().stamp_ns) {
    return edge(samples_.front());
  }
  if (stamp_ns >= samples_.back().stamp_ns) {
    return edge(samples_.back());
  }
  for (std::size_t i = 1; i < samples_.size(); ++i) {
    const auto & a = samples_[i - 1];
    const auto & b = samples_[i];
    if (stamp_ns > b.stamp_ns) {
      continue;
    }
    if (stamp_ns == b.stamp_ns) {
      return b;
    }
    // A bracket alone is insufficient: a telemetry outage is not an observation.
    const double span = static_cast<double>(b.stamp_ns - a.stamp_ns);
    if (span > max_gap_s * 1e9) {
      return std::nullopt;
    }
    const double f = static_cast<double>(stamp_ns - a.stamp_ns) / span;
    return AttitudeSample{
      stamp_ns, angle_difference(a.yaw + f * angle_difference(b.yaw, a.yaw), 0.0),
      a.pitch + f * (b.pitch - a.pitch),
      angle_difference(a.roll + f * angle_difference(b.roll, a.roll), 0.0)};
  }
  return std::nullopt;
}

cv::Matx33d camera_to_inertial_rotation(const AttitudeSample & a)
{
  const double cy = std::cos(a.yaw), sy = std::sin(a.yaw);
  const double cp = std::cos(a.pitch), sp = std::sin(a.pitch);
  const double cr = std::cos(a.roll), sr = std::sin(a.roll);
  const cv::Matx33d yaw(cy, 0, sy, 0, 1, 0, -sy, 0, cy);
  const cv::Matx33d pitch(1, 0, 0, 0, cp, -sp, 0, sp, cp);
  const cv::Matx33d roll(cr, -sr, 0, sr, cr, 0, 0, 0, 1);
  return yaw * pitch * roll;
}

cv::Matx33d quaternion_rotation(const std::array<double, 4> & q)
{
  double n = 0.0;
  for (const double v : q) {
    n += v * v;
  }
  if (!std::isfinite(n) || n < 1e-12) {
    return cv::Matx33d::all(std::numeric_limits<double>::quiet_NaN());
  }
  const double x = q[0] / std::sqrt(n), y = q[1] / std::sqrt(n);
  const double z = q[2] / std::sqrt(n), w = q[3] / std::sqrt(n);
  return {1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w),
          2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
          2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)};
}

std::array<double, 4> rotation_quaternion(const cv::Matx33d & m)
{
  std::array<double, 4> q{};
  const double trace = m(0, 0) + m(1, 1) + m(2, 2);
  if (trace > 0.0) {
    const double s = 2.0 * std::sqrt(trace + 1.0);
    q = {(m(2, 1) - m(1, 2)) / s, (m(0, 2) - m(2, 0)) / s, (m(1, 0) - m(0, 1)) / s, s / 4.0};
  } else {
    int i = m(1, 1) > m(0, 0) ? 1 : 0;
    if (m(2, 2) > m(i, i)) {
      i = 2;
    }
    const int j = (i + 1) % 3, k = (i + 2) % 3;
    const double s = 2.0 * std::sqrt(std::max(0.0, 1.0 + m(i, i) - m(j, j) - m(k, k)));
    q[i] = s / 4.0;
    q[j] = (m(j, i) + m(i, j)) / s;
    q[k] = (m(k, i) + m(i, k)) / s;
    q[3] = (m(k, j) - m(j, k)) / s;
  }
  return q;
}

cv::Point3d rotate_point(const cv::Matx33d & r, const cv::Point3d & p)
{
  const cv::Vec3d v = r * cv::Vec3d(p.x, p.y, p.z);
  return {v[0], v[1], v[2]};
}

bool finite_point(const cv::Point3d & p)
{
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
}  // namespace yolo_cpp
