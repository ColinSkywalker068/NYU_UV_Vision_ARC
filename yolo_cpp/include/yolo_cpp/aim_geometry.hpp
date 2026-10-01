#ifndef YOLO_CPP__AIM_GEOMETRY_HPP_
#define YOLO_CPP__AIM_GEOMETRY_HPP_

#include <array>
#include <cstdint>
#include <deque>
#include <opencv2/core.hpp>
#include <optional>

namespace yolo_cpp
{

// Fixed orientation with optical-style axes: X right, Y down, Z forward at
// controller yaw=pitch=roll=0. Origin is the gimbal pivot; translation from
// chassis motion requires external odometry and is not inferred from attitude.
struct AttitudeSample
{
  std::int64_t stamp_ns{0};
  double yaw{0.0};
  double pitch{0.0};
  double roll{0.0};
};

class AttitudeHistory
{
public:
  bool insert(const AttitudeSample & sample);
  void clear();
  std::optional<AttitudeSample> at(
    std::int64_t stamp_ns, std::int64_t now_ns, double edge_tolerance_s = 0.03,
    double max_gap_s = 0.05, double max_age_s = 0.25) const;
  std::optional<AttitudeSample> latest(std::int64_t now_ns, double max_age_s = 0.25) const;

private:
  std::deque<AttitudeSample> samples_;
};

double angle_difference(double lhs, double rhs);
cv::Matx33d camera_to_inertial_rotation(const AttitudeSample & attitude);
cv::Matx33d quaternion_rotation(const std::array<double, 4> & xyzw);
std::array<double, 4> rotation_quaternion(const cv::Matx33d & rotation);
cv::Point3d rotate_point(const cv::Matx33d & rotation, const cv::Point3d & point);
bool finite_point(const cv::Point3d & point);

}  // namespace yolo_cpp
#endif
