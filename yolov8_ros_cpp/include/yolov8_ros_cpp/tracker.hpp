#ifndef YOLOV8_ROS_CPP__TRACKER_HPP_
#define YOLOV8_ROS_CPP__TRACKER_HPP_

#include <array>
#include <optional>
#include <utility>

namespace yolov8_ros_cpp
{

class ArmorPlateTracker
{
public:
  ArmorPlateTracker(
    double x, double y, double width, double height, double proc_noise_pos = 0.01,
    double proc_noise_vel = 0.05, double proc_noise_acc = 0.1, double meas_noise = 0.15,
    double init_cov = 0.1);

  std::pair<double, double> predict(double dt);
  std::pair<double, double> update(double x, double y, double width, double height);

  const std::array<double, 6> & state() const;
  double width() const;
  double height() const;

private:
  std::array<double, 6> state_;
  std::array<double, 6> q_diag_;
  double r_{0.15};
  double p_[6][6]{};
  double width_{0.0};
  double height_{0.0};
};

struct TrackerWrapper
{
  explicit TrackerWrapper(ArmorPlateTracker tracker_in) : tracker(std::move(tracker_in)) {}
  ArmorPlateTracker tracker;
  int lost_frames{0};
  std::optional<float> obb_angle;
};

}  // namespace yolov8_ros_cpp

#endif
