#ifndef YOLO_CPP__ROBOT_EKF_HPP_
#define YOLO_CPP__ROBOT_EKF_HPP_

#include <opencv2/core.hpp>

#include "yolo_cpp/robot_state_estimator.hpp"

namespace yolo_cpp
{
// Center/phase/alternating-geometry state inspired by Tongji sp_vision_25 (MIT),
// adapted to our X-right, Y-down, Z-forward frame. No upstream runtime required.
class RobotEkf
{
public:
  explicit RobotEkf(RobotEstimatorConfig config);
  void reset();
  void update(const std::vector<RobotPlateObservation> & observations, double stamp);
  const RobotEstimate & state() const { return state_; }

private:
  // [cx,vx,cy,vy,cz,vz,phase,omega,radius,alternating_radius,alternating_height]
  using State = cv::Vec<double, 11>;
  using Covariance = cv::Matx<double, 11, 11>;
  using Jacobian = cv::Matx<double, 4, 11>;
  cv::Vec4d measurement(int slot, Jacobian * jacobian) const;
  void predict(double dt);
  void synchronize(double stamp);
  RobotEstimatorConfig config_;
  State x_{};
  Covariance covariance_{Covariance::eye()};
  RobotEstimate state_;
  unsigned seen_slots_{0};
  int confident_rotation_updates_{0};
  bool multiple_plate_evidence_{false};
};
}  // namespace yolo_cpp
#endif
