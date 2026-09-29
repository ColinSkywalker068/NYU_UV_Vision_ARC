#ifndef YOLO_CPP__ROBOT_STATE_ESTIMATOR_HPP_
#define YOLO_CPP__ROBOT_STATE_ESTIMATOR_HPP_

#include <cstddef>
#include <vector>

namespace yolo_cpp
{

struct Vector3d
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct RobotPlateObservation
{
  Vector3d position;
  // Unit vector from the robot center toward this plate.
  Vector3d outward_normal;
  int track_id{-1};
};

struct RobotEstimatorConfig
{
  double armor_radius_m{0.26};
  int armor_count{4};
  double position_alpha{0.65};
  double position_beta{0.08};
  double phase_alpha{0.55};
  double phase_beta{0.06};
  double center_gate_m{0.45};
  double plate_cluster_gate_m{0.18};
  double max_angular_speed_rad_s{20.0};
  double rotating_threshold_rad_s{0.7};
  int min_updates{5};
  int coast_frames{8};
};

struct RobotEstimate
{
  bool initialized{false};
  bool model_confirmed{false};
  bool rotating{false};
  Vector3d center;
  Vector3d velocity;
  double phase_rad{0.0};
  double angular_velocity_rad_s{0.0};
  double timestamp_s{0.0};
  double last_observation_s{0.0};
  int update_count{0};
  int coast_frames{0};
  std::size_t accepted_observations{0};
  std::size_t rejected_observations{0};
};

struct FutureArmor
{
  Vector3d position;
  Vector3d outward_normal;
  int slot{0};
};

double wrap_angle(double angle);

/// Select the plate facing the camera at the requested prediction horizon.
FutureArmor select_future_armor(
  const RobotEstimate & state, double prediction_horizon_s, double armor_radius_m, int armor_count);

/// Tracks a shared robot center and rotation phase across individual armor
/// switches.
class RobotStateEstimator
{
public:
  explicit RobotStateEstimator(RobotEstimatorConfig config = RobotEstimatorConfig());

  void reset();
  void update(const std::vector<RobotPlateObservation> & observations, double timestamp_s);
  const RobotEstimate & state() const;
  RobotEstimate predicted(double timestamp_s) const;

private:
  RobotEstimatorConfig config_;
  RobotEstimate state_;
  bool robot_geometry_evidence_{false};
};

}  // namespace yolo_cpp

#endif
