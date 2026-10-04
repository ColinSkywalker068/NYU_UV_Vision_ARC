#ifndef YOLO_CPP__AIM_PREDICTION_HPP_
#define YOLO_CPP__AIM_PREDICTION_HPP_

#include "yolo_cpp/aim_geometry.hpp"
#include "yolo_cpp/robot_state_estimator.hpp"

namespace yolo_cpp
{
struct AimPredictionConfig
{
  double transport_delay_s{0.0};
  double actuator_delay_s{0.0};
  double max_frame_age_s{0.25};
  double max_measurement_age_s{0.15};
  double max_prediction_s{1.0};
  double yaw_speed_rad_s{1.0471975512};    // Sentry: 60 deg/s.
  double pitch_speed_rad_s{1.0471975512};  // Sentry: 60 deg/s.
  double yaw_accel_rad_s2{6.2831853072};
  double pitch_accel_rad_s2{4.1887902048};
  double plate_switch_margin{0.15};
  double bullet_speed_mps{15.0};
  double gravity_mps2{9.80665};
  bool compensate_gravity{true};
};

struct AimPredictionInput
{
  std::int64_t frame_stamp_ns{0};
  std::int64_t measurement_stamp_ns{0};
  std::int64_t now_ns{0};
  cv::Point3d position;
  cv::Point3d velocity;
  bool has_velocity{false};
  bool has_robot_state{false};
  RobotEstimate robot;
  double armor_radius_m{0.26};
  int armor_count{4};
  int previous_slot{-1};
  AttitudeSample current_attitude;
};

struct AimPredictionResult
{
  bool valid{false};
  bool reachable{false};
  const char * reason{"invalid_input"};
  double yaw{0.0};
  double pitch{0.0};
  double frame_age_s{0.0};
  double measurement_age_s{0.0};
  double horizon_s{0.0};
  double flight_time_s{0.0};
  double range_m{0.0};
  double minimum_slew_s{0.0};
  int selected_slot{-1};
};

struct AimAngularVelocity
{
  bool valid{false};
  double yaw_rad_s{0.0};
  double pitch_rad_s{0.0};
};

// Rates of the absolute ballistic aim point, never camera-relative pixel
// motion. Differentiate the same world-motion/ballistic model over 10ms, then
// require continuous measured identity/plate and bounded model/command
// innovations. Invalid/unavailable rates are zero; they do not invalidate
// position aiming.
class AimRateEstimator
{
public:
  AimAngularVelocity update(
    const AimPredictionInput & input, const AimPredictionConfig & config,
    const AimPredictionResult & solution, int identity, bool fresh_tracking);
  void reset() { have_previous_ = false; }

private:
  bool have_previous_{false};
  int identity_{-1};
  int slot_{-1};
  std::int64_t source_stamp_ns_{0};
  std::int64_t evaluation_stamp_ns_{0};
  double yaw_{0.0}, pitch_{0.0}, bullet_speed_{0.0};
  AimAngularVelocity previous_rate_;
};

// Production solver core, shared by ROS and deterministic offline tests.
// Positions and velocities are in fixed-orientation gimbal_inertial axes.
AimPredictionResult predict_aim(
  const AimPredictionInput & input, const AimPredictionConfig & config);
double minimum_slew_time(double displacement_rad, double max_speed, double max_acceleration);
}  // namespace yolo_cpp
#endif
