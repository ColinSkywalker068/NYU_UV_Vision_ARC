#include "yolov8_ros_cpp/tracker.hpp"

#include <algorithm>
#include <cstring>

namespace yolov8_ros_cpp
{

namespace
{

void multiply_6x6(const double a[6][6], const double b[6][6], double out[6][6])
{
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      double sum = 0.0;
      for (int k = 0; k < 6; ++k) {
        sum += a[r][k] * b[k][c];
      }
      out[r][c] = sum;
    }
  }
}

void transpose_6x6(const double in[6][6], double out[6][6])
{
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      out[c][r] = in[r][c];
    }
  }
}

}  // namespace

ArmorPlateTracker::ArmorPlateTracker(
  double x, double y, double width, double height, double proc_noise_pos, double proc_noise_vel,
  double proc_noise_acc, double meas_noise, double init_cov)
: state_{x, y, 0.0, 0.0, 0.0, 0.0},
  q_diag_{proc_noise_pos, proc_noise_pos, proc_noise_vel, proc_noise_vel, proc_noise_acc,
    proc_noise_acc},
  r_(meas_noise),
  width_(width),
  height_(height)
{
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      p_[i][j] = (i == j) ? init_cov : 0.0;
    }
  }
}

std::pair<double, double> ArmorPlateTracker::predict(double dt)
{
  const double dt2 = 0.5 * dt * dt;
  double f[6][6] = {
    {1, 0, dt, 0, dt2, 0},
    {0, 1, 0, dt, 0, dt2},
    {0, 0, 1, 0, dt, 0},
    {0, 0, 0, 1, 0, dt},
    {0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 1}};

  std::array<double, 6> new_state{};
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      new_state[r] += f[r][c] * state_[c];
    }
  }
  state_ = new_state;

  double ft[6][6]{};
  double fp[6][6]{};
  double fpf_t[6][6]{};
  transpose_6x6(f, ft);
  multiply_6x6(f, p_, fp);
  multiply_6x6(fp, ft, fpf_t);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      p_[i][j] = fpf_t[i][j] + ((i == j) ? q_diag_[i] : 0.0);
    }
  }

  return {state_[0], state_[1]};
}

std::pair<double, double> ArmorPlateTracker::update(double x, double y, double width, double height)
{
  const double s00 = p_[0][0] + r_;
  const double s01 = p_[0][1];
  const double s10 = p_[1][0];
  const double s11 = p_[1][1] + r_;
  const double det = (s00 * s11) - (s01 * s10);
  if (det == 0.0) {
    return {state_[0], state_[1]};
  }

  const double inv00 = s11 / det;
  const double inv01 = -s01 / det;
  const double inv10 = -s10 / det;
  const double inv11 = s00 / det;

  double k[6][2]{};
  for (int i = 0; i < 6; ++i) {
    k[i][0] = (p_[i][0] * inv00) + (p_[i][1] * inv10);
    k[i][1] = (p_[i][0] * inv01) + (p_[i][1] * inv11);
  }

  const double innovation0 = x - state_[0];
  const double innovation1 = y - state_[1];
  for (int i = 0; i < 6; ++i) {
    state_[i] += (k[i][0] * innovation0) + (k[i][1] * innovation1);
  }

  double new_p[6][6]{};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      new_p[i][j] = p_[i][j] - (k[i][0] * p_[0][j]) - (k[i][1] * p_[1][j]);
    }
  }
  std::memcpy(p_, new_p, sizeof(p_));

  width_ = width;
  height_ = height;
  return {state_[0], state_[1]};
}

const std::array<double, 6> & ArmorPlateTracker::state() const
{
  return state_;
}

double ArmorPlateTracker::width() const
{
  return width_;
}

double ArmorPlateTracker::height() const
{
  return height_;
}

}  // namespace yolov8_ros_cpp
