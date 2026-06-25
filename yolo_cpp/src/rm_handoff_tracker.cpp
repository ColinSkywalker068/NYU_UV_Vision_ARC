#include "yolo_cpp/rm_handoff_tracker.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace yolo_cpp {

namespace {

double shortest_angular_distance(double from, double to) {
  return std::atan2(std::sin(to - from), std::cos(to - from));
}

} // namespace

RmHandoffEkf::RmHandoffEkf(const VecVecFunc &f, const VecVecFunc &h,
                           const VecMatFunc &j_f, const VecMatFunc &j_h,
                           const VoidMatFunc &u_q, const VecMatFunc &u_r,
                           const Eigen::MatrixXd &p0)
    : f_(f), h_(h), jacobian_f_(j_f), jacobian_h_(j_h), update_q_(u_q),
      update_r_(u_r), p_post_(p0),
      identity_(Eigen::MatrixXd::Identity(p0.rows(), p0.rows())),
      x_pri_(p0.rows()), x_post_(p0.rows()) {}

void RmHandoffEkf::set_state(const Eigen::VectorXd &x0) { x_post_ = x0; }

Eigen::VectorXd RmHandoffEkf::predict() {
  const Eigen::MatrixXd f = jacobian_f_(x_post_);
  x_pri_ = f_(x_post_);
  p_pri_ = f * p_post_ * f.transpose() + update_q_();

  x_post_ = x_pri_;
  p_post_ = p_pri_;

  return x_pri_;
}

Eigen::VectorXd RmHandoffEkf::update(const Eigen::VectorXd &z) {
  const Eigen::MatrixXd h = jacobian_h_(x_pri_);
  const Eigen::MatrixXd r = update_r_(z);
  const Eigen::MatrixXd k =
      p_pri_ * h.transpose() * (h * p_pri_ * h.transpose() + r).inverse();

  x_post_ = x_pri_ + k * (z - h_(x_pri_));
  p_post_ = (identity_ - k * h) * p_pri_;

  return x_post_;
}

RmHandoffTracker::RmHandoffTracker(double max_match_distance,
                                   double max_match_yaw_diff)
    : max_match_distance_(max_match_distance),
      max_match_yaw_diff_(max_match_yaw_diff) {}

void RmHandoffTracker::configure_ekf(double dt, double sigma2_q_xyz,
                                     double sigma2_q_yaw, double sigma2_q_r,
                                     double r_xyz_factor, double r_yaw) {
  dt_ = dt;
  sigma2_q_xyz_ = sigma2_q_xyz;
  sigma2_q_yaw_ = sigma2_q_yaw;
  sigma2_q_r_ = sigma2_q_r;
  r_xyz_factor_ = r_xyz_factor;
  r_yaw_ = r_yaw;

  auto f = [this](const Eigen::VectorXd &x) {
    Eigen::VectorXd x_new = x;
    x_new(0) += x(1) * dt_;
    x_new(2) += x(3) * dt_;
    x_new(4) += x(5) * dt_;
    x_new(6) += x(7) * dt_;
    return x_new;
  };

  auto j_f = [this](const Eigen::VectorXd &) {
    Eigen::MatrixXd f = Eigen::MatrixXd::Identity(9, 9);
    f(0, 1) = dt_;
    f(2, 3) = dt_;
    f(4, 5) = dt_;
    f(6, 7) = dt_;
    return f;
  };

  auto h = [](const Eigen::VectorXd &x) {
    Eigen::VectorXd z(4);
    const double xc = x(0);
    const double yc = x(2);
    const double yaw = x(6);
    const double r = x(8);
    z(0) = xc - r * std::cos(yaw);
    z(1) = yc - r * std::sin(yaw);
    z(2) = x(4);
    z(3) = x(6);
    return z;
  };

  auto j_h = [](const Eigen::VectorXd &x) {
    Eigen::MatrixXd h(4, 9);
    const double yaw = x(6);
    const double r = x(8);
    h << 1, 0, 0, 0, 0, 0, r * std::sin(yaw), 0, -std::cos(yaw), 0, 0, 1, 0, 0,
        0, -r * std::cos(yaw), 0, -std::sin(yaw), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0;
    return h;
  };

  auto u_q = [this]() {
    Eigen::MatrixXd q(9, 9);
    const double t = dt_;
    const double x = sigma2_q_xyz_;
    const double y = sigma2_q_yaw_;
    const double r = sigma2_q_r_;
    const double q_x_x = std::pow(t, 4) / 4.0 * x;
    const double q_x_vx = std::pow(t, 3) / 2.0 * x;
    const double q_vx_vx = std::pow(t, 2) * x;
    const double q_y_y = std::pow(t, 4) / 4.0 * y;
    const double q_y_vy = std::pow(t, 3) / 2.0 * x;
    const double q_vy_vy = std::pow(t, 2) * y;
    const double q_r = std::pow(t, 4) / 4.0 * r;
    q << q_x_x, q_x_vx, 0, 0, 0, 0, 0, 0, 0, q_x_vx, q_vx_vx, 0, 0, 0, 0, 0, 0,
        0, 0, 0, q_x_x, q_x_vx, 0, 0, 0, 0, 0, 0, 0, q_x_vx, q_vx_vx, 0, 0, 0,
        0, 0, 0, 0, 0, 0, q_x_x, q_x_vx, 0, 0, 0, 0, 0, 0, 0, q_x_vx, q_vx_vx,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, q_y_y, q_y_vy, 0, 0, 0, 0, 0, 0, 0,
        q_y_vy, q_vy_vy, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, q_r;
    return q;
  };

  auto u_r = [this](const Eigen::VectorXd &z) {
    Eigen::DiagonalMatrix<double, 4> r;
    r.diagonal() << std::abs(r_xyz_factor_ * z[0]),
        std::abs(r_xyz_factor_ * z[1]), std::abs(r_xyz_factor_ * z[2]), r_yaw_;
    return Eigen::MatrixXd(r);
  };

  Eigen::DiagonalMatrix<double, 9> p0;
  p0.setIdentity();
  ekf = RmHandoffEkf{f, h, j_f, j_h, u_q, u_r, p0};
}

void RmHandoffTracker::set_dt(double dt) { dt_ = dt; }

void RmHandoffTracker::init(const Armors &armors_msg) {
  if (armors_msg.armors.empty()) {
    return;
  }

  double min_distance = DBL_MAX;
  tracked_armor = armors_msg.armors.front();
  for (const auto &armor : armors_msg.armors) {
    if (armor.distance_to_image_center < min_distance) {
      min_distance = armor.distance_to_image_center;
      tracked_armor = armor;
    }
  }

  init_ekf(tracked_armor);
  tracked_id = armor_id(tracked_armor);
  tracker_state = kDetecting;
  update_armors_num(tracked_armor);
}

void RmHandoffTracker::update(const Armors &armors_msg) {
  Eigen::VectorXd ekf_prediction = ekf.predict();

  bool matched = false;
  target_state = ekf_prediction;

  if (!armors_msg.armors.empty()) {
    Armor same_id_armor;
    int same_id_armors_count = 0;
    const auto predicted_position =
        get_armor_position_from_state(ekf_prediction);
    double min_position_diff = DBL_MAX;
    double yaw_diff = DBL_MAX;

    for (const auto &armor : armors_msg.armors) {
      if (armor_id(armor) == tracked_id) {
        same_id_armor = armor;
        ++same_id_armors_count;
        const auto &p = armor.pose.position;
        const Eigen::Vector3d position_vec(p.x, p.y, p.z);
        const double position_diff = (predicted_position - position_vec).norm();
        if (position_diff < min_position_diff) {
          min_position_diff = position_diff;
          yaw_diff = std::abs(orientation_to_yaw(armor.pose.orientation) -
                              ekf_prediction(6));
          tracked_armor = armor;
        }
      }
    }

    info_position_diff = min_position_diff;
    info_yaw_diff = yaw_diff;

    if (min_position_diff < max_match_distance_ &&
        yaw_diff < max_match_yaw_diff_) {
      matched = true;
      const auto &p = tracked_armor.pose.position;
      const double measured_yaw =
          orientation_to_yaw(tracked_armor.pose.orientation);
      measurement = Eigen::Vector4d(p.x, p.y, p.z, measured_yaw);
      target_state = ekf.update(measurement);
    } else if (same_id_armors_count == 1 && yaw_diff > max_match_yaw_diff_) {
      handle_armor_jump(same_id_armor);
    }
  }

  if (target_state(8) < 0.12) {
    target_state(8) = 0.12;
    ekf.set_state(target_state);
  } else if (target_state(8) > 0.4) {
    target_state(8) = 0.4;
    ekf.set_state(target_state);
  }

  if (tracker_state == kDetecting) {
    if (matched) {
      ++detect_count_;
      if (detect_count_ > tracking_thres) {
        detect_count_ = 0;
        tracker_state = kTracking;
      }
    } else {
      detect_count_ = 0;
      tracker_state = kLost;
    }
  } else if (tracker_state == kTracking) {
    if (!matched) {
      tracker_state = kTempLost;
      ++lost_count_;
    }
  } else if (tracker_state == kTempLost) {
    if (!matched) {
      ++lost_count_;
      if (lost_count_ > lost_thres) {
        lost_count_ = 0;
        tracker_state = kLost;
      }
    } else {
      tracker_state = kTracking;
      lost_count_ = 0;
    }
  }
}

std::string RmHandoffTracker::armor_id(const Armor &armor) {
  return armor.number;
}

std::string RmHandoffTracker::armor_type(const Armor &armor) {
  return armor.type;
}

void RmHandoffTracker::init_ekf(const Armor &armor) {
  const double xa = armor.pose.position.x;
  const double ya = armor.pose.position.y;
  const double za = armor.pose.position.z;
  last_yaw_ = 0.0;
  const double yaw = orientation_to_yaw(armor.pose.orientation);

  target_state = Eigen::VectorXd::Zero(9);
  const double r = 0.26;
  const double xc = xa + r * std::cos(yaw);
  const double yc = ya + r * std::sin(yaw);
  dz = 0.0;
  another_r = r;
  target_state << xc, 0.0, yc, 0.0, za, 0.0, yaw, 0.0, r;

  ekf.set_state(target_state);
}

void RmHandoffTracker::update_armors_num(const Armor &armor) {
  if (armor_type(armor) == "large" &&
      (tracked_id == "3" || tracked_id == "4" || tracked_id == "5")) {
    tracked_armors_num = RmHandoffArmorsNum::kBalance2;
  } else if (tracked_id == "outpost") {
    tracked_armors_num = RmHandoffArmorsNum::kOutpost3;
  } else {
    tracked_armors_num = RmHandoffArmorsNum::kNormal4;
  }
}

void RmHandoffTracker::handle_armor_jump(const Armor &armor) {
  const double yaw = orientation_to_yaw(armor.pose.orientation);
  target_state(6) = yaw;
  update_armors_num(armor);

  if (tracked_armors_num == RmHandoffArmorsNum::kNormal4) {
    dz = target_state(4) - armor.pose.position.z;
    target_state(4) = armor.pose.position.z;
    std::swap(target_state(8), another_r);
  }

  const auto &p = armor.pose.position;
  const Eigen::Vector3d current_p(p.x, p.y, p.z);
  const Eigen::Vector3d infer_p = get_armor_position_from_state(target_state);
  if ((current_p - infer_p).norm() > max_match_distance_) {
    const double r = target_state(8);
    target_state(0) = p.x + r * std::cos(yaw);
    target_state(1) = 0.0;
    target_state(2) = p.y + r * std::sin(yaw);
    target_state(3) = 0.0;
    target_state(4) = p.z;
    target_state(5) = 0.0;
  }

  ekf.set_state(target_state);
}

double
RmHandoffTracker::orientation_to_yaw(const geometry_msgs::msg::Quaternion &q) {
  tf2::Quaternion tf_q;
  tf2::fromMsg(q, tf_q);
  if (tf_q.length2() < 1e-12) {
    tf_q.setRPY(0.0, 0.0, 0.0);
  }
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
  yaw = last_yaw_ + shortest_angular_distance(last_yaw_, yaw);
  last_yaw_ = yaw;
  return yaw;
}

Eigen::Vector3d RmHandoffTracker::get_armor_position_from_state(
    const Eigen::VectorXd &x) const {
  const double xc = x(0);
  const double yc = x(2);
  const double za = x(4);
  const double yaw = x(6);
  const double r = x(8);
  return Eigen::Vector3d(xc - r * std::cos(yaw), yc - r * std::sin(yaw), za);
}

} // namespace yolo_cpp
