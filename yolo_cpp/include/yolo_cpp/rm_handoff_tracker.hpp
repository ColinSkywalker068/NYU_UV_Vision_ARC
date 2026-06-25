#ifndef YOLO_CPP__RM_HANDOFF_TRACKER_HPP_
#define YOLO_CPP__RM_HANDOFF_TRACKER_HPP_

#include <Eigen/Eigen>

#include <geometry_msgs/msg/quaternion.hpp>

#include <functional>
#include <string>

#include "auto_aim_interfaces/msg/armor.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"

namespace yolo_cpp {

enum class RmHandoffArmorsNum { kNormal4 = 4, kBalance2 = 2, kOutpost3 = 3 };

class RmHandoffEkf {
public:
  using VecVecFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;
  using VecMatFunc = std::function<Eigen::MatrixXd(const Eigen::VectorXd &)>;
  using VoidMatFunc = std::function<Eigen::MatrixXd()>;

  RmHandoffEkf() = default;
  RmHandoffEkf(const VecVecFunc &f, const VecVecFunc &h, const VecMatFunc &j_f,
               const VecMatFunc &j_h, const VoidMatFunc &u_q,
               const VecMatFunc &u_r, const Eigen::MatrixXd &p0);

  void set_state(const Eigen::VectorXd &x0);
  Eigen::VectorXd predict();
  Eigen::VectorXd update(const Eigen::VectorXd &z);

private:
  VecVecFunc f_;
  VecVecFunc h_;
  VecMatFunc jacobian_f_;
  VecMatFunc jacobian_h_;
  VoidMatFunc update_q_;
  VecMatFunc update_r_;

  Eigen::MatrixXd p_pri_;
  Eigen::MatrixXd p_post_;
  Eigen::MatrixXd identity_;
  Eigen::VectorXd x_pri_;
  Eigen::VectorXd x_post_;
};

class RmHandoffTracker {
public:
  using Armor = auto_aim_interfaces::msg::Armor;
  using Armors = auto_aim_interfaces::msg::Armors;

  enum State { kLost, kDetecting, kTracking, kTempLost };

  RmHandoffTracker(double max_match_distance, double max_match_yaw_diff);

  void init(const Armors &armors_msg);
  void update(const Armors &armors_msg);

  void configure_ekf(double dt, double sigma2_q_xyz, double sigma2_q_yaw,
                     double sigma2_q_r, double r_xyz_factor, double r_yaw);
  void set_dt(double dt);

  static std::string armor_id(const Armor &armor);
  static std::string armor_type(const Armor &armor);

  RmHandoffEkf ekf;
  int tracking_thres{5};
  int lost_thres{30};
  State tracker_state{kLost};

  std::string tracked_id;
  Armor tracked_armor;
  RmHandoffArmorsNum tracked_armors_num{RmHandoffArmorsNum::kNormal4};

  double info_position_diff{0.0};
  double info_yaw_diff{0.0};
  Eigen::VectorXd measurement{Eigen::VectorXd::Zero(4)};
  Eigen::VectorXd target_state{Eigen::VectorXd::Zero(9)};
  double dz{0.0};
  double another_r{0.26};

  Eigen::Vector3d get_armor_position_from_state(const Eigen::VectorXd &x) const;

private:
  void init_ekf(const Armor &armor);
  void update_armors_num(const Armor &armor);
  void handle_armor_jump(const Armor &armor);
  double orientation_to_yaw(const geometry_msgs::msg::Quaternion &q);

  double max_match_distance_;
  double max_match_yaw_diff_;
  double dt_{1.0 / 60.0};
  double sigma2_q_xyz_{20.0};
  double sigma2_q_yaw_{100.0};
  double sigma2_q_r_{800.0};
  double r_xyz_factor_{0.05};
  double r_yaw_{0.02};
  int detect_count_{0};
  int lost_count_{0};
  double last_yaw_{0.0};
};

} // namespace yolo_cpp

#endif // YOLO_CPP__RM_HANDOFF_TRACKER_HPP_
