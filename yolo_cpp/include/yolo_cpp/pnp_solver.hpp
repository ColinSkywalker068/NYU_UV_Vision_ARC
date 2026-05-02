#ifndef YOLO_CPP__PNP_SOLVER_HPP_
#define YOLO_CPP__PNP_SOLVER_HPP_

#include <geometry_msgs/msg/pose.hpp>
#include <opencv2/core.hpp>

#include <array>
#include <vector>

namespace yolo_cpp
{

struct PnPCameraModel
{
  cv::Matx33d camera_matrix{cv::Matx33d::eye()};
  std::vector<double> distortion_coefficients;
};

class ArmorPnPSolver
{
public:
  ArmorPnPSolver() = default;
  ArmorPnPSolver(
    PnPCameraModel camera_model, double small_armor_width_m, double small_armor_height_m,
    double large_armor_width_m, double large_armor_height_m);

  bool solve(
    const std::array<cv::Point2f, 4> & image_points, int armor_type,
    geometry_msgs::msg::Pose & pose) const;

private:
  PnPCameraModel camera_model_;
  std::vector<cv::Point3f> small_armor_points_;
  std::vector<cv::Point3f> large_armor_points_;
};

}  // namespace yolo_cpp

#endif
