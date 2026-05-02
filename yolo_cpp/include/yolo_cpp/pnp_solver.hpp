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
  /// Camera intrinsic matrix in OpenCV convention.
  cv::Matx33d camera_matrix{cv::Matx33d::eye()};
  /// Distortion coefficients accepted by OpenCV solvePnP.
  std::vector<double> distortion_coefficients;
};

/// Solves armor corner observations into camera-frame poses.
class ArmorPnPSolver
{
public:
  ArmorPnPSolver() = default;

  /// Build object-point templates for small and large armor plates.
  ArmorPnPSolver(
    PnPCameraModel camera_model, double small_armor_width_m, double small_armor_height_m,
    double large_armor_width_m, double large_armor_height_m);

  /// Estimate the 3D pose for one observed armor.
  ///
  /// Args:
  ///   image_points: Four image-space armor corners in consistent order.
  ///   armor_type: cv_auto_aim_interfaces Armor type constant.
  ///   pose: Output camera-frame pose when solving succeeds.
  ///
  /// Returns:
  ///   True when OpenCV solvePnP produced a finite pose.
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
