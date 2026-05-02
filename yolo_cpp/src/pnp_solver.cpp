/// @file pnp_solver.cpp
/// @brief Implements armor-plane object points and OpenCV solvePnP pose conversion.

#include "yolo_cpp/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <utility>

namespace yolo_cpp
{

namespace
{

/// Build the 3D corner template for an armor plate centered on the origin.
std::vector<cv::Point3f> make_armor_points(double width_m, double height_m)
{
  const auto half_x = static_cast<float>(width_m / 2.0);
  const auto half_y = static_cast<float>(height_m / 2.0);
  return {
    cv::Point3f(-half_x, half_y, 0.0F),
    cv::Point3f(-half_x, -half_y, 0.0F),
    cv::Point3f(half_x, -half_y, 0.0F),
    cv::Point3f(half_x, half_y, 0.0F)};
}

/// Convert an OpenCV rotation matrix into a geometry_msgs quaternion.
void rotation_matrix_to_quaternion(const cv::Mat & rotation, geometry_msgs::msg::Pose & pose)
{
  const double m00 = rotation.at<double>(0, 0);
  const double m01 = rotation.at<double>(0, 1);
  const double m02 = rotation.at<double>(0, 2);
  const double m10 = rotation.at<double>(1, 0);
  const double m11 = rotation.at<double>(1, 1);
  const double m12 = rotation.at<double>(1, 2);
  const double m20 = rotation.at<double>(2, 0);
  const double m21 = rotation.at<double>(2, 1);
  const double m22 = rotation.at<double>(2, 2);

  const double trace = m00 + m11 + m22;
  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    pose.orientation.w = 0.25 * s;
    pose.orientation.x = (m21 - m12) / s;
    pose.orientation.y = (m02 - m20) / s;
    pose.orientation.z = (m10 - m01) / s;
    return;
  }

  if (m00 > m11 && m00 > m22) {
    const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
    pose.orientation.w = (m21 - m12) / s;
    pose.orientation.x = 0.25 * s;
    pose.orientation.y = (m01 + m10) / s;
    pose.orientation.z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
    pose.orientation.w = (m02 - m20) / s;
    pose.orientation.x = (m01 + m10) / s;
    pose.orientation.y = 0.25 * s;
    pose.orientation.z = (m12 + m21) / s;
  } else {
    const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
    pose.orientation.w = (m10 - m01) / s;
    pose.orientation.x = (m02 + m20) / s;
    pose.orientation.y = (m12 + m21) / s;
    pose.orientation.z = 0.25 * s;
  }
}

}  // namespace

ArmorPnPSolver::ArmorPnPSolver(
  PnPCameraModel camera_model, double small_armor_width_m, double small_armor_height_m,
  double large_armor_width_m, double large_armor_height_m)
: camera_model_(std::move(camera_model)),
  small_armor_points_(make_armor_points(small_armor_width_m, small_armor_height_m)),
  large_armor_points_(make_armor_points(large_armor_width_m, large_armor_height_m))
{
}

bool ArmorPnPSolver::solve(
  const std::array<cv::Point2f, 4> & image_points, int armor_type,
  geometry_msgs::msg::Pose & pose) const
{
  // SOLVEPNP_IPPE is appropriate for planar rectangular armor targets.
  const auto & object_points = armor_type == 1 ? large_armor_points_ : small_armor_points_;
  cv::Mat camera_matrix(camera_model_.camera_matrix);
  cv::Mat dist_coeffs(camera_model_.distortion_coefficients, true);
  if (dist_coeffs.empty()) {
    dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
  }

  cv::Mat rvec;
  cv::Mat tvec;
  const std::vector<cv::Point2f> image_vector(image_points.begin(), image_points.end());
  const bool ok = cv::solvePnP(
    object_points, image_vector, camera_matrix, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);
  if (!ok) {
    return false;
  }

  pose.position.x = tvec.at<double>(0);
  pose.position.y = tvec.at<double>(1);
  pose.position.z = tvec.at<double>(2);
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);
  rotation_matrix_to_quaternion(rotation, pose);
  return true;
}

}  // namespace yolo_cpp
