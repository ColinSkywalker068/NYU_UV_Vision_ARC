/// @file pnp_solver.cpp
/// @brief Implements armor-plane object points and OpenCV solvePnP pose conversion.

#include "yolo_cpp/pnp_solver.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
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
  return {cv::Point3f(-half_x, half_y, 0.0F), cv::Point3f(-half_x, -half_y, 0.0F),
    cv::Point3f(half_x, -half_y, 0.0F), cv::Point3f(half_x, half_y, 0.0F)};
}

/// Convert an OpenCV rotation matrix into a geometry_msgs quaternion.
void rotation_matrix_to_quaternion(const cv::Mat& rotation, geometry_msgs::msg::Pose& pose)
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

ArmorPnPSolver::ArmorPnPSolver(PnPCameraModel camera_model, double small_armor_width_m,
  double small_armor_height_m, double large_armor_width_m, double large_armor_height_m)
    : camera_model_(std::move(camera_model)),
      small_armor_points_(make_armor_points(small_armor_width_m, small_armor_height_m)),
      large_armor_points_(make_armor_points(large_armor_width_m, large_armor_height_m))
{
}

bool ArmorPnPSolver::solve(const std::array<cv::Point2f, 4>& image_points, int armor_type,
  geometry_msgs::msg::Pose& pose, PoseQuality * quality, double corner_sigma_px,
  double max_relative_depth_uncertainty) const
{
  // SOLVEPNP_IPPE is appropriate for planar rectangular armor targets.
  const auto& object_points = armor_type == 1 ? large_armor_points_ : small_armor_points_;
  cv::Mat camera_matrix(camera_model_.camera_matrix);
  cv::Mat dist_coeffs(camera_model_.distortion_coefficients, true);
  if (dist_coeffs.empty()) {
    dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
  }

  if (quality) {*quality = {};}
  const std::vector<cv::Point2f> image_vector(image_points.begin(), image_points.end());
  if (!std::all_of(image_points.begin(), image_points.end(), [](const auto & p) {
      return std::isfinite(p.x) && std::isfinite(p.y);
    }) || !cv::isContourConvex(image_vector) ||
    cv::contourArea(image_vector, true) < 4.0 ||
    !cv::checkRange(camera_matrix) || camera_model_.camera_matrix(0, 0) <= 0 ||
    camera_model_.camera_matrix(1, 1) <= 0 || object_points.size() != 4)
  {return false;}
  // Range from a small plate is ill-conditioned even when reprojection error
  // is tiny. Propagate endpoint pixel noise through z ~ f*height/pixel_height.
  // Use the shorter observed edge conservatively (including foreshortening).
  double shortest_edge = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < image_points.size(); ++i) {
    shortest_edge = std::min(shortest_edge,
      cv::norm(image_points[i] - image_points[(i + 1) % image_points.size()]));
  }
  const double relative_uncertainty = std::sqrt(2.0) * corner_sigma_px / shortest_edge;
  if (quality) {quality->relative_depth_uncertainty = relative_uncertainty;}
  if (!std::isfinite(corner_sigma_px) || corner_sigma_px <= 0 ||
    !std::isfinite(max_relative_depth_uncertainty) || max_relative_depth_uncertainty <= 0 ||
    !std::isfinite(relative_uncertainty) || relative_uncertainty > max_relative_depth_uncertainty) {
    return false;
  }
  std::vector<cv::Mat> rotations, translations;
  try {
    if (!cv::solvePnPGeneric(object_points, image_vector, camera_matrix, dist_coeffs,
        rotations, translations, false, cv::SOLVEPNP_IPPE)) {return false;}
  } catch (const cv::Exception &) {return false;}
  struct Hypothesis {cv::Mat rotation; cv::Mat translation; double error;};
  std::vector<Hypothesis> hypotheses;
  for (std::size_t i = 0; i < rotations.size(); ++i) {
    if (!cv::checkRange(rotations[i]) || !cv::checkRange(translations[i])) {continue;}
    cv::Mat rotation;
    cv::Rodrigues(rotations[i], rotation);
    bool positive_depth = true;
    for (const auto & corner : object_points) {
      const double depth = rotation.at<double>(2, 0) * corner.x +
        rotation.at<double>(2, 1) * corner.y + translations[i].at<double>(2);
      positive_depth = positive_depth && depth > 0.05;
    }
    if (!positive_depth || cv::norm(translations[i]) > 30.0) {continue;}
    std::vector<cv::Point2f> reprojections;
    cv::projectPoints(object_points, rotations[i], translations[i], camera_matrix,
      dist_coeffs, reprojections);
    double squared_error = 0;
    for (std::size_t j = 0; j < image_points.size(); ++j) {
      const auto residual = reprojections[j] - image_points[j];
      squared_error += residual.dot(residual);
    }
    const double error = std::sqrt(squared_error / image_points.size());
    if (std::isfinite(error) && error <= 3.0) {
      hypotheses.push_back({rotation, translations[i], error});
    }
  }
  if (hypotheses.empty()) {return false;}
  std::sort(hypotheses.begin(), hypotheses.end(), [](const auto & a, const auto & b) {
    return a.error < b.error;
  });
  const auto & best = hypotheses.front();
  pose.position.x = best.translation.at<double>(0);
  pose.position.y = best.translation.at<double>(1);
  pose.position.z = best.translation.at<double>(2);
  rotation_matrix_to_quaternion(best.rotation, pose);
  if (quality) {
    quality->reprojection_rms_px = best.error;
    quality->normal_reliable = true;
    if (hypotheses.size() > 1) {
      quality->alternative_error_gap_px = hypotheses[1].error - best.error;
      const double normal_dot = best.rotation.col(2).dot(hypotheses[1].rotation.col(2));
      // A plausible translation can coexist with an ambiguous planar normal.
      quality->normal_reliable = normal_dot > std::cos(15.0 * 3.141592653589793 / 180) ||
        quality->alternative_error_gap_px >= std::max(0.5, 2.0 * best.error);
    }
  }
  return true;
}

}  // namespace yolo_cpp
