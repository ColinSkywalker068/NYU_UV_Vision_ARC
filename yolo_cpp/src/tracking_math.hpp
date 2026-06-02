#ifndef YOLO_CPP_SRC_TRACKING_MATH_HPP_
#define YOLO_CPP_SRC_TRACKING_MATH_HPP_

#include <opencv2/core.hpp>
#include <vector>

#include "yolo_cpp/tracker.hpp"

namespace yolo_cpp::tracking_math
{

constexpr double kMinimumDt = 1e-3;
constexpr double kInvalidAssignmentCost = 1e6;

/// Multiply two 6x6 matrices used by the image-space Kalman filter.
void multiply_6x6(const double a[6][6], const double b[6][6], double out[6][6]);

/// Transpose a 6x6 matrix used by covariance propagation.
void transpose_6x6(const double in[6][6], double out[6][6]);

/// Clamp a predicted center into image bounds.
cv::Point2f clamp_center(const cv::Point2f& center, int image_width, int image_height);

/// Build an image-bounded rectangle from a center and size.
cv::Rect rect_from_center(
  const cv::Point2f& center, double width, double height, int image_width, int image_height);

/// Compute axis-aligned box IoU for track association.
double rect_iou(const cv::Rect& lhs, const cv::Rect& rhs);

/// Compute center distance normalized by image diagonal.
double normalized_center_distance(
  const cv::Point2f& lhs, const cv::Point2f& rhs, int image_width, int image_height);

/// Compute Euclidean distance between two camera-frame 3D points.
double distance_3d(const cv::Point3d& lhs, const cv::Point3d& rhs);

/// Solve rectangular assignment with a square-padded Hungarian algorithm.
std::vector<int> solve_hungarian(const std::vector<std::vector<double>>& cost_matrix);

/// Return a stable priority used when ranking selectable tracks.
int state_priority(TrackState state);

}  // namespace yolo_cpp::tracking_math

#endif  // YOLO_CPP_SRC_TRACKING_MATH_HPP_
