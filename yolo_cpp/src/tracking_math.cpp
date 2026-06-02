/// @file tracking_math.cpp
/// @brief Geometry, Kalman matrix, and assignment helpers for armor tracking.

#include "tracking_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace yolo_cpp::tracking_math
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

cv::Point2f clamp_center(const cv::Point2f& center, int image_width, int image_height)
{
  const float max_x = static_cast<float>(std::max(0, image_width - 1));
  const float max_y = static_cast<float>(std::max(0, image_height - 1));
  return cv::Point2f(std::clamp(center.x, 0.0F, max_x), std::clamp(center.y, 0.0F, max_y));
}

cv::Rect rect_from_center(
  const cv::Point2f& center, double width, double height, int image_width, int image_height)
{
  const int max_w = std::max(1, image_width);
  const int max_h = std::max(1, image_height);
  int x1 = static_cast<int>(std::round(static_cast<double>(center.x) - (width / 2.0)));
  int y1 = static_cast<int>(std::round(static_cast<double>(center.y) - (height / 2.0)));
  int x2 = static_cast<int>(std::round(static_cast<double>(center.x) + (width / 2.0)));
  int y2 = static_cast<int>(std::round(static_cast<double>(center.y) + (height / 2.0)));
  x1 = std::clamp(x1, 0, max_w - 1);
  y1 = std::clamp(y1, 0, max_h - 1);
  x2 = std::clamp(x2, 0, max_w - 1);
  y2 = std::clamp(y2, 0, max_h - 1);

  if (x2 <= x1) {
    x2 = std::min(max_w, x1 + 1);
  }
  if (y2 <= y1) {
    y2 = std::min(max_h, y1 + 1);
  }
  return cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1));
}

double rect_iou(const cv::Rect& lhs, const cv::Rect& rhs)
{
  const int x1 = std::max(lhs.x, rhs.x);
  const int y1 = std::max(lhs.y, rhs.y);
  const int x2 = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
  const int y2 = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
  const int inter_w = std::max(0, x2 - x1);
  const int inter_h = std::max(0, y2 - y1);
  const double inter = static_cast<double>(inter_w) * static_cast<double>(inter_h);
  const double lhs_area = static_cast<double>(lhs.width) * static_cast<double>(lhs.height);
  const double rhs_area = static_cast<double>(rhs.width) * static_cast<double>(rhs.height);
  const double denom = lhs_area + rhs_area - inter;
  if (denom <= 0.0) {
    return 0.0;
  }
  return inter / denom;
}

double normalized_center_distance(
  const cv::Point2f& lhs, const cv::Point2f& rhs, int image_width, int image_height)
{
  const double dx = static_cast<double>(lhs.x - rhs.x);
  const double dy = static_cast<double>(lhs.y - rhs.y);
  const double diagonal = std::hypot(
    static_cast<double>(std::max(1, image_width)), static_cast<double>(std::max(1, image_height)));
  if (diagonal <= 0.0) {
    return 0.0;
  }
  return std::hypot(dx, dy) / diagonal;
}

double distance_3d(const cv::Point3d& lhs, const cv::Point3d& rhs)
{
  return std::sqrt(((lhs.x - rhs.x) * (lhs.x - rhs.x)) + ((lhs.y - rhs.y) * (lhs.y - rhs.y))
                   + ((lhs.z - rhs.z) * (lhs.z - rhs.z)));
}

std::vector<int> solve_hungarian(const std::vector<std::vector<double>>& cost_matrix)
{
  if (cost_matrix.empty() || cost_matrix.front().empty()) {
    return {};
  }

  const int rows = static_cast<int>(cost_matrix.size());
  const int cols = static_cast<int>(cost_matrix.front().size());
  const int size = std::max(rows, cols);

  std::vector<std::vector<double>> padded(static_cast<std::size_t>(size),
    std::vector<double>(static_cast<std::size_t>(size), kInvalidAssignmentCost));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      padded[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = cost_matrix[r][c];
    }
  }

  std::vector<double> u(static_cast<std::size_t>(size + 1), 0.0);
  std::vector<double> v(static_cast<std::size_t>(size + 1), 0.0);
  std::vector<int> p(static_cast<std::size_t>(size + 1), 0);
  std::vector<int> way(static_cast<std::size_t>(size + 1), 0);

  for (int i = 1; i <= size; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(static_cast<std::size_t>(size + 1), kInvalidAssignmentCost);
    std::vector<bool> used(static_cast<std::size_t>(size + 1), false);
    do {
      used[static_cast<std::size_t>(j0)] = true;
      const int i0 = p[static_cast<std::size_t>(j0)];
      int j1 = 0;
      double delta = kInvalidAssignmentCost;
      for (int j = 1; j <= size; ++j) {
        if (used[static_cast<std::size_t>(j)]) {
          continue;
        }
        const double cur = padded[static_cast<std::size_t>(i0 - 1)][static_cast<std::size_t>(j - 1)]
                           - u[static_cast<std::size_t>(i0)] - v[static_cast<std::size_t>(j)];
        if (cur < minv[static_cast<std::size_t>(j)]) {
          minv[static_cast<std::size_t>(j)] = cur;
          way[static_cast<std::size_t>(j)] = j0;
        }
        if (minv[static_cast<std::size_t>(j)] < delta) {
          delta = minv[static_cast<std::size_t>(j)];
          j1 = j;
        }
      }
      for (int j = 0; j <= size; ++j) {
        if (used[static_cast<std::size_t>(j)]) {
          u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta;
          v[static_cast<std::size_t>(j)] -= delta;
        } else {
          minv[static_cast<std::size_t>(j)] -= delta;
        }
      }
      j0 = j1;
    } while (p[static_cast<std::size_t>(j0)] != 0);

    do {
      const int j1 = way[static_cast<std::size_t>(j0)];
      p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> assignment(static_cast<std::size_t>(rows), -1);
  for (int j = 1; j <= size; ++j) {
    const int row = p[static_cast<std::size_t>(j)] - 1;
    const int col = j - 1;
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      assignment[static_cast<std::size_t>(row)] = col;
    }
  }
  return assignment;
}

int state_priority(TrackState state)
{
  switch (state) {
    case TrackState::kTracking:
      return 2;
    case TrackState::kDetecting:
      return 1;
    case TrackState::kCoasting:
      return 0;
  }
  return -1;
}

}  // namespace yolo_cpp::tracking_math
