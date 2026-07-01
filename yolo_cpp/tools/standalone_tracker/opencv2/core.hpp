#ifndef YOLO_CPP_TOOLS_STANDALONE_TRACKER_OPENCV2_CORE_HPP_
#define YOLO_CPP_TOOLS_STANDALONE_TRACKER_OPENCV2_CORE_HPP_

#include <array>

namespace cv {

struct Point2f {
  float x{0.0F};
  float y{0.0F};

  Point2f() = default;
  Point2f(float x_in, float y_in) : x(x_in), y(y_in) {}
};

struct Point3d {
  double x{0.0};
  double y{0.0};
  double z{0.0};

  Point3d() = default;
  Point3d(double x_in, double y_in, double z_in) : x(x_in), y(y_in), z(z_in) {}
};

struct Rect {
  int x{0};
  int y{0};
  int width{0};
  int height{0};

  Rect() = default;
  Rect(int x_in, int y_in, int width_in, int height_in)
      : x(x_in), y(y_in), width(width_in), height(height_in) {}
};

class Vec3d {
public:
  Vec3d() = default;
  Vec3d(double x_in, double y_in, double z_in) : values_{x_in, y_in, z_in} {}

  double &operator[](int index) {
    return values_[static_cast<std::size_t>(index)];
  }
  const double &operator[](int index) const {
    return values_[static_cast<std::size_t>(index)];
  }

private:
  std::array<double, 3> values_{0.0, 0.0, 0.0};
};

} // namespace cv

#endif // YOLO_CPP_TOOLS_STANDALONE_TRACKER_OPENCV2_CORE_HPP_
