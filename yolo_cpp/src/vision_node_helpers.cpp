/// @file vision_node_helpers.cpp
/// @brief Shared helpers for detector and perception node logging, images, and previews.

#include "vision_node_helpers.hpp"

#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <memory>
#include <sstream>

namespace yolo_cpp::vision_node_helpers
{

std::string timestamp_now_iso8601_ms()
{
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(now);
  const auto milliseconds =
    std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds).count();
  const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &now_c);
#else
  localtime_r(&now_c, &local_tm);
#endif

  std::ostringstream out;
  out << std::put_time(&local_tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3)
      << std::setfill('0') << milliseconds;
  return out.str();
}

std::string timestamp_now_compact()
{
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  std::tm local_tm{};
#ifdef _WIN32
  localtime_s(&local_tm, &now_c);
#else
  localtime_r(&now_c, &local_tm);
#endif

  std::ostringstream out;
  out << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
  return out.str();
}

sensor_msgs::msg::Image::UniquePtr frame_to_image_message(
  const FramePacket & packet, const rclcpp::Time & stamp)
{
  if (packet.empty()) {
    return nullptr;
  }

  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  msg->header.stamp = stamp;
  msg->header.frame_id = packet.source_id;
  msg->height = static_cast<uint32_t>(packet.frame.rows);
  msg->width = static_cast<uint32_t>(packet.frame.cols);
  msg->encoding = sensor_msgs::image_encodings::BGR8;
  msg->is_bigendian = false;
  msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(packet.frame.step);

  const std::size_t byte_count = packet.frame.total() * packet.frame.elemSize();
  msg->data.resize(byte_count);
  std::memcpy(msg->data.data(), packet.frame.data, byte_count);
  return msg;
}

double rect_iou(const cv::Rect & lhs, const cv::Rect & rhs)
{
  const cv::Rect overlap = lhs & rhs;
  if (overlap.empty()) {
    return 0.0;
  }

  const double intersection = static_cast<double>(overlap.area());
  const double union_area = static_cast<double>(lhs.area() + rhs.area()) - intersection;
  return union_area > 0.0 ? intersection / union_area : 0.0;
}

cv::Rect scale_rect_to_preview(
  const cv::Rect & rect,
  double scale_x,
  double scale_y,
  const cv::Rect & image_rect)
{
  const int x = static_cast<int>(std::lround(static_cast<double>(rect.x) * scale_x));
  const int y = static_cast<int>(std::lround(static_cast<double>(rect.y) * scale_y));
  const int width =
    std::max(1, static_cast<int>(std::lround(static_cast<double>(rect.width) * scale_x)));
  const int height =
    std::max(1, static_cast<int>(std::lround(static_cast<double>(rect.height) * scale_y)));
  return cv::Rect(x, y, width, height) & image_rect;
}

}  // namespace yolo_cpp::vision_node_helpers
