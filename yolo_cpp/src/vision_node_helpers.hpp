#ifndef YOLO_CPP_SRC_VISION_NODE_HELPERS_HPP_
#define YOLO_CPP_SRC_VISION_NODE_HELPERS_HPP_

#include "yolo_cpp/frame_packet.hpp"

#include <opencv2/core.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <string>

namespace yolo_cpp::vision_node_helpers
{

/// Return local wall-clock time formatted for human-readable runtime logs.
std::string timestamp_now_iso8601_ms();

/// Return compact local wall-clock time suitable for log file names.
std::string timestamp_now_compact();

/// Copy an internal BGR frame packet into a ROS Image message.
sensor_msgs::msg::Image::UniquePtr frame_to_image_message(
  const FramePacket & packet, const rclcpp::Time & stamp);

/// Compute axis-aligned rectangle IoU for target-frame capture matching.
double rect_iou(const cv::Rect & lhs, const cv::Rect & rhs);

/// Scale a source-space rectangle into the preview image and clip it to the image bounds.
cv::Rect scale_rect_to_preview(
  const cv::Rect & rect,
  double scale_x,
  double scale_y,
  const cv::Rect & image_rect);

}  // namespace yolo_cpp::vision_node_helpers

#endif  // YOLO_CPP_SRC_VISION_NODE_HELPERS_HPP_
