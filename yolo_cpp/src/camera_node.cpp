#include "yolo_cpp/camera_node.hpp"

#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace yolo_cpp
{

CameraNode::CameraNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("camera_node", options),
  last_warning_time_(this->get_clock()->now())
{
  camera_config_ = load_camera_config(*this);
  camera_type_ = camera_config_.camera_type;
  backend_ = create_camera_backend(camera_config_, this->get_logger());
  backend_->initialize();
  fps_ = backend_->actual_fps() > 0.0 ? backend_->actual_fps() : 60.0;

  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
  image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", qos);
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / std::max(1.0, fps_)),
    std::bind(&CameraNode::timer_callback, this));

  RCLCPP_INFO(this->get_logger(), "Initialized legacy %s camera node at %.2f FPS", camera_type_.c_str(), fps_);
}

CameraNode::~CameraNode()
{
  if (backend_) {
    backend_->release();
  }
}

void CameraNode::timer_callback()
{
  FramePacket packet{};
  if (!backend_->capture(packet)) {
    publish_no_frame_warning();
    return;
  }

  auto msg = to_image_message(packet);
  if (msg) {
    image_publisher_->publish(std::move(msg));
  }
}

void CameraNode::publish_no_frame_warning()
{
  const auto now = this->get_clock()->now();
  if ((now - last_warning_time_).seconds() >= 1.0) {
    RCLCPP_WARN(this->get_logger(), "Failed to get frame from %s camera backend", camera_type_.c_str());
    last_warning_time_ = now;
  }
}

sensor_msgs::msg::Image::UniquePtr CameraNode::to_image_message(const FramePacket & packet)
{
  if (packet.empty()) {
    return nullptr;
  }

  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  msg->header.stamp = this->now();
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

}  // namespace yolo_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolo_cpp::CameraNode)
