#ifndef YOLO_CPP__RM_HANDOFF_TRACKER_NODE_HPP_
#define YOLO_CPP__RM_HANDOFF_TRACKER_NODE_HPP_

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"
#include "auto_aim_interfaces/msg/tracker_info.hpp"
#include "yolo_cpp/rm_handoff_tracker.hpp"
#include "yolo_cpp/visibility_control.hpp"

namespace yolo_cpp {

class YOLO_CPP_PUBLIC RmHandoffTrackerNode : public rclcpp::Node {
public:
  explicit RmHandoffTrackerNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
  ~RmHandoffTrackerNode() override = default;

private:
  void armors_callback(const auto_aim_interfaces::msg::Armors::SharedPtr msg);
  void publish_target(const rclcpp::Time &stamp,
                      auto_aim_interfaces::msg::Target &target_msg);
  void publish_markers(const auto_aim_interfaces::msg::Target &target_msg);
  void configure_tracker_ekf(double dt);

  double max_armor_distance_{10.0};
  double lost_time_thres_{0.3};
  double s2qxyz_{20.0};
  double s2qyaw_{100.0};
  double s2qr_{800.0};
  double r_xyz_factor_{0.05};
  double r_yaw_{0.02};
  double last_dt_{1.0 / 60.0};

  std::string target_frame_{"gimbal_odom"};
  rclcpp::Time last_time_;
  bool have_last_time_{false};

  std::unique_ptr<RmHandoffTracker> tracker_;
  std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;

  rclcpp::Subscription<auto_aim_interfaces::msg::Armors>::SharedPtr
      armors_subscription_;
  rclcpp::Publisher<auto_aim_interfaces::msg::Target>::SharedPtr
      target_publisher_;
  rclcpp::Publisher<auto_aim_interfaces::msg::TrackerInfo>::SharedPtr
      info_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_publisher_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;

  visualization_msgs::msg::Marker position_marker_;
  visualization_msgs::msg::Marker linear_v_marker_;
  visualization_msgs::msg::Marker angular_v_marker_;
  visualization_msgs::msg::Marker armor_marker_;
};

} // namespace yolo_cpp

#endif // YOLO_CPP__RM_HANDOFF_TRACKER_NODE_HPP_
