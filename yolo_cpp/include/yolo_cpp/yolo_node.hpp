#ifndef YOLO_CPP__YOLO_NODE_HPP_
#define YOLO_CPP__YOLO_NODE_HPP_

#include "yolo_cpp/deepstream_detector.hpp"
#include "yolo_cpp/tracker.hpp"
#include "yolo_cpp/utils.hpp"
#include "yolo_cpp/visibility_control.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>

#include <map>
#include <opencv2/core.hpp>

namespace yolo_cpp
{

class YOLO_CPP_PUBLIC YoloNode : public rclcpp::Node
{
public:
  explicit YoloNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~YoloNode() override = default;

private:
  void id_listener_callback(const std_msgs::msg::Int32::SharedPtr msg);
  void image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg);

  cv::Mat decode_frame(const sensor_msgs::msg::Image & msg);
  std::vector<Candidate> extract_candidates(
    const std::vector<Detection> & detections, const LetterboxResult & lb) const;
  std::vector<Candidate> rank_candidates(std::vector<Candidate> candidates) const;
  std::optional<Candidate> select_target(const std::vector<Candidate> & candidates) const;
  void publish_target(const Candidate & target, const LetterboxResult & lb);
  void publish_no_target(int src_w, int src_h);
  bool publish_kalman_target(
    double xf, double yf, double width, double height, const LetterboxResult & lb,
    const std::optional<float> & obb_angle);
  bool run_kalman_pipeline(
    const std::vector<Candidate> & candidates, double dt, const LetterboxResult & lb);
  std::string resolve_default_model_path() const;

  DeepStreamDetector detector_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr robot_id_subscription_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr bbox_publisher_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr override_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr angle_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr obb_angle_publisher_;

  std_msgs::msg::Int32MultiArray bbox_msg_;
  std_msgs::msg::Bool override_msg_;
  std_msgs::msg::Float32MultiArray angle_msg_;
  std_msgs::msg::Float32 obb_angle_msg_;

  std::map<int, TrackerWrapper> active_trackers_;
  rclcpp::Time last_frame_time_;
  bool have_last_frame_time_{false};

  std::string model_path_;
  std::string task_;
  std::string model_type_;
  std::string inference_mode_;
  std::string target_color_;
  std::string vis_window_name_;
  bool half_{true};
  bool show_{false};
  bool persist_tracking_{true};
  bool use_kalman_{false};
  bool enable_color_verification_{true};
  bool use_robot_id_targeting_{false};
  int imgsz_width_{640};
  int imgsz_height_{640};
  int tracker_timeout_{15};
  int max_hsv_candidates_{3};
  int max_coast_frames_{3};
  float conf_{0.45F};
  float lookahead_steps_{0.5F};
  float hfov_rad_{0.0F};
  float vfov_rad_{0.0F};
  int target_class_{BLUE_CLASS};
};

}  // namespace yolo_cpp

#endif
