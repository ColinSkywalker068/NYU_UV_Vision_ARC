#include "yolov8_ros_cpp/camera_node.hpp"

#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_LIBREALSENSE2
#include <librealsense2/rs.hpp>
#endif

#ifdef HAVE_PYLON
#include <pylon/PylonIncludes.h>
#include <GenApi/GenApi.h>
#endif

namespace yolov8_ros_cpp
{

class CameraBackend
{
public:
  virtual ~CameraBackend() = default;
  virtual void initialize() = 0;
  virtual bool get_frame(cv::Mat & frame) = 0;
  virtual void release() = 0;
  virtual double actual_fps() const = 0;
};

namespace
{

class UnsupportedCameraBackend : public CameraBackend
{
public:
  explicit UnsupportedCameraBackend(std::string reason) : reason_(std::move(reason)) {}

  void initialize() override
  {
    throw std::runtime_error(reason_);
  }

  bool get_frame(cv::Mat &) override
  {
    return false;
  }

  void release() override {}
  double actual_fps() const override {return 0.0;}

private:
  std::string reason_;
};

#ifdef HAVE_LIBREALSENSE2
rs2_format format_by_name(const std::string & name)
{
  if (name == "bgr8") {
    return RS2_FORMAT_BGR8;
  }
  if (name == "rgb8") {
    return RS2_FORMAT_RGB8;
  }
  return RS2_FORMAT_YUYV;
}

class RealSenseBackend : public CameraBackend
{
public:
  RealSenseBackend(
    int width, int height, int fps, double exposure, double gain, bool auto_exposure,
    bool auto_white_balance, double white_balance, bool use_poll, bool use_max_native_fps)
  : target_width_(width),
    target_height_(height),
    target_fps_(fps),
    exposure_(exposure),
    gain_(gain),
    auto_exposure_(auto_exposure),
    auto_white_balance_(auto_white_balance),
    white_balance_(white_balance),
    use_poll_(use_poll),
    use_max_native_fps_(use_max_native_fps)
  {
  }

  void initialize() override
  {
    rs2::context ctx;
    auto devices = ctx.query_devices();
    if (devices.size() == 0) {
      throw std::runtime_error("No RealSense camera found");
    }

    rs2::config config;
    try {
      config.disable_all_streams();
    } catch (...) {
    }

    discover_best_stream(ctx);
    config.enable_stream(RS2_STREAM_COLOR, requested_width_, requested_height_, stream_format_, requested_fps_);
    profile_ = pipeline_.start(config);
    configure_sensor();

    auto stream_profile = profile_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    actual_fps_ = static_cast<double>(stream_profile.fps());
  }

  bool get_frame(cv::Mat & frame) override
  {
    rs2::frameset frames;
    if (use_poll_) {
      if (!pipeline_.poll_for_frames(&frames)) {
        return false;
      }
    } else {
      frames = pipeline_.wait_for_frames(1000);
    }

    rs2::video_frame color_frame = frames.get_color_frame();
    if (!color_frame) {
      return false;
    }

    const int width = color_frame.get_width();
    const int height = color_frame.get_height();
    const auto * data = static_cast<const uint8_t *>(color_frame.get_data());

    if (stream_format_ == RS2_FORMAT_BGR8) {
      cv::Mat view(height, width, CV_8UC3, const_cast<uint8_t *>(data), cv::Mat::AUTO_STEP);
      frame = view.clone();
      return true;
    }

    if (stream_format_ == RS2_FORMAT_RGB8) {
      cv::Mat rgb(height, width, CV_8UC3, const_cast<uint8_t *>(data), cv::Mat::AUTO_STEP);
      cv::cvtColor(rgb, frame, cv::COLOR_RGB2BGR);
      return true;
    }

    cv::Mat yuyv(height, width, CV_8UC2, const_cast<uint8_t *>(data), cv::Mat::AUTO_STEP);
    cv::cvtColor(yuyv, frame, cv::COLOR_YUV2BGR_YUY2);
    return true;
  }

  void release() override
  {
    try {
      pipeline_.stop();
    } catch (...) {
    }
  }

  double actual_fps() const override
  {
    return actual_fps_;
  }

private:
  void discover_best_stream(const rs2::context & ctx)
  {
    static const std::vector<std::string> preferred_formats{"bgr8", "rgb8", "yuyv"};
    requested_width_ = target_width_;
    requested_height_ = target_height_;
    requested_fps_ = target_fps_;
    stream_format_ = RS2_FORMAT_YUYV;

    int best_fps = -1;
    int best_rank = static_cast<int>(preferred_formats.size());

    for (auto && device : ctx.query_devices()) {
      for (auto && sensor : device.query_sensors()) {
        if (!sensor.is<rs2::color_sensor>()) {
          continue;
        }
        for (auto && profile : sensor.get_stream_profiles()) {
          auto video = profile.as<rs2::video_stream_profile>();
          if (video.stream_type() != RS2_STREAM_COLOR) {
            continue;
          }
          if (video.width() != target_width_ || video.height() != target_height_) {
            continue;
          }
          const int fps = video.fps();
          if (!use_max_native_fps_ && fps < target_fps_) {
            continue;
          }
          const auto fmt = video.format();
          for (std::size_t rank = 0; rank < preferred_formats.size(); ++rank) {
            if (fmt != format_by_name(preferred_formats[rank])) {
              continue;
            }
            if (fps > best_fps || (fps == best_fps && static_cast<int>(rank) < best_rank)) {
              best_fps = fps;
              best_rank = static_cast<int>(rank);
              requested_fps_ = fps;
              stream_format_ = fmt;
            }
          }
        }
      }
    }
  }

  void configure_sensor()
  {
    auto device = profile_.get_device();
    for (auto && sensor : device.query_sensors()) {
      if (!sensor.is<rs2::color_sensor>()) {
        continue;
      }
      if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
        sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, auto_exposure_ ? 1.0F : 0.0F);
      }
      if (!auto_exposure_) {
        if (sensor.supports(RS2_OPTION_EXPOSURE)) {
          sensor.set_option(RS2_OPTION_EXPOSURE, static_cast<float>(exposure_));
        }
        if (sensor.supports(RS2_OPTION_GAIN)) {
          sensor.set_option(RS2_OPTION_GAIN, static_cast<float>(gain_));
        }
      }
      if (sensor.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE)) {
        sensor.set_option(
          RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, auto_white_balance_ ? 1.0F : 0.0F);
      }
      if (!auto_white_balance_ && sensor.supports(RS2_OPTION_WHITE_BALANCE)) {
        sensor.set_option(RS2_OPTION_WHITE_BALANCE, static_cast<float>(white_balance_));
      }
    }
  }

  int target_width_{640};
  int target_height_{480};
  int target_fps_{60};
  int requested_width_{640};
  int requested_height_{480};
  int requested_fps_{60};
  double exposure_{40.0};
  double gain_{64.0};
  bool auto_exposure_{false};
  bool auto_white_balance_{true};
  double white_balance_{6500.0};
  bool use_poll_{false};
  bool use_max_native_fps_{false};
  double actual_fps_{0.0};
  rs2_format stream_format_{RS2_FORMAT_YUYV};
  rs2::pipeline pipeline_;
  rs2::pipeline_profile profile_;
};
#endif

#ifdef HAVE_PYLON
class BaslerBackend : public CameraBackend
{
public:
  BaslerBackend(
    int width, int height, int fps, double exposure_us, double gain_db, bool use_full_sensor,
    bool centered_roi)
  : width_(width),
    height_(height),
    fps_(fps),
    exposure_us_(exposure_us),
    gain_db_(gain_db),
    use_full_sensor_(use_full_sensor),
    centered_roi_(centered_roi)
  {
    Pylon::PylonInitialize();
  }

  ~BaslerBackend() override
  {
    release();
    Pylon::PylonTerminate();
  }

  void initialize() override
  {
    camera_.Attach(Pylon::CTlFactory::GetInstance().CreateFirstDevice());
    camera_.Open();
    configure_camera();
    converter_.OutputPixelFormat = Pylon::PixelType_BGR8packed;
    converter_.OutputBitAlignment = Pylon::OutputBitAlignment_MsbAligned;
    camera_.StartGrabbing(Pylon::GrabStrategy_LatestImageOnly, Pylon::GrabLoop_ProvidedByInstantCamera);
  }

  bool get_frame(cv::Mat & frame) override
  {
    if (!camera_.IsGrabbing()) {
      return false;
    }
    Pylon::CGrabResultPtr result;
    camera_.RetrieveResult(1000, result, Pylon::TimeoutHandling_Return);
    if (!result || !result->GrabSucceeded()) {
      return false;
    }
    Pylon::CPylonImage image;
    converter_.Convert(image, result);
    cv::Mat view(
      static_cast<int>(result->GetHeight()),
      static_cast<int>(result->GetWidth()),
      CV_8UC3,
      image.GetBuffer());
    frame = view.clone();
    return true;
  }

  void release() override
  {
    if (camera_.IsGrabbing()) {
      camera_.StopGrabbing();
    }
    if (camera_.IsOpen()) {
      camera_.Close();
    }
  }

  double actual_fps() const override
  {
    return static_cast<double>(fps_);
  }

private:
  template<typename NodeT, typename ValueT>
  void set_if_writable(GenApi::INodeMap & node_map, const char * name, const ValueT & value)
  {
    NodeT node(node_map.GetNode(name));
    if (node && GenApi::IsWritable(node)) {
      node->SetValue(value);
    }
  }

  void configure_camera()
  {
    auto & node_map = camera_.GetNodeMap();
    set_if_writable<GenApi::CBooleanPtr>(node_map, "AcquisitionFrameRateEnable", true);
    set_if_writable<GenApi::CFloatPtr>(node_map, "AcquisitionFrameRate", static_cast<double>(fps_));
    set_if_writable<GenApi::CEnumerationPtr>(node_map, "ExposureAuto", "Off");
    set_if_writable<GenApi::CFloatPtr>(node_map, "ExposureTime", exposure_us_);
    set_if_writable<GenApi::CFloatPtr>(node_map, "Gain", gain_db_);

    if (!use_full_sensor_) {
      GenApi::CIntegerPtr width_node(node_map.GetNode("Width"));
      GenApi::CIntegerPtr height_node(node_map.GetNode("Height"));
      GenApi::CIntegerPtr offset_x(node_map.GetNode("OffsetX"));
      GenApi::CIntegerPtr offset_y(node_map.GetNode("OffsetY"));
      if (width_node && GenApi::IsWritable(width_node)) {
        width_node->SetValue(std::min(width_node->GetMax(), static_cast<int64_t>(width_)));
      }
      if (height_node && GenApi::IsWritable(height_node)) {
        height_node->SetValue(std::min(height_node->GetMax(), static_cast<int64_t>(height_)));
      }
      if (centered_roi_ && offset_x && offset_y && width_node && height_node) {
        const int64_t centered_x =
          std::max<int64_t>(0, (width_node->GetMax() - width_node->GetValue()) / 2);
        const int64_t centered_y =
          std::max<int64_t>(0, (height_node->GetMax() - height_node->GetValue()) / 2);
        if (GenApi::IsWritable(offset_x)) {
          offset_x->SetValue(centered_x);
        }
        if (GenApi::IsWritable(offset_y)) {
          offset_y->SetValue(centered_y);
        }
      }
    }
  }

  int width_{640};
  int height_{640};
  int fps_{60};
  double exposure_us_{5000.0};
  double gain_db_{5.0};
  bool use_full_sensor_{true};
  bool centered_roi_{true};
  Pylon::CInstantCamera camera_;
  Pylon::CImageFormatConverter converter_;
};
#endif

}  // namespace

CameraNode::CameraNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("camera_node", options),
  last_warning_time_(this->get_clock()->now())
{
  this->declare_parameter<std::string>("camera_type", "basler");
  this->declare_parameter<int>("imgsz_width", 640);
  this->declare_parameter<int>("imgsz_height", 640);
  this->declare_parameter<double>("basler_exposure_us", 5000.0);
  this->declare_parameter<double>("basler_gain_db", 5.0);
  this->declare_parameter<int>("basler_fps", 60);
  this->declare_parameter<bool>("basler_use_full_sensor", true);
  this->declare_parameter<bool>("basler_centered_roi", true);
  this->declare_parameter<int>("basler_binning", 1);
  this->declare_parameter<int>("basler_decimation", 1);
  this->declare_parameter<bool>("basler_auto_exposure", false);
  this->declare_parameter<bool>("basler_auto_gain", false);
  this->declare_parameter<bool>("basler_auto_white_balance", false);
  this->declare_parameter<double>("basler_balance_red", 1.6);
  this->declare_parameter<double>("basler_balance_green", 1.0);
  this->declare_parameter<double>("basler_balance_blue", 2.2);
  this->declare_parameter<bool>("basler_enable_grab_thread", true);
  this->declare_parameter<double>("basler_queue_timeout_ms", 50.0);
  this->declare_parameter<int>("realsense_width", 640);
  this->declare_parameter<int>("realsense_height", 480);
  this->declare_parameter<int>("realsense_fps", 60);
  this->declare_parameter<double>("realsense_exposure", 40.0);
  this->declare_parameter<double>("realsense_gain", 64.0);
  this->declare_parameter<bool>("realsense_auto_exposure", false);
  this->declare_parameter<bool>("realsense_auto_white_balance", true);
  this->declare_parameter<double>("realsense_white_balance", 6500.0);
  this->declare_parameter<bool>("realsense_use_poll", false);
  this->declare_parameter<bool>("realsense_use_max_native_fps", false);

  camera_type_ = this->get_parameter("camera_type").as_string();
  const int imgsz_width = this->get_parameter("imgsz_width").as_int();
  const int imgsz_height = this->get_parameter("imgsz_height").as_int();

  if (camera_type_ == "basler") {
#ifdef HAVE_PYLON
    backend_ = std::make_shared<BaslerBackend>(
      imgsz_width,
      imgsz_height,
      this->get_parameter("basler_fps").as_int(),
      this->get_parameter("basler_exposure_us").as_double(),
      this->get_parameter("basler_gain_db").as_double(),
      this->get_parameter("basler_use_full_sensor").as_bool(),
      this->get_parameter("basler_centered_roi").as_bool());
#else
    backend_ = std::make_shared<UnsupportedCameraBackend>(
      "Basler backend requested but Pylon SDK headers/libs were not available at build time.");
#endif
  } else if (camera_type_ == "realsense") {
#ifdef HAVE_LIBREALSENSE2
    backend_ = std::make_shared<RealSenseBackend>(
      this->get_parameter("realsense_width").as_int(),
      this->get_parameter("realsense_height").as_int(),
      this->get_parameter("realsense_fps").as_int(),
      this->get_parameter("realsense_exposure").as_double(),
      this->get_parameter("realsense_gain").as_double(),
      this->get_parameter("realsense_auto_exposure").as_bool(),
      this->get_parameter("realsense_auto_white_balance").as_bool(),
      this->get_parameter("realsense_white_balance").as_double(),
      this->get_parameter("realsense_use_poll").as_bool(),
      this->get_parameter("realsense_use_max_native_fps").as_bool());
#else
    backend_ = std::make_shared<UnsupportedCameraBackend>(
      "RealSense backend requested but librealsense2 headers/libs were not available at build time.");
#endif
  } else {
    throw std::runtime_error("Unknown camera type: " + camera_type_);
  }

  backend_->initialize();
  fps_ = backend_->actual_fps() > 0.0 ? backend_->actual_fps() :
    static_cast<double>((camera_type_ == "basler") ?
    this->get_parameter("basler_fps").as_int() : this->get_parameter("realsense_fps").as_int());

  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
  image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", qos);
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / std::max(1.0, fps_)),
    std::bind(&CameraNode::timer_callback, this));

  RCLCPP_INFO(this->get_logger(), "Initialized %s camera node at %.2f FPS", camera_type_.c_str(), fps_);
}

CameraNode::~CameraNode()
{
  if (backend_) {
    backend_->release();
  }
}

void CameraNode::timer_callback()
{
  cv::Mat frame;
  if (!backend_->get_frame(frame)) {
    publish_no_frame_warning();
    return;
  }

  auto msg = to_image_message(frame);
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

sensor_msgs::msg::Image::UniquePtr CameraNode::to_image_message(const cv::Mat & frame) const
{
  if (frame.empty()) {
    return nullptr;
  }

  auto msg = std::make_unique<sensor_msgs::msg::Image>();
  msg->header.stamp = this->get_clock()->now();
  msg->header.frame_id = camera_type_;
  msg->height = static_cast<uint32_t>(frame.rows);
  msg->width = static_cast<uint32_t>(frame.cols);
  msg->encoding = sensor_msgs::image_encodings::BGR8;
  msg->is_bigendian = false;
  msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(frame.step);
  const std::size_t byte_count = frame.total() * frame.elemSize();
  msg->data.resize(byte_count);
  std::memcpy(msg->data.data(), frame.data, byte_count);
  return msg;
}

}  // namespace yolov8_ros_cpp

RCLCPP_COMPONENTS_REGISTER_NODE(yolov8_ros_cpp::CameraNode)
