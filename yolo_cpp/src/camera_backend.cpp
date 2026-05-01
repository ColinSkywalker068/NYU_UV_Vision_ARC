#include "yolo_cpp/camera_backend.hpp"

#include <opencv2/imgproc.hpp>
#include <rclcpp/node.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef HAVE_LIBREALSENSE2
#include <librealsense2/rs.hpp>
#endif

#ifdef HAVE_PYLON
#include <GenApi/GenApi.h>
#include <pylon/PylonIncludes.h>
#endif

namespace yolo_cpp
{

namespace
{

class UnsupportedCameraBackend : public CameraBackend
{
public:
  explicit UnsupportedCameraBackend(std::string reason, std::string source_id)
  : reason_(std::move(reason)), source_id_(std::move(source_id)) {}

  void initialize() override
  {
    throw std::runtime_error(reason_);
  }

  bool capture(FramePacket &) override
  {
    return false;
  }

  void release() override {}
  double actual_fps() const override {return 0.0;}
  const std::string & source_id() const override {return source_id_;}

private:
  std::string reason_;
  std::string source_id_;
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

class RealSenseBackend final : public CameraBackend
{
public:
  explicit RealSenseBackend(const CameraConfig & config)
  : config_(config)
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
    config.enable_stream(
      RS2_STREAM_COLOR, requested_width_, requested_height_, stream_format_, requested_fps_);
    profile_ = pipeline_.start(config);
    configure_sensor();

    auto stream_profile = profile_.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    actual_fps_ = static_cast<double>(stream_profile.fps());
    converted_bgr_.create(stream_profile.height(), stream_profile.width(), CV_8UC3);
  }

  bool capture(FramePacket & packet) override
  {
    rs2::frameset frames;
    if (config_.realsense_use_poll) {
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
    const auto * data = static_cast<const std::uint8_t *>(color_frame.get_data());
    packet.capture_time_ns = static_cast<std::int64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    packet.source_id = source_id_;

    if (stream_format_ == RS2_FORMAT_BGR8) {
      auto owner = std::make_shared<rs2::video_frame>(color_frame);
      packet.frame = cv::Mat(
        height, width, CV_8UC3, const_cast<std::uint8_t *>(data), cv::Mat::AUTO_STEP);
      packet.pixel_format = PixelFormat::kBgr8;
      packet.storage_kind = BackingStoreKind::kHost;
      packet.owner = owner;
      return true;
    }

    if (stream_format_ == RS2_FORMAT_RGB8) {
      cv::Mat rgb(
        height, width, CV_8UC3, const_cast<std::uint8_t *>(data), cv::Mat::AUTO_STEP);
      cv::cvtColor(rgb, converted_bgr_, cv::COLOR_RGB2BGR);
    } else {
      cv::Mat yuyv(
        height, width, CV_8UC2, const_cast<std::uint8_t *>(data), cv::Mat::AUTO_STEP);
      cv::cvtColor(yuyv, converted_bgr_, cv::COLOR_YUV2BGR_YUY2);
    }

    packet.frame = converted_bgr_;
    packet.pixel_format = PixelFormat::kBgr8;
    packet.storage_kind = BackingStoreKind::kHost;
    packet.owner.reset();
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

  const std::string & source_id() const override
  {
    return source_id_;
  }

private:
  void discover_best_stream(const rs2::context & ctx)
  {
    static const std::vector<std::string> preferred_formats{"bgr8", "rgb8", "yuyv"};
    requested_width_ = config_.realsense_width;
    requested_height_ = config_.realsense_height;
    requested_fps_ = config_.realsense_fps;
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
          if (
            video.width() != config_.realsense_width ||
            video.height() != config_.realsense_height)
          {
            continue;
          }
          const int fps = video.fps();
          if (!config_.realsense_use_max_native_fps && fps < config_.realsense_fps) {
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
        sensor.set_option(
          RS2_OPTION_ENABLE_AUTO_EXPOSURE, config_.realsense_auto_exposure ? 1.0F : 0.0F);
      }
      if (!config_.realsense_auto_exposure) {
        if (sensor.supports(RS2_OPTION_EXPOSURE)) {
          sensor.set_option(RS2_OPTION_EXPOSURE, static_cast<float>(config_.realsense_exposure));
        }
        if (sensor.supports(RS2_OPTION_GAIN)) {
          sensor.set_option(RS2_OPTION_GAIN, static_cast<float>(config_.realsense_gain));
        }
      }
      if (sensor.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE)) {
        sensor.set_option(
          RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE,
          config_.realsense_auto_white_balance ? 1.0F : 0.0F);
      }
      if (!config_.realsense_auto_white_balance && sensor.supports(RS2_OPTION_WHITE_BALANCE)) {
        sensor.set_option(
          RS2_OPTION_WHITE_BALANCE, static_cast<float>(config_.realsense_white_balance));
      }
    }
  }

  CameraConfig config_;
  const std::string source_id_{"realsense"};
  int requested_width_{640};
  int requested_height_{480};
  int requested_fps_{60};
  double actual_fps_{0.0};
  rs2_format stream_format_{RS2_FORMAT_YUYV};
  rs2::pipeline pipeline_;
  rs2::pipeline_profile profile_;
  cv::Mat converted_bgr_;
};
#endif

#ifdef HAVE_PYLON
class BaslerBackend final : public CameraBackend
{
public:
  explicit BaslerBackend(const CameraConfig & config)
  : config_(config)
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
    camera_.StartGrabbing(
      Pylon::GrabStrategy_LatestImageOnly, Pylon::GrabLoop_ProvidedByInstantCamera);
  }

  bool capture(FramePacket & packet) override
  {
    if (!camera_.IsGrabbing()) {
      return false;
    }

    Pylon::CGrabResultPtr result;
    camera_.RetrieveResult(1000, result, Pylon::TimeoutHandling_Return);
    if (!result || !result->GrabSucceeded()) {
      return false;
    }

    auto image = std::make_shared<Pylon::CPylonImage>();
    converter_.Convert(*image, result);
    packet.frame = cv::Mat(
      static_cast<int>(result->GetHeight()),
      static_cast<int>(result->GetWidth()),
      CV_8UC3,
      image->GetBuffer());
    packet.pixel_format = PixelFormat::kBgr8;
    packet.storage_kind = BackingStoreKind::kHost;
    packet.capture_time_ns = static_cast<std::int64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    packet.source_id = source_id_;
    packet.owner = image;
    if (config_.basler_flip) {
      cv::flip(packet.frame, packet.frame, -1);
    }
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
    return static_cast<double>(config_.basler_fps);
  }

  const std::string & source_id() const override
  {
    return source_id_;
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
    set_if_writable<GenApi::CFloatPtr>(
      node_map, "AcquisitionFrameRate", static_cast<double>(config_.basler_fps));
    set_if_writable<GenApi::CEnumerationPtr>(node_map, "ExposureAuto", "Off");
    set_if_writable<GenApi::CFloatPtr>(node_map, "ExposureTime", config_.basler_exposure_us);
    set_if_writable<GenApi::CFloatPtr>(node_map, "Gain", config_.basler_gain_db);

    if (!config_.basler_use_full_sensor) {
      GenApi::CIntegerPtr width_node(node_map.GetNode("Width"));
      GenApi::CIntegerPtr height_node(node_map.GetNode("Height"));
      GenApi::CIntegerPtr offset_x(node_map.GetNode("OffsetX"));
      GenApi::CIntegerPtr offset_y(node_map.GetNode("OffsetY"));
      if (width_node && GenApi::IsWritable(width_node)) {
        width_node->SetValue(
          std::min(width_node->GetMax(), static_cast<int64_t>(config_.imgsz_width)));
      }
      if (height_node && GenApi::IsWritable(height_node)) {
        height_node->SetValue(
          std::min(height_node->GetMax(), static_cast<int64_t>(config_.imgsz_height)));
      }
      if (config_.basler_centered_roi && offset_x && offset_y && width_node && height_node) {
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

  CameraConfig config_;
  const std::string source_id_{"basler"};
  Pylon::CInstantCamera camera_;
  Pylon::CImageFormatConverter converter_;
};
#endif

}  // namespace

CameraConfig load_camera_config(rclcpp::Node & node)
{
  CameraConfig config{};
  node.declare_parameter<std::string>("camera_type", config.camera_type);
  node.declare_parameter<int>("imgsz_width", config.imgsz_width);
  node.declare_parameter<int>("imgsz_height", config.imgsz_height);
  node.declare_parameter<double>("basler_exposure_us", config.basler_exposure_us);
  node.declare_parameter<double>("basler_gain_db", config.basler_gain_db);
  node.declare_parameter<int>("basler_fps", config.basler_fps);
  node.declare_parameter<bool>("basler_flip", config.basler_flip);
  node.declare_parameter<bool>("basler_use_full_sensor", config.basler_use_full_sensor);
  node.declare_parameter<bool>("basler_centered_roi", config.basler_centered_roi);
  node.declare_parameter<int>("realsense_width", config.realsense_width);
  node.declare_parameter<int>("realsense_height", config.realsense_height);
  node.declare_parameter<int>("realsense_fps", config.realsense_fps);
  node.declare_parameter<double>("realsense_exposure", config.realsense_exposure);
  node.declare_parameter<double>("realsense_gain", config.realsense_gain);
  node.declare_parameter<bool>("realsense_auto_exposure", config.realsense_auto_exposure);
  node.declare_parameter<bool>(
    "realsense_auto_white_balance", config.realsense_auto_white_balance);
  node.declare_parameter<double>("realsense_white_balance", config.realsense_white_balance);
  node.declare_parameter<bool>("realsense_use_poll", config.realsense_use_poll);
  node.declare_parameter<bool>(
    "realsense_use_max_native_fps", config.realsense_use_max_native_fps);

  config.camera_type = node.get_parameter("camera_type").as_string();
  config.imgsz_width = node.get_parameter("imgsz_width").as_int();
  config.imgsz_height = node.get_parameter("imgsz_height").as_int();
  config.basler_exposure_us = node.get_parameter("basler_exposure_us").as_double();
  config.basler_gain_db = node.get_parameter("basler_gain_db").as_double();
  config.basler_fps = node.get_parameter("basler_fps").as_int();
  config.basler_flip = node.get_parameter("basler_flip").as_bool();
  config.basler_use_full_sensor = node.get_parameter("basler_use_full_sensor").as_bool();
  config.basler_centered_roi = node.get_parameter("basler_centered_roi").as_bool();
  config.realsense_width = node.get_parameter("realsense_width").as_int();
  config.realsense_height = node.get_parameter("realsense_height").as_int();
  config.realsense_fps = node.get_parameter("realsense_fps").as_int();
  config.realsense_exposure = node.get_parameter("realsense_exposure").as_double();
  config.realsense_gain = node.get_parameter("realsense_gain").as_double();
  config.realsense_auto_exposure = node.get_parameter("realsense_auto_exposure").as_bool();
  config.realsense_auto_white_balance =
    node.get_parameter("realsense_auto_white_balance").as_bool();
  config.realsense_white_balance = node.get_parameter("realsense_white_balance").as_double();
  config.realsense_use_poll = node.get_parameter("realsense_use_poll").as_bool();
  config.realsense_use_max_native_fps =
    node.get_parameter("realsense_use_max_native_fps").as_bool();
  return config;
}

std::shared_ptr<CameraBackend> create_camera_backend(
  const CameraConfig & config, const rclcpp::Logger &)
{
  if (config.camera_type == "basler") {
#ifdef HAVE_PYLON
    return std::make_shared<BaslerBackend>(config);
#else
    return std::make_shared<UnsupportedCameraBackend>(
      "Basler backend requested but Pylon SDK headers/libs were not available at build time.",
      "basler");
#endif
  }

  if (config.camera_type == "realsense") {
#ifdef HAVE_LIBREALSENSE2
    return std::make_shared<RealSenseBackend>(config);
#else
    return std::make_shared<UnsupportedCameraBackend>(
      "RealSense backend requested but librealsense2 headers/libs were not available at build time.",
      "realsense");
#endif
  }

  return std::make_shared<UnsupportedCameraBackend>(
    "Unknown camera type: " + config.camera_type, config.camera_type);
}

}  // namespace yolo_cpp
