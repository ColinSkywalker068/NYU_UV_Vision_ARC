/// @file vision_node_helpers.cpp
/// @brief Shared helpers for detector and perception node logging, images, and previews.

#include "vision_node_helpers.hpp"

#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sensor_msgs/image_encodings.hpp>
#include <sstream>

namespace yolo_cpp::vision_node_helpers
{

namespace
{

constexpr char kModelPathEnv[] = "CV_STM_MODEL_PATH";
constexpr char kDefaultModelPathFile[] = "default_model_path.txt";
constexpr char kPackageName[] = "yolo_cpp";

std::optional<std::string> non_empty_env(const char* name)
{
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::string(value);
}

std::string trim_ascii(std::string value)
{
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string resolve_default_entry(
  const std::filesystem::path& default_file_path, const std::string& entry)
{
  const std::filesystem::path model_path(entry);
  if (model_path.is_absolute()) {
    return model_path.string();
  }
  return (default_file_path.parent_path() / model_path).lexically_normal().string();
}

std::optional<std::string> read_default_model_path_file(const std::filesystem::path& path)
{
  std::ifstream stream(path);
  if (!stream) {
    return std::nullopt;
  }

  std::string line;
  while (std::getline(stream, line)) {
    const auto trimmed = trim_ascii(line);
    if (!trimmed.empty() && trimmed.front() != '#') {
      return resolve_default_entry(path, trimmed);
    }
  }
  return std::nullopt;
}

std::optional<std::string> default_model_path_from_file()
{
  const std::filesystem::path cwd_candidate =
    std::filesystem::current_path() / "yolo_cpp" / "config" / kDefaultModelPathFile;
  if (const auto path = read_default_model_path_file(cwd_candidate)) {
    return path;
  }

  try {
    const std::filesystem::path share_candidate =
      std::filesystem::path(ament_index_cpp::get_package_share_directory(kPackageName)) / "config"
      / kDefaultModelPathFile;
    if (const auto path = read_default_model_path_file(share_candidate)) {
      return path;
    }
  } catch (const std::exception&) {
  }

  const std::filesystem::path source_candidate =
    std::filesystem::path(__FILE__).parent_path().parent_path() / "config" / kDefaultModelPathFile;
  return read_default_model_path_file(source_candidate);
}

}  // namespace

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
  out << std::put_time(&local_tm, "%Y-%m-%dT%H:%M:%S") << '.' << std::setw(3) << std::setfill('0')
      << milliseconds;
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

std::string resolve_model_path(const std::string& explicit_model_path)
{
  const auto trimmed_explicit = trim_ascii(explicit_model_path);
  if (!trimmed_explicit.empty()) {
    return trimmed_explicit;
  }
  if (const auto env_path = non_empty_env(kModelPathEnv)) {
    return *env_path;
  }
  return default_model_path_from_file().value_or(std::string{});
}

sensor_msgs::msg::Image::UniquePtr frame_to_image_message(
  const FramePacket& packet, const rclcpp::Time& stamp)
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

double rect_iou(const cv::Rect& lhs, const cv::Rect& rhs)
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
  const cv::Rect& rect, double scale_x, double scale_y, const cv::Rect& image_rect)
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
