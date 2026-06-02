/// @file deepstream_runtime_helpers.cpp
/// @brief Pure config and source-format helpers for DeepStream runtime assembly.

#include "deepstream_runtime_helpers.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>

namespace yolo_cpp::deepstream_helpers
{

std::string read_text_file(const std::string & path)
{
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("Failed to open DeepStream config template: " + path);
  }
  return {
    (std::istreambuf_iterator<char>(stream)),
    std::istreambuf_iterator<char>()};
}

void replace_all(std::string & text, const std::string & needle, const std::string & replacement)
{
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.replace(pos, needle.size(), replacement);
    pos += replacement.size();
  }
}

bool path_has_extension(const std::string & path, const std::string & extension)
{
  if (path.size() < extension.size()) {
    return false;
  }
  return std::equal(extension.rbegin(), extension.rend(), path.rbegin());
}

std::string make_video_caps_text(
  const std::string & media_type,
  const std::string & format,
  std::optional<int> width,
  std::optional<int> height,
  std::optional<int> fps)
{
  std::string caps = media_type;
  if (!format.empty()) {
    caps += ",format=" + format;
  }
  if (width.has_value() && height.has_value()) {
    caps += ",width=" + std::to_string(*width);
    caps += ",height=" + std::to_string(*height);
  }
  if (fps.has_value() && *fps > 0) {
    caps += ",framerate=" + std::to_string(*fps) + "/1";
  }
  return caps;
}

std::optional<int> positive_dimension(int value)
{
  if (value > 0) {
    return value;
  }
  return std::nullopt;
}

std::string normalize_format_token(std::string value)
{
  value.erase(
    std::remove_if(value.begin(), value.end(), [](unsigned char ch) {
      return std::isspace(ch) != 0 || ch == '-' || ch == '_';
    }),
    value.end());
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value;
}

namespace
{

int even_dimension(int value)
{
  value = std::max(2, value);
  if ((value % 2) != 0) {
    --value;
  }
  return std::max(2, value);
}

}  // namespace

std::optional<std::pair<int, int>> resolve_pre_mux_resize(
  const DeepStreamRuntime::Config & config)
{
  if (!config.pre_mux_resize) {
    return std::nullopt;
  }

  const int source_width = positive_dimension(config.source_width).value_or(config.imgsz_width);
  const int source_height = positive_dimension(config.source_height).value_or(config.imgsz_height);
  if (source_width <= 0 || source_height <= 0) {
    return std::nullopt;
  }

  int box_width = config.pre_mux_width > 0 ? config.pre_mux_width : config.imgsz_width;
  int box_height = config.pre_mux_height > 0 ? config.pre_mux_height : config.imgsz_height;
  if (config.pre_mux_width > 0 && config.pre_mux_height <= 0) {
    box_height = static_cast<int>(
      std::lround(static_cast<double>(source_height) * box_width / source_width));
  } else if (config.pre_mux_height > 0 && config.pre_mux_width <= 0) {
    box_width = static_cast<int>(
      std::lround(static_cast<double>(source_width) * box_height / source_height));
  }

  box_width = even_dimension(box_width);
  box_height = even_dimension(box_height);
  const double scale = std::min(
    static_cast<double>(box_width) / source_width,
    static_cast<double>(box_height) / source_height);
  if (scale <= 0.0) {
    return std::nullopt;
  }

  const int width = std::min(
    box_width,
    even_dimension(static_cast<int>(std::lround(source_width * scale))));
  const int height = std::min(
    box_height,
    even_dimension(static_cast<int>(std::lround(source_height * scale))));
  return std::make_pair(width, height);
}

SourceFormatSpec resolve_source_format(const std::string & value)
{
  const auto normalized = normalize_format_token(value);
  if (normalized == "bayerrggb8" || normalized == "rggb" || normalized == "bayerrg8") {
    return {"bayer_rggb8", "video/x-bayer", "rggb", "BayerRG8", true};
  }
  if (normalized == "bayerbggr8" || normalized == "bggr" || normalized == "bayerbg8") {
    return {"bayer_bggr8", "video/x-bayer", "bggr", "BayerBG8", true};
  }
  if (normalized == "bayergrbg8" || normalized == "grbg" || normalized == "bayergr8") {
    return {"bayer_grbg8", "video/x-bayer", "grbg", "BayerGR8", true};
  }
  if (normalized == "bayergbrg8" || normalized == "gbrg" || normalized == "bayergb8") {
    return {"bayer_gbrg8", "video/x-bayer", "gbrg", "BayerGB8", true};
  }
  if (normalized == "mono8" || normalized == "gray8" || normalized == "grey8") {
    return {"mono8", "video/x-raw", "GRAY8", "Mono8", false};
  }
  if (normalized == "yuy2" || normalized == "yuyv" || normalized == "yuv422") {
    return {"yuy2", "video/x-raw", "YUY2", "YUV422Packed", false};
  }
  if (normalized == "bgr" || normalized == "bgr8") {
    return {"bgr8", "video/x-raw", "BGR", "BGR8", false};
  }
  if (normalized == "rgb" || normalized == "rgb8") {
    return {"rgb8", "video/x-raw", "RGB", "RGB8", false};
  }

  throw std::runtime_error(
    "Unsupported Basler source pixel format '" + value +
    "'. Use bayer_rggb8, bayer_bggr8, bayer_grbg8, bayer_gbrg8, mono8, yuy2, bgr8, or rgb8.");
}

}  // namespace yolo_cpp::deepstream_helpers
