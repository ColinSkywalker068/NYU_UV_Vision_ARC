#ifndef YOLO_CPP_SRC_DEEPSTREAM_RUNTIME_HELPERS_HPP_
#define YOLO_CPP_SRC_DEEPSTREAM_RUNTIME_HELPERS_HPP_

#include "yolo_cpp/deepstream_runtime.hpp"

#include <optional>
#include <string>
#include <utility>

namespace yolo_cpp::deepstream_helpers
{

struct SourceFormatSpec
{
  std::string label;
  std::string media_type;
  std::string caps_format;
  std::string camera_pixel_format;
  bool needs_bayer_convert{false};
};

/// Read a text config template or throw a runtime error with path context.
std::string read_text_file(const std::string & path);

/// Replace every occurrence of needle with replacement inside text.
void replace_all(std::string & text, const std::string & needle, const std::string & replacement);

bool path_has_extension(const std::string & path, const std::string & extension);

std::string make_video_caps_text(
  const std::string & media_type,
  const std::string & format,
  std::optional<int> width,
  std::optional<int> height,
  std::optional<int> fps);

std::optional<int> positive_dimension(int value);
std::string normalize_format_token(std::string value);
std::optional<std::pair<int, int>> resolve_pre_mux_resize(
  const DeepStreamRuntime::Config & config);
SourceFormatSpec resolve_source_format(const std::string & value);

}  // namespace yolo_cpp::deepstream_helpers

#endif  // YOLO_CPP_SRC_DEEPSTREAM_RUNTIME_HELPERS_HPP_
