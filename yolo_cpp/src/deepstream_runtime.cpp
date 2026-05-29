/// @file deepstream_runtime.cpp
/// @brief Implements the optional asynchronous GStreamer/DeepStream inference runtime.

#include "yolo_cpp/deepstream_runtime.hpp"

#include "yolo_cpp/detection_postprocess.hpp"
#include "yolo_cpp/utils.hpp"

#include <rclcpp/logging.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/gstchildproxy.h>
#include <gst/video/video.h>
#include <gstnvdsinfer.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#endif

namespace yolo_cpp
{

namespace
{

/// Read a text config template or throw a runtime error with path context.
std::string read_text_file(const std::string & path)
{
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("Failed to open DeepStream config template: " + path);
  }
  std::string content(
    (std::istreambuf_iterator<char>(stream)),
    std::istreambuf_iterator<char>());
  return content;
}

/// Replace every occurrence of needle with replacement inside text.
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

struct SourceFormatSpec
{
  std::string label;
  std::string media_type;
  std::string caps_format;
  std::string camera_pixel_format;
  bool needs_bayer_convert{false};
};

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

#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
/// Format a GstClockTime for diagnostics without losing GST_CLOCK_TIME_NONE.
std::string gst_clock_to_string(GstClockTime value)
{
  if (value == GST_CLOCK_TIME_NONE) {
    return "GST_CLOCK_TIME_NONE";
  }
  return std::to_string(static_cast<std::uint64_t>(value));
}

/// Release a GStreamer request pad and null out the caller's pointer.
void release_request_pad(GstElement * element, GstPad *& pad)
{
  if (element == nullptr || pad == nullptr) {
    return;
  }
  gst_element_release_request_pad(element, pad);
  gst_object_unref(pad);
  pad = nullptr;
}

bool set_child_proxy_bool_if_present(GstElement * element, const char * name, bool value)
{
  if (element == nullptr || name == nullptr) {
    return false;
  }

  if (GParamSpec * pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
    GValue gvalue = G_VALUE_INIT;
    g_value_init(&gvalue, G_PARAM_SPEC_VALUE_TYPE(pspec));
    bool did_set = false;
    if (G_VALUE_HOLDS_BOOLEAN(&gvalue)) {
      g_value_set_boolean(&gvalue, value ? TRUE : FALSE);
      g_object_set_property(G_OBJECT(element), name, &gvalue);
      did_set = true;
    }
    g_value_unset(&gvalue);
    return did_set;
  }
  if (!GST_IS_CHILD_PROXY(element)) {
    return false;
  }

  GObject * target = nullptr;
  GParamSpec * pspec = nullptr;
  if (!gst_child_proxy_lookup(GST_CHILD_PROXY(element), name, &target, &pspec)) {
    return false;
  }

  GValue gvalue = G_VALUE_INIT;
  g_value_init(&gvalue, G_PARAM_SPEC_VALUE_TYPE(pspec));
  bool did_set = false;
  if (G_VALUE_HOLDS_BOOLEAN(&gvalue)) {
    g_value_set_boolean(&gvalue, value ? TRUE : FALSE);
    gst_child_proxy_set_property(GST_CHILD_PROXY(element), name, &gvalue);
    did_set = true;
  }
  g_value_unset(&gvalue);
  if (target != nullptr) {
    g_object_unref(target);
  }
  return did_set;
}

bool string_matches_enum_value(const GEnumValue * enum_value, const char * value)
{
  if (enum_value == nullptr || value == nullptr) {
    return false;
  }
  if (
    g_strcmp0(enum_value->value_name, value) == 0 ||
    g_strcmp0(enum_value->value_nick, value) == 0 ||
    g_ascii_strcasecmp(enum_value->value_name, value) == 0 ||
    g_ascii_strcasecmp(enum_value->value_nick, value) == 0)
  {
    return true;
  }

  const char * suffix = std::strrchr(enum_value->value_name, '_');
  return suffix != nullptr && g_ascii_strcasecmp(suffix + 1, value) == 0;
}

bool set_enum_value_from_string(GValue * gvalue, const char * value)
{
  GEnumClass * enum_class = G_ENUM_CLASS(g_type_class_ref(G_VALUE_TYPE(gvalue)));
  if (enum_class == nullptr) {
    return false;
  }

  GEnumValue * enum_value = g_enum_get_value_by_name(enum_class, value);
  if (enum_value == nullptr) {
    enum_value = g_enum_get_value_by_nick(enum_class, value);
  }
  if (enum_value == nullptr) {
    for (std::uint32_t i = 0; i < enum_class->n_values; ++i) {
      if (string_matches_enum_value(&enum_class->values[i], value)) {
        enum_value = &enum_class->values[i];
        break;
      }
    }
  }

  const bool did_set = enum_value != nullptr;
  if (did_set) {
    g_value_set_enum(gvalue, enum_value->value);
  }
  g_type_class_unref(enum_class);
  return did_set;
}

bool set_string_value(GValue * gvalue, const char * value)
{
  if (G_VALUE_HOLDS_STRING(gvalue)) {
    g_value_set_string(gvalue, value);
    return true;
  }
  if (G_VALUE_HOLDS_ENUM(gvalue)) {
    return gst_value_deserialize(gvalue, value) || set_enum_value_from_string(gvalue, value);
  }
  return gst_value_deserialize(gvalue, value);
}

bool set_numeric_value(GValue * gvalue, double value)
{
  if (G_VALUE_HOLDS_DOUBLE(gvalue)) {
    g_value_set_double(gvalue, value);
    return true;
  }
  if (G_VALUE_HOLDS_FLOAT(gvalue)) {
    g_value_set_float(gvalue, static_cast<float>(value));
    return true;
  }
  if (G_VALUE_HOLDS_ENUM(gvalue)) {
    g_value_set_enum(gvalue, static_cast<int>(value));
    return true;
  }
  if (G_VALUE_HOLDS_INT(gvalue)) {
    g_value_set_int(gvalue, static_cast<int>(value));
    return true;
  }
  if (G_VALUE_HOLDS_UINT(gvalue)) {
    g_value_set_uint(gvalue, static_cast<unsigned int>(std::max(0.0, value)));
    return true;
  }
  if (G_VALUE_HOLDS_INT64(gvalue)) {
    g_value_set_int64(gvalue, static_cast<gint64>(value));
    return true;
  }
  if (G_VALUE_HOLDS_UINT64(gvalue)) {
    g_value_set_uint64(gvalue, static_cast<guint64>(std::max(0.0, value)));
    return true;
  }
  if (G_VALUE_HOLDS_LONG(gvalue)) {
    g_value_set_long(gvalue, static_cast<long>(value));
    return true;
  }
  if (G_VALUE_HOLDS_ULONG(gvalue)) {
    g_value_set_ulong(gvalue, static_cast<unsigned long>(std::max(0.0, value)));
    return true;
  }
  return false;
}

bool set_child_proxy_string_if_present(GstElement * element, const char * name, const char * value)
{
  if (element == nullptr || name == nullptr || value == nullptr) {
    return false;
  }

  if (GParamSpec * pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
    GValue gvalue = G_VALUE_INIT;
    g_value_init(&gvalue, G_PARAM_SPEC_VALUE_TYPE(pspec));
    const bool did_set = set_string_value(&gvalue, value);
    if (did_set) {
      g_object_set_property(G_OBJECT(element), name, &gvalue);
    }
    g_value_unset(&gvalue);
    return did_set;
  }
  if (!GST_IS_CHILD_PROXY(element)) {
    return false;
  }

  GObject * target = nullptr;
  GParamSpec * pspec = nullptr;
  if (!gst_child_proxy_lookup(GST_CHILD_PROXY(element), name, &target, &pspec)) {
    return false;
  }

  GValue gvalue = G_VALUE_INIT;
  g_value_init(&gvalue, G_PARAM_SPEC_VALUE_TYPE(pspec));
  const bool did_set = set_string_value(&gvalue, value);
  if (did_set) {
    gst_child_proxy_set_property(GST_CHILD_PROXY(element), name, &gvalue);
  }
  g_value_unset(&gvalue);
  if (target != nullptr) {
    g_object_unref(target);
  }
  return did_set;
}

bool set_child_proxy_double_if_present(GstElement * element, const char * name, double value)
{
  if (element == nullptr || name == nullptr) {
    return false;
  }

  if (GParamSpec * pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
    GValue gvalue = G_VALUE_INIT;
    g_value_init(&gvalue, G_PARAM_SPEC_VALUE_TYPE(pspec));
    const bool did_set = set_numeric_value(&gvalue, value);
    if (did_set) {
      g_object_set_property(G_OBJECT(element), name, &gvalue);
    }
    g_value_unset(&gvalue);
    return did_set;
  }
  if (!GST_IS_CHILD_PROXY(element)) {
    return false;
  }

  GObject * target = nullptr;
  GParamSpec * pspec = nullptr;
  if (!gst_child_proxy_lookup(GST_CHILD_PROXY(element), name, &target, &pspec)) {
    return false;
  }

  GValue gvalue = G_VALUE_INIT;
  g_value_init(&gvalue, G_PARAM_SPEC_VALUE_TYPE(pspec));
  const bool did_set = set_numeric_value(&gvalue, value);
  if (did_set) {
    gst_child_proxy_set_property(GST_CHILD_PROXY(element), name, &gvalue);
  }
  g_value_unset(&gvalue);
  if (target != nullptr) {
    g_object_unref(target);
  }
  return did_set;
}

std::optional<double> get_numeric_value(const GValue * gvalue)
{
  if (G_VALUE_HOLDS_DOUBLE(gvalue)) {
    return g_value_get_double(gvalue);
  }
  if (G_VALUE_HOLDS_FLOAT(gvalue)) {
    return static_cast<double>(g_value_get_float(gvalue));
  }
  if (G_VALUE_HOLDS_INT(gvalue)) {
    return static_cast<double>(g_value_get_int(gvalue));
  }
  if (G_VALUE_HOLDS_UINT(gvalue)) {
    return static_cast<double>(g_value_get_uint(gvalue));
  }
  if (G_VALUE_HOLDS_INT64(gvalue)) {
    return static_cast<double>(g_value_get_int64(gvalue));
  }
  if (G_VALUE_HOLDS_UINT64(gvalue)) {
    return static_cast<double>(g_value_get_uint64(gvalue));
  }
  if (G_VALUE_HOLDS_LONG(gvalue)) {
    return static_cast<double>(g_value_get_long(gvalue));
  }
  if (G_VALUE_HOLDS_ULONG(gvalue)) {
    return static_cast<double>(g_value_get_ulong(gvalue));
  }
  return std::nullopt;
}

std::optional<std::string> get_string_value(const GValue * gvalue)
{
  if (G_VALUE_HOLDS_STRING(gvalue)) {
    const char * value = g_value_get_string(gvalue);
    if (value != nullptr && std::strlen(value) > 0U) {
      return std::string(value);
    }
    return std::nullopt;
  }
  if (G_VALUE_HOLDS_ENUM(gvalue)) {
    GEnumClass * enum_class = G_ENUM_CLASS(g_type_class_ref(G_VALUE_TYPE(gvalue)));
    if (enum_class != nullptr) {
      GEnumValue * enum_value = g_enum_get_value(enum_class, g_value_get_enum(gvalue));
      std::optional<std::string> result;
      if (enum_value != nullptr) {
        result = enum_value->value_nick != nullptr ?
          std::string(enum_value->value_nick) :
          std::string(enum_value->value_name);
      }
      g_type_class_unref(enum_class);
      if (result.has_value() && !result->empty()) {
        return result;
      }
    }
  }

  gchar * serialized = gst_value_serialize(gvalue);
  if (serialized == nullptr) {
    return std::nullopt;
  }
  std::string result(serialized);
  g_free(serialized);
  if (result.empty()) {
    return std::nullopt;
  }
  return result;
}

std::optional<double> get_child_proxy_numeric_if_present(GstElement * element, const char * name)
{
  if (element == nullptr || name == nullptr) {
    return std::nullopt;
  }

  if (GParamSpec * pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
    GValue gvalue = G_VALUE_INIT;
    g_value_init(&gvalue, G_PARAM_SPEC_VALUE_TYPE(pspec));
    g_object_get_property(G_OBJECT(element), name, &gvalue);
    const auto value = get_numeric_value(&gvalue);
    g_value_unset(&gvalue);
    return value;
  }
  if (!GST_IS_CHILD_PROXY(element)) {
    return std::nullopt;
  }

  GObject * target = nullptr;
  GParamSpec * pspec = nullptr;
  if (!gst_child_proxy_lookup(GST_CHILD_PROXY(element), name, &target, &pspec)) {
    return std::nullopt;
  }

  GValue gvalue = G_VALUE_INIT;
  g_value_init(&gvalue, G_PARAM_SPEC_VALUE_TYPE(pspec));
  gst_child_proxy_get_property(GST_CHILD_PROXY(element), name, &gvalue);
  const auto value = get_numeric_value(&gvalue);
  g_value_unset(&gvalue);
  if (target != nullptr) {
    g_object_unref(target);
  }
  return value;
}

std::optional<std::string> get_child_proxy_string_if_present(GstElement * element, const char * name)
{
  if (element == nullptr || name == nullptr) {
    return std::nullopt;
  }

  if (GParamSpec * pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
    GValue gvalue = G_VALUE_INIT;
    g_value_init(&gvalue, G_PARAM_SPEC_VALUE_TYPE(pspec));
    g_object_get_property(G_OBJECT(element), name, &gvalue);
    const auto value = get_string_value(&gvalue);
    g_value_unset(&gvalue);
    return value;
  }
  if (!GST_IS_CHILD_PROXY(element)) {
    return std::nullopt;
  }

  GObject * target = nullptr;
  GParamSpec * pspec = nullptr;
  if (!gst_child_proxy_lookup(GST_CHILD_PROXY(element), name, &target, &pspec)) {
    return std::nullopt;
  }

  GValue gvalue = G_VALUE_INIT;
  g_value_init(&gvalue, G_PARAM_SPEC_VALUE_TYPE(pspec));
  gst_child_proxy_get_property(GST_CHILD_PROXY(element), name, &gvalue);
  const auto value = get_string_value(&gvalue);
  g_value_unset(&gvalue);
  if (target != nullptr) {
    g_object_unref(target);
  }
  return value;
}

std::string caps_to_string(GstCaps * caps)
{
  if (caps == nullptr) {
    return "none";
  }
  gchar * text = gst_caps_to_string(caps);
  if (text == nullptr) {
    return "unknown";
  }
  std::string result(text);
  g_free(text);
  return result;
}

std::string current_caps_for_pad(GstElement * element, const char * pad_name)
{
  if (element == nullptr || pad_name == nullptr) {
    return "none";
  }
  GstPad * pad = gst_element_get_static_pad(element, pad_name);
  if (pad == nullptr) {
    return "none";
  }
  GstCaps * caps = gst_pad_get_current_caps(pad);
  if (caps == nullptr) {
    caps = gst_pad_query_caps(pad, nullptr);
  }
  const std::string result = caps_to_string(caps);
  if (caps != nullptr) {
    gst_caps_unref(caps);
  }
  gst_object_unref(pad);
  return result;
}

std::string collect_bus_diagnostics(GstBus * bus)
{
  if (bus == nullptr) {
    return "";
  }

  std::ostringstream diagnostics;
  for (int i = 0; i < 8; ++i) {
    GstMessage * message = gst_bus_timed_pop_filtered(
      bus,
      100 * GST_MSECOND,
      static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING));
    if (message == nullptr) {
      break;
    }

    GstObject * source = GST_MESSAGE_SRC(message);
    const char * source_name = source != nullptr ? GST_OBJECT_NAME(source) : nullptr;
    GError * error = nullptr;
    gchar * debug = nullptr;
    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
      gst_message_parse_error(message, &error, &debug);
      diagnostics << " error";
    } else {
      gst_message_parse_warning(message, &error, &debug);
      diagnostics << " warning";
    }
    diagnostics << " from " << (source_name != nullptr ? source_name : "unknown") << ": "
                << (error != nullptr ? error->message : "unknown");
    if (debug != nullptr && std::strlen(debug) > 0U) {
      diagnostics << " debug=" << debug;
    }
    if (i < 7) {
      diagnostics << ';';
    }

    if (error != nullptr) {
      g_error_free(error);
    }
    if (debug != nullptr) {
      g_free(debug);
    }
    gst_message_unref(message);
  }

  return diagnostics.str();
}
#endif

}  // namespace

/// Private runtime state because GStreamer and DeepStream headers are optional.
struct DeepStreamRuntime::Impl
{
  Impl(rclcpp::Logger logger_in, Config config_in, ResultCallback callback_in)
  : logger(std::move(logger_in)),
    config(std::move(config_in)),
    callback(std::move(callback_in))
  {
  }

  rclcpp::Logger logger;
  Config config;
  ResultCallback callback;
  std::string backend_name{"deepstream"};

#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
  GstElement * pipeline{nullptr};
  GstElement * source{nullptr};
  GstElement * source_caps{nullptr};
  GstElement * source_bayer_convert{nullptr};
  GstElement * source_flip{nullptr};
  GstElement * tee{nullptr};
  GstElement * infer_queue{nullptr};
  GstElement * infer_convert{nullptr};
  GstElement * infer_cpu_caps{nullptr};
  GstElement * infer_nvconvert{nullptr};
  GstElement * infer_nvmm_caps{nullptr};
  GstElement * streammux{nullptr};
  GstElement * nvinfer{nullptr};
  GstElement * infer_post_queue{nullptr};
  GstElement * infer_sink_elem{nullptr};
  GstElement * color_queue{nullptr};
  GstElement * color_convert{nullptr};
  GstElement * color_caps{nullptr};
  GstElement * color_sink_elem{nullptr};
  GstBus * bus{nullptr};
  GstAppSink * infer_sink{nullptr};
  GstAppSink * color_sink{nullptr};
  GstPad * tee_infer_pad{nullptr};
  GstPad * streammux_sink_pad{nullptr};
  GstPad * tee_color_pad{nullptr};
  std::filesystem::path runtime_config_path;
  std::thread infer_thread;
  std::thread color_thread;
  std::thread bus_thread;
  std::atomic<bool> running{false};
  std::mutex frame_cache_mutex;
  std::map<std::uint64_t, FramePacket> frame_cache;
  int num_classes{1};
  bool task_obb{true};
  bool end2end{false};
  std::optional<bool> export_includes_nms;
  double source_actual_fps{0.0};
  std::string requested_source_caps;
  std::string requested_infer_cpu_caps;
  std::string requested_infer_nvmm_caps;
  std::string requested_color_caps;
  std::string fps_cap_status{"unknown"};
#endif

  ~Impl()
  {
    stop();
  }

  /// Validate dependencies, materialize config, build the graph, and start worker threads.
  void start()
  {
#if !defined(HAVE_GSTREAMER) || !defined(HAVE_DEEPSTREAM)
    throw std::runtime_error(
            "DeepStream runtime unavailable: build missing GStreamer or DeepStream development headers/libs");
#else
    validate_runtime();
    materialize_config();
    build_pipeline();
    start_threads();
#endif
  }

  /// Stop workers and release GStreamer objects in reverse ownership order.
  void stop()
  {
#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
    running = false;

    if (infer_thread.joinable()) {
      infer_thread.join();
    }
    if (color_thread.joinable()) {
      color_thread.join();
    }
    if (bus_thread.joinable()) {
      bus_thread.join();
    }

    if (pipeline != nullptr) {
      gst_element_set_state(pipeline, GST_STATE_NULL);
    }

    release_request_pad(tee, tee_infer_pad);
    release_request_pad(streammux, streammux_sink_pad);
    release_request_pad(tee, tee_color_pad);

    if (bus != nullptr) {
      gst_object_unref(bus);
      bus = nullptr;
    }
    if (pipeline != nullptr) {
      gst_object_unref(pipeline);
      pipeline = nullptr;
    }

    infer_sink = nullptr;
    color_sink = nullptr;

    if (!runtime_config_path.empty()) {
      std::error_code ec;
      std::filesystem::remove(runtime_config_path, ec);
    }
#endif
  }

#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
  /// Verify GStreamer/DeepStream elements and read model metadata needed for parsing.
  void validate_runtime()
  {
    GError * error = nullptr;
    if (!gst_init_check(nullptr, nullptr, &error)) {
      const std::string message = error != nullptr ? error->message : "unknown gst init error";
      if (error != nullptr) {
        g_error_free(error);
      }
      throw std::runtime_error("Failed to initialize GStreamer: " + message);
    }

    const auto metadata = read_engine_metadata_from_file(config.model_path);
    num_classes = metadata.class_names.empty() ? 1 : static_cast<int>(metadata.class_names.size());
    task_obb = metadata.task.value_or(config.task) == "obb";
    end2end = metadata.end2end;
    export_includes_nms = metadata.export_includes_nms;
    if (metadata.class_names.empty() && task_obb && num_classes < 2) {
      num_classes = 2;
      RCLCPP_WARN(
        logger,
        "Model metadata did not expose class names from %s; assuming 2 classes for OBB armor model",
        config.model_path.c_str());
    }

    const auto source_format = resolve_source_format(config.source_pixel_format);
    std::vector<std::string> required_elements = {
      config.source_element,
      "videoflip",
      "tee",
      "queue",
      "videoconvert",
      "nvvideoconvert",
      "nvstreammux",
      "nvinfer",
      "appsink",
    };
    if (source_format.needs_bayer_convert) {
      required_elements.push_back("bayer2rgb");
    }
    for (const auto & element_name : required_elements) {
      GstElementFactory * factory = gst_element_factory_find(element_name.c_str());
      if (factory == nullptr) {
        throw std::runtime_error(
                "Required GStreamer element not found: " + element_name +
                ". Check DeepStream and Basler plugin installation.");
      }
      gst_object_unref(factory);
    }
  }

  /// Render the nvinfer config template into a temporary runtime config file.
  void materialize_config()
  {
    std::string config_text = read_text_file(config.config_template_path);
    const auto model_source_config = path_has_extension(config.model_path, ".onnx") ?
      ("onnx-file=" + config.model_path + "\nmodel-engine-file=" + config.model_path + ".engine") :
      ("model-engine-file=" + config.model_path);
    replace_all(config_text, "__MODEL_SOURCE_CONFIG__", model_source_config);
    replace_all(config_text, "__MODEL_ENGINE_FILE__", config.model_path);
    replace_all(config_text, "__INPUT_WIDTH__", std::to_string(config.imgsz_width));
    replace_all(config_text, "__INPUT_HEIGHT__", std::to_string(config.imgsz_height));
    replace_all(config_text, "__NUM_CLASSES__", std::to_string(std::max(1, num_classes)));
    replace_all(config_text, "__CONF_THRESHOLD__", std::to_string(config.conf_threshold));
    replace_all(config_text, "__NETWORK_MODE__", config.half ? "2" : "0");
    replace_all(config_text, "__OUTPUT_BLOB_NAMES__", "output0");

    runtime_config_path = std::filesystem::temp_directory_path() /
      "yolo_cpp_deepstream_runtime_config.txt";
    std::ofstream stream(runtime_config_path);
    if (!stream) {
      throw std::runtime_error(
              "Failed to create runtime DeepStream config: " + runtime_config_path.string());
    }
    stream << config_text;
  }

  /// Create a required GStreamer element or throw with the missing factory name.
  GstElement * make_element(const char * factory, const char * name)
  {
    GstElement * element = gst_element_factory_make(factory, name);
    if (element == nullptr) {
      throw std::runtime_error(
              "Failed to create GStreamer element '" + std::string(factory) + "'");
    }
    return element;
  }

  bool set_camera_enum_off(const char * name)
  {
    return set_child_proxy_string_if_present(source, name, "Off") ||
           set_child_proxy_double_if_present(source, name, 0.0);
  }

  void configure_low_latency_queue(GstElement * queue)
  {
    if (queue == nullptr) {
      return;
    }
    g_object_set(
      G_OBJECT(queue),
      "max-size-buffers", 1,
      "max-size-bytes", 0,
      "max-size-time", 0,
      "leaky", 2,
      nullptr);
  }

  void configure_source_dimensions()
  {
    if (config.source_width > 0 &&
      !set_child_proxy_double_if_present(source, "cam::Width", config.source_width))
    {
      RCLCPP_WARN(
        logger,
        "Could not set Basler cam::Width=%d through pylonsrc",
        config.source_width);
    }
    if (config.source_height > 0 &&
      !set_child_proxy_double_if_present(source, "cam::Height", config.source_height))
    {
      RCLCPP_WARN(
        logger,
        "Could not set Basler cam::Height=%d through pylonsrc",
        config.source_height);
    }
  }

  void configure_basler_source(const SourceFormatSpec & source_format)
  {
    if (!source_format.camera_pixel_format.empty() &&
      !set_child_proxy_string_if_present(
        source, "cam::PixelFormat", source_format.camera_pixel_format.c_str()))
    {
      RCLCPP_WARN(
        logger,
        "Could not set Basler cam::PixelFormat=%s through pylonsrc",
        source_format.camera_pixel_format.c_str());
    }
    configure_source_dimensions();
    set_child_proxy_string_if_present(source, "cam::ExposureMode", "Timed");
    if (!set_camera_enum_off("cam::ExposureAuto")) {
      RCLCPP_WARN(logger, "Could not set Basler cam::ExposureAuto=Off through pylonsrc");
    }
    if (!set_child_proxy_double_if_present(
        source, "cam::ExposureTime", config.source_exposure_us))
    {
      RCLCPP_WARN(
        logger,
        "Could not set Basler cam::ExposureTime=%.2f through pylonsrc",
        config.source_exposure_us);
    }

    if (!set_camera_enum_off("cam::GainAuto")) {
      RCLCPP_WARN(logger, "Could not set Basler cam::GainAuto=Off through pylonsrc");
    }
    set_child_proxy_string_if_present(source, "cam::GainSelector", "All");
    if (!set_child_proxy_double_if_present(source, "cam::Gain", config.source_gain_db)) {
      RCLCPP_WARN(
        logger,
        "Could not set Basler cam::Gain=%.2f through pylonsrc",
        config.source_gain_db);
    }
  }

  std::optional<double> read_source_actual_fps() const
  {
    if (auto fps = get_child_proxy_numeric_if_present(
        source, "cam::BslResultingAcquisitionFrameRate");
      fps.has_value() && *fps > 0.0)
    {
      return fps;
    }
    if (auto fps = get_child_proxy_numeric_if_present(source, "cam::ResultingFrameRate");
      fps.has_value() && *fps > 0.0)
    {
      return fps;
    }
    if (auto fps = get_child_proxy_numeric_if_present(source, "cam::ResultingFrameRateAbs");
      fps.has_value() && *fps > 0.0)
    {
      return fps;
    }
    return std::nullopt;
  }

  void query_source_actuals()
  {
    const auto fps = read_source_actual_fps();
    source_actual_fps = fps.value_or(static_cast<double>(std::max(1, config.source_fps)));
    const auto pixel_format = get_child_proxy_string_if_present(source, "cam::PixelFormat");

    const auto width = get_child_proxy_numeric_if_present(source, "cam::Width");
    const auto height = get_child_proxy_numeric_if_present(source, "cam::Height");
    if (width.has_value() && *width > 0.0) {
      config.source_width = static_cast<int>(*width);
    }
    if (height.has_value() && *height > 0.0) {
      config.source_height = static_cast<int>(*height);
    }

    RCLCPP_INFO(
      logger,
      "DeepStream Basler source %s: %dx%d pixel_format=%s @ %.2f FPS (full_sensor=%s, fps_cap=%s)",
      fps.has_value() ? "reported" : "fallback",
      config.source_width,
      config.source_height,
      pixel_format.value_or("unknown").c_str(),
      source_actual_fps,
      config.source_use_full_sensor ? "true" : "false",
      fps_cap_status.c_str());
    RCLCPP_INFO(
      logger,
      "DeepStream Basler caps requested: source='%s' infer_cpu='%s' infer_nvmm='%s' color='%s'",
      requested_source_caps.c_str(),
      requested_infer_cpu_caps.c_str(),
      requested_infer_nvmm_caps.c_str(),
      requested_color_caps.empty() ? "disabled" : requested_color_caps.c_str());
    RCLCPP_INFO(
      logger,
      "DeepStream Basler caps negotiated: source='%s' infer_nvmm='%s' color='%s'",
      current_caps_for_pad(source_caps, "src").c_str(),
      current_caps_for_pad(infer_nvmm_caps, "src").c_str(),
      color_caps != nullptr ? current_caps_for_pad(color_caps, "src").c_str() : "disabled");
  }

  void build_pipeline()
  {
    pipeline = gst_pipeline_new("yolo-deepstream-pipeline");
    if (pipeline == nullptr) {
      throw std::runtime_error("Failed to create DeepStream pipeline");
    }

    const auto source_format = resolve_source_format(config.source_pixel_format);
    source = make_element(config.source_element.c_str(), "source");
    source_caps = make_element("capsfilter", "source_caps");
    if (source_format.needs_bayer_convert) {
      source_bayer_convert = make_element("bayer2rgb", "source_bayer_convert");
    }
    source_flip = make_element("videoflip", "source_flip");
    tee = make_element("tee", "tee");
    infer_queue = make_element("queue", "infer_queue");
    infer_convert = make_element("videoconvert", "infer_convert");
    infer_cpu_caps = make_element("capsfilter", "infer_cpu_caps");
    infer_nvconvert = make_element("nvvideoconvert", "infer_nvconvert");
    infer_nvmm_caps = make_element("capsfilter", "infer_nvmm_caps");
    streammux = make_element("nvstreammux", "streammux");
    nvinfer = make_element("nvinfer", "primary_infer");
    infer_post_queue = make_element("queue", "infer_post_queue");
    infer_sink_elem = make_element("appsink", "infer_sink");
    infer_sink = GST_APP_SINK(infer_sink_elem);

    const bool need_color_branch =
      config.enable_color_verification || config.debug_publish_image_raw;
    if (need_color_branch) {
      color_queue = make_element("queue", "color_queue");
      color_convert = make_element("videoconvert", "color_convert");
      color_caps = make_element("capsfilter", "color_caps");
      color_sink_elem = make_element("appsink", "color_sink");
      color_sink = GST_APP_SINK(color_sink_elem);
    }

    gst_bin_add_many(
      GST_BIN(pipeline),
      source,
      source_caps,
      source_flip,
      tee,
      infer_queue,
      infer_convert,
      infer_cpu_caps,
      infer_nvconvert,
      infer_nvmm_caps,
      streammux,
      nvinfer,
      infer_post_queue,
      infer_sink_elem,
      nullptr);
    if (source_bayer_convert != nullptr) {
      gst_bin_add(GST_BIN(pipeline), source_bayer_convert);
    }
    if (need_color_branch) {
      gst_bin_add_many(
        GST_BIN(pipeline),
        color_queue,
        color_convert,
        color_caps,
        color_sink_elem,
        nullptr);
    }

    const std::optional<int> source_width = positive_dimension(config.source_width);
    const std::optional<int> source_height = positive_dimension(config.source_height);
    const std::optional<int> processing_width =
      config.source_use_full_sensor ? std::nullopt : source_width;
    const std::optional<int> processing_height =
      config.source_use_full_sensor ? std::nullopt : source_height;

    if (g_object_class_find_property(G_OBJECT_GET_CLASS(source), "do-timestamp") != nullptr) {
      g_object_set(G_OBJECT(source), "do-timestamp", TRUE, nullptr);
    }
    configure_basler_source(source_format);
    configure_low_latency_queue(infer_queue);
    configure_low_latency_queue(infer_post_queue);

    bool uncapped_fps_applied = false;
    if (
      config.source_fps_uncapped &&
      set_child_proxy_bool_if_present(source, "cam::AcquisitionFrameRateEnable", false))
    {
      uncapped_fps_applied = true;
      fps_cap_status = "disabled camera AcquisitionFrameRate limit";
    } else if (config.source_fps_uncapped) {
      RCLCPP_WARN(
        logger,
        "Could not disable Basler AcquisitionFrameRateEnable through pylonsrc; requesting %d FPS through caps instead",
        config.source_fps);
      fps_cap_status = "uncapped requested, falling back to caps framerate";
    }

    std::optional<int> capped_fps;
    if (!config.source_fps_uncapped) {
      set_child_proxy_bool_if_present(source, "cam::AcquisitionFrameRateEnable", true);
      if (!set_child_proxy_double_if_present(
          source, "cam::AcquisitionFrameRate", static_cast<double>(config.source_fps)))
      {
        RCLCPP_WARN(
          logger,
          "Could not set Basler cam::AcquisitionFrameRate=%d through pylonsrc",
          config.source_fps);
      }
      capped_fps = config.source_fps;
      fps_cap_status = "camera AcquisitionFrameRate cap requested";
    } else if (!uncapped_fps_applied) {
      capped_fps = config.source_fps;
      fps_cap_status = "caps framerate fallback";
    }
    requested_source_caps = make_video_caps_text(
      source_format.media_type,
      source_format.caps_format,
      source_width,
      source_height,
      capped_fps);
    GstCaps * source_caps_value = gst_caps_from_string(requested_source_caps.c_str());
    if (source_caps_value == nullptr) {
      throw std::runtime_error("Failed to create Basler source caps: " + requested_source_caps);
    }
    g_object_set(G_OBJECT(source_caps), "caps", source_caps_value, nullptr);
    gst_caps_unref(source_caps_value);
    g_object_set(G_OBJECT(source_flip), "method", config.flip_source ? 2 : 0, nullptr);

    requested_infer_cpu_caps = make_video_caps_text(
      "video/x-raw", "RGBA", processing_width, processing_height, capped_fps);
    GstCaps * infer_cpu_caps_value = gst_caps_from_string(requested_infer_cpu_caps.c_str());
    if (infer_cpu_caps_value == nullptr) {
      throw std::runtime_error(
              "Failed to create DeepStream CPU conversion caps: " + requested_infer_cpu_caps);
    }
    g_object_set(G_OBJECT(infer_cpu_caps), "caps", infer_cpu_caps_value, nullptr);
    gst_caps_unref(infer_cpu_caps_value);

    const std::optional<int> infer_width =
      config.source_use_full_sensor ? std::nullopt : std::optional<int>(config.imgsz_width);
    const std::optional<int> infer_height =
      config.source_use_full_sensor ? std::nullopt : std::optional<int>(config.imgsz_height);
    requested_infer_nvmm_caps = make_video_caps_text(
      "video/x-raw(memory:NVMM)",
      "NV12",
      infer_width,
      infer_height,
      capped_fps);
    GstCaps * infer_caps_value = gst_caps_from_string(requested_infer_nvmm_caps.c_str());
    if (infer_caps_value == nullptr) {
      throw std::runtime_error("Failed to create DeepStream NVMM caps: " + requested_infer_nvmm_caps);
    }
    g_object_set(G_OBJECT(infer_nvmm_caps), "caps", infer_caps_value, nullptr);
    gst_caps_unref(infer_caps_value);

    g_object_set(
      G_OBJECT(streammux),
      "batch-size", 1,
      "width", config.imgsz_width,
      "height", config.imgsz_height,
      "live-source", TRUE,
      "enable-padding", TRUE,
      nullptr);
    g_object_set(
      G_OBJECT(nvinfer),
      "config-file-path", runtime_config_path.c_str(),
      nullptr);
    g_object_set(
      G_OBJECT(infer_sink_elem),
      "emit-signals", FALSE,
      "sync", FALSE,
      "max-buffers", 1,
      "drop", TRUE,
      nullptr);

    if (need_color_branch) {
      configure_low_latency_queue(color_queue);
      requested_color_caps = make_video_caps_text(
        "video/x-raw", "BGR", processing_width, processing_height, capped_fps);
      GstCaps * color_caps_value = gst_caps_from_string(requested_color_caps.c_str());
      if (color_caps_value == nullptr) {
        throw std::runtime_error("Failed to create DeepStream color caps: " + requested_color_caps);
      }
      g_object_set(G_OBJECT(color_caps), "caps", color_caps_value, nullptr);
      gst_caps_unref(color_caps_value);
      g_object_set(
        G_OBJECT(color_sink_elem),
        "emit-signals", FALSE,
        "sync", FALSE,
        "max-buffers", 4,
        "drop", TRUE,
        nullptr);
    }

    const gboolean source_linked = source_bayer_convert != nullptr ?
      gst_element_link_many(source, source_caps, source_bayer_convert, source_flip, tee, nullptr) :
      gst_element_link_many(source, source_caps, source_flip, tee, nullptr);
    if (!source_linked) {
      throw std::runtime_error(
              source_bayer_convert != nullptr ?
              "Failed to link DeepStream Bayer source path" :
              "Failed to link DeepStream source path");
    }
    if (!gst_element_link_many(
        infer_queue, infer_convert, infer_cpu_caps, infer_nvconvert, infer_nvmm_caps, nullptr))
    {
      throw std::runtime_error("Failed to link DeepStream inference pre-mux path");
    }
    if (!gst_element_link_many(streammux, nvinfer, infer_post_queue, infer_sink_elem, nullptr)) {
      throw std::runtime_error("Failed to link DeepStream inference path");
    }

    tee_infer_pad = gst_element_request_pad_simple(tee, "src_%u");
    GstPad * infer_queue_sink_pad = gst_element_get_static_pad(infer_queue, "sink");
    if (
      tee_infer_pad == nullptr || infer_queue_sink_pad == nullptr ||
      gst_pad_link(tee_infer_pad, infer_queue_sink_pad) != GST_PAD_LINK_OK)
    {
      release_request_pad(tee, tee_infer_pad);
      if (infer_queue_sink_pad != nullptr) {
        gst_object_unref(infer_queue_sink_pad);
      }
      throw std::runtime_error("Failed to link tee to inference queue");
    }
    gst_object_unref(infer_queue_sink_pad);

    GstPad * infer_src_pad = gst_element_get_static_pad(infer_nvmm_caps, "src");
    streammux_sink_pad = gst_element_request_pad_simple(streammux, "sink_0");
    if (
      infer_src_pad == nullptr || streammux_sink_pad == nullptr ||
      gst_pad_link(infer_src_pad, streammux_sink_pad) != GST_PAD_LINK_OK)
    {
      if (infer_src_pad != nullptr) {
        gst_object_unref(infer_src_pad);
      }
      release_request_pad(streammux, streammux_sink_pad);
      throw std::runtime_error("Failed to link inference branch to nvstreammux");
    }
    gst_object_unref(infer_src_pad);

    if (need_color_branch) {
      if (!gst_element_link_many(color_queue, color_convert, color_caps, color_sink_elem, nullptr)) {
        throw std::runtime_error("Failed to link DeepStream color branch");
      }
      tee_color_pad = gst_element_request_pad_simple(tee, "src_%u");
      GstPad * color_queue_sink_pad = gst_element_get_static_pad(color_queue, "sink");
      if (
        tee_color_pad == nullptr || color_queue_sink_pad == nullptr ||
        gst_pad_link(tee_color_pad, color_queue_sink_pad) != GST_PAD_LINK_OK)
      {
        release_request_pad(tee, tee_color_pad);
        if (color_queue_sink_pad != nullptr) {
          gst_object_unref(color_queue_sink_pad);
        }
        throw std::runtime_error("Failed to link tee to color queue");
      }
      gst_object_unref(color_queue_sink_pad);
    }

    bus = gst_element_get_bus(pipeline);
    if (bus == nullptr) {
      throw std::runtime_error("Failed to acquire pipeline bus");
    }

    const GstStateChangeReturn state_ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (state_ret == GST_STATE_CHANGE_FAILURE) {
      const std::string diagnostics = collect_bus_diagnostics(bus);
      throw std::runtime_error(
              "Failed to set DeepStream pipeline to PLAYING" +
              (diagnostics.empty() ? std::string{} : (": " + diagnostics)));
    }
    if (state_ret == GST_STATE_CHANGE_ASYNC) {
      const GstStateChangeReturn wait_ret = gst_element_get_state(
        pipeline, nullptr, nullptr, 2 * GST_SECOND);
      if (wait_ret == GST_STATE_CHANGE_FAILURE) {
        const std::string diagnostics = collect_bus_diagnostics(bus);
        throw std::runtime_error(
                "DeepStream pipeline failed while waiting for PLAYING" +
                (diagnostics.empty() ? std::string{} : (": " + diagnostics)));
      }
    }
    query_source_actuals();
  }

  void start_threads()
  {
    running = true;
    infer_thread = std::thread([this]() { this->infer_loop(); });
    if (color_sink != nullptr) {
      color_thread = std::thread([this]() { this->color_loop(); });
    }
    bus_thread = std::thread([this]() { this->bus_loop(); });
  }

  void bus_loop()
  {
    while (running) {
      GstMessage * message = gst_bus_timed_pop_filtered(
        bus,
        100 * GST_MSECOND,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_WARNING));
      if (message == nullptr) {
        continue;
      }

      switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR:
        {
          GError * error = nullptr;
          gchar * debug = nullptr;
          gst_message_parse_error(message, &error, &debug);
          RCLCPP_ERROR(
            logger,
            "DeepStream pipeline error: %s debug=%s",
            error != nullptr ? error->message : "unknown",
            debug != nullptr ? debug : "");
          if (error != nullptr) {
            g_error_free(error);
          }
          if (debug != nullptr) {
            g_free(debug);
          }
          break;
        }
        case GST_MESSAGE_WARNING:
        {
          GError * error = nullptr;
          gchar * debug = nullptr;
          gst_message_parse_warning(message, &error, &debug);
          RCLCPP_WARN(
            logger,
            "DeepStream pipeline warning: %s debug=%s",
            error != nullptr ? error->message : "unknown",
            debug != nullptr ? debug : "");
          if (error != nullptr) {
            g_error_free(error);
          }
          if (debug != nullptr) {
            g_free(debug);
          }
          break;
        }
        case GST_MESSAGE_EOS:
          RCLCPP_WARN(logger, "DeepStream pipeline reached EOS");
          break;
        default:
          break;
      }
      gst_message_unref(message);
    }
  }

  FramePacket sample_to_bgr_packet(GstSample * sample)
  {
    FramePacket packet{};
    if (sample == nullptr) {
      return packet;
    }

    GstBuffer * buffer = gst_sample_get_buffer(sample);
    GstCaps * caps = gst_sample_get_caps(sample);
    if (buffer == nullptr || caps == nullptr) {
      return packet;
    }

    GstVideoInfo info;
    gst_video_info_init(&info);
    if (!gst_video_info_from_caps(&info, caps)) {
      throw std::runtime_error("Failed to decode appsink BGR caps");
    }

    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, GST_MAP_READ)) {
      throw std::runtime_error("Failed to map BGR appsink buffer");
    }

    const int width = static_cast<int>(GST_VIDEO_INFO_WIDTH(&info));
    const int height = static_cast<int>(GST_VIDEO_INFO_HEIGHT(&info));
    const int stride = GST_VIDEO_INFO_PLANE_STRIDE(&info, 0);
    cv::Mat wrapped(height, width, CV_8UC3, map_info.data, stride);
    packet.frame = wrapped.clone();
    packet.pixel_format = PixelFormat::kBgr8;
    packet.storage_kind = BackingStoreKind::kHost;
    packet.capture_time_ns = static_cast<std::int64_t>(GST_BUFFER_PTS(buffer));
    packet.source_id = config.source_id;
    gst_buffer_unmap(buffer, &map_info);
    return packet;
  }

  void store_frame(FramePacket packet)
  {
    std::lock_guard<std::mutex> lock(frame_cache_mutex);
    const auto key = static_cast<std::uint64_t>(std::max<std::int64_t>(0, packet.capture_time_ns));
    frame_cache[key] = std::move(packet);
    while (frame_cache.size() > 8U) {
      frame_cache.erase(frame_cache.begin());
    }
  }

  std::optional<FramePacket> lookup_frame(std::uint64_t pts)
  {
    std::lock_guard<std::mutex> lock(frame_cache_mutex);
    if (frame_cache.empty()) {
      return std::nullopt;
    }
    auto it = frame_cache.find(pts);
    if (it != frame_cache.end()) {
      FramePacket packet = it->second;
      frame_cache.erase(it);
      return packet;
    }

    auto lower = frame_cache.lower_bound(pts);
    if (lower != frame_cache.end()) {
      return lower->second;
    }
    return frame_cache.rbegin()->second;
  }

  std::vector<int64_t> dims_to_shape(const NvDsInferDims & dims) const
  {
    std::vector<int64_t> shape;
    if (dims.numDims == 2) {
      shape = {1, dims.d[0], dims.d[1]};
    } else {
      shape.reserve(static_cast<std::size_t>(dims.numDims));
      for (unsigned int i = 0; i < dims.numDims; ++i) {
        shape.push_back(static_cast<int64_t>(dims.d[i]));
      }
    }
    return shape;
  }

  std::vector<float> tensor_to_float_vector(
    const NvDsInferTensorMeta * tensor_meta,
    const NvDsInferLayerInfo & layer) const
  {
    const auto shape = dims_to_shape(layer.inferDims);
    std::size_t volume = 1;
    for (const auto dim : shape) {
      volume *= static_cast<std::size_t>(std::max<int64_t>(1, dim));
    }

    const int output_index = static_cast<int>(&layer - tensor_meta->output_layers_info);
    const void * host_ptr = nullptr;
    if (tensor_meta->out_buf_ptrs_host != nullptr) {
      host_ptr = tensor_meta->out_buf_ptrs_host[output_index];
    }
    if (host_ptr == nullptr) {
      host_ptr = layer.buffer;
    }
    if (host_ptr == nullptr) {
      throw std::runtime_error("DeepStream tensor meta missing host output buffer");
    }

    std::vector<float> values(volume, 0.0F);
    if (layer.dataType == FLOAT) {
      const auto * floats = static_cast<const float *>(host_ptr);
      std::copy(floats, floats + volume, values.begin());
      return values;
    }
    if (layer.dataType == HALF) {
      const auto * halves = static_cast<const std::uint16_t *>(host_ptr);
      for (std::size_t i = 0; i < volume; ++i) {
        const std::uint16_t value = halves[i];
        const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000U)) << 16;
        const std::uint32_t exponent = (value >> 10) & 0x1FU;
        const std::uint32_t mantissa = value & 0x03FFU;
        std::uint32_t bits = 0;
        if (exponent == 0) {
          if (mantissa == 0) {
            bits = sign;
          } else {
            std::uint32_t mant = mantissa;
            std::int32_t exp = -14;
            while ((mant & 0x0400U) == 0U) {
              mant <<= 1U;
              --exp;
            }
            mant &= 0x03FFU;
            bits = sign | (static_cast<std::uint32_t>(exp + 127) << 23) | (mant << 13);
          }
        } else if (exponent == 0x1FU) {
          bits = sign | 0x7F800000U | (mantissa << 13);
        } else {
          bits = sign | ((exponent + 112U) << 23) | (mantissa << 13);
        }
        float result = 0.0F;
        std::memcpy(&result, &bits, sizeof(result));
        values[i] = result;
      }
      return values;
    }

    throw std::runtime_error("Unsupported DeepStream tensor output datatype");
  }

  std::optional<NvDsInferTensorMeta *> find_tensor_meta(NvDsFrameMeta * frame_meta) const
  {
    if (frame_meta == nullptr) {
      return std::nullopt;
    }
    for (NvDsMetaList * user_meta = frame_meta->frame_user_meta_list;
      user_meta != nullptr;
      user_meta = user_meta->next)
    {
      auto * meta = static_cast<NvDsUserMeta *>(user_meta->data);
      if (meta == nullptr || meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) {
        continue;
      }
      return static_cast<NvDsInferTensorMeta *>(meta->user_meta_data);
    }
    return std::nullopt;
  }

  void infer_loop()
  {
    while (running) {
      GstSample * sample = gst_app_sink_try_pull_sample(
        infer_sink,
        100 * GST_MSECOND);
      if (sample == nullptr) {
        continue;
      }

      try {
        process_infer_sample(sample);
      } catch (const std::exception & exc) {
        RCLCPP_ERROR(logger, "Failed processing DeepStream inference sample: %s", exc.what());
      }

      gst_sample_unref(sample);
    }
  }

  void color_loop()
  {
    while (running) {
      GstSample * sample = gst_app_sink_try_pull_sample(
        color_sink,
        100 * GST_MSECOND);
      if (sample == nullptr) {
        continue;
      }

      try {
        store_frame(sample_to_bgr_packet(sample));
      } catch (const std::exception & exc) {
        RCLCPP_ERROR(logger, "Failed processing DeepStream color sample: %s", exc.what());
      }

      gst_sample_unref(sample);
    }
  }

  void process_infer_sample(GstSample * sample)
  {
    GstBuffer * buffer = gst_sample_get_buffer(sample);
    if (buffer == nullptr) {
      return;
    }

    NvDsBatchMeta * batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (batch_meta == nullptr) {
      throw std::runtime_error("DeepStream buffer missing batch meta");
    }

    for (NvDsMetaList * frame_list = batch_meta->frame_meta_list;
      frame_list != nullptr;
      frame_list = frame_list->next)
    {
      auto * frame_meta = static_cast<NvDsFrameMeta *>(frame_list->data);
      if (frame_meta == nullptr) {
        continue;
      }

      const auto tensor_meta = find_tensor_meta(frame_meta);
      if (!tensor_meta.has_value() || *tensor_meta == nullptr) {
        throw std::runtime_error("DeepStream frame missing tensor output meta");
      }
      if ((*tensor_meta)->num_output_layers <= 0) {
        throw std::runtime_error("DeepStream tensor meta has no output layers");
      }

      const auto parse_start = std::chrono::steady_clock::now();
      const auto & output_layer = (*tensor_meta)->output_layers_info[0];
      const auto tensor_values = tensor_to_float_vector(*tensor_meta, output_layer);
      const auto output_shape = dims_to_shape(output_layer.inferDims);

      Result result{};
      if (config.enable_color_verification || config.debug_publish_image_raw) {
        const auto pts = frame_meta->buf_pts != GST_CLOCK_TIME_NONE ?
          frame_meta->buf_pts : GST_BUFFER_PTS(buffer);
        if (pts != GST_CLOCK_TIME_NONE) {
          result.source_packet = lookup_frame(static_cast<std::uint64_t>(pts));
        } else {
          RCLCPP_WARN(
            logger,
            "DeepStream inference sample missing valid PTS; color/debug frame lookup skipped (%s)",
            gst_clock_to_string(pts).c_str());
        }
      }

      const int src_w = result.source_packet.has_value() && !result.source_packet->empty() ?
        result.source_packet->width() :
        (frame_meta->source_frame_width > 0 ?
        static_cast<int>(frame_meta->source_frame_width) : config.source_width);
      const int src_h = result.source_packet.has_value() && !result.source_packet->empty() ?
        result.source_packet->height() :
        (frame_meta->source_frame_height > 0 ?
        static_cast<int>(frame_meta->source_frame_height) : config.source_height);
      result.batch.letterbox = make_letterbox_metadata(
        src_w, src_h, config.imgsz_width, config.imgsz_height);
      result.batch.detections = parse_yolo_output_tensor(
        tensor_values.data(),
        output_shape,
        config.imgsz_width,
        config.imgsz_height,
        std::max(1, num_classes),
        config.conf_threshold,
        task_obb,
        end2end,
        export_includes_nms,
        &logger);
      const auto parse_done = std::chrono::steady_clock::now();
      result.timings.parse_ms = std::chrono::duration<double, std::milli>(
        parse_done - parse_start).count();

      callback(std::move(result));
    }
  }
#endif
};

DeepStreamRuntime::DeepStreamRuntime(
  rclcpp::Logger logger,
  Config config,
  ResultCallback callback)
: impl_(std::make_unique<Impl>(std::move(logger), std::move(config), std::move(callback)))
{
  impl_->start();
}

DeepStreamRuntime::~DeepStreamRuntime() = default;

const std::string & DeepStreamRuntime::backend_name() const
{
  return impl_->backend_name;
}

double DeepStreamRuntime::source_actual_fps() const
{
#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
  return impl_->source_actual_fps;
#else
  return 0.0;
#endif
}

}  // namespace yolo_cpp
