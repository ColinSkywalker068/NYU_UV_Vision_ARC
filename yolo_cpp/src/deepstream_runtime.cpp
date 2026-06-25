/// @file deepstream_runtime.cpp
/// @brief Implements the optional asynchronous GStreamer/DeepStream inference runtime.

#include "yolo_cpp/deepstream_runtime.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <rclcpp/logging.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "deepstream_runtime_helpers.hpp"
#include "yolo_cpp/detection_postprocess.hpp"
#include "yolo_cpp/gst_cuda_bayer_resize.hpp"
#include "yolo_cpp/utils.hpp"

#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/gstchildproxy.h>
#include <gst/video/video.h>
#include <gstnvdsinfer.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#endif

#if defined(HAVE_LIBREALSENSE2)
#include <librealsense2/rs.hpp>
#endif

namespace yolo_cpp
{

namespace
{

using deepstream_helpers::make_video_caps_text;
using deepstream_helpers::normalize_format_token;
using deepstream_helpers::path_has_extension;
using deepstream_helpers::positive_dimension;
using deepstream_helpers::read_text_file;
using deepstream_helpers::replace_all;
using deepstream_helpers::resolve_pre_mux_resize;
using deepstream_helpers::resolve_source_format;
using deepstream_helpers::SourceFormatSpec;

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
void release_request_pad(GstElement* element, GstPad*& pad)
{
  if (element == nullptr || pad == nullptr) {
    return;
  }
  gst_element_release_request_pad(element, pad);
  gst_object_unref(pad);
  pad = nullptr;
}

bool set_child_proxy_bool_if_present(GstElement* element, const char* name, bool value)
{
  if (element == nullptr || name == nullptr) {
    return false;
  }

  if (GParamSpec* pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
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

  GObject* target = nullptr;
  GParamSpec* pspec = nullptr;
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

bool string_matches_enum_value(const GEnumValue* enum_value, const char* value)
{
  if (enum_value == nullptr || value == nullptr) {
    return false;
  }
  if (g_strcmp0(enum_value->value_name, value) == 0 || g_strcmp0(enum_value->value_nick, value) == 0
      || g_ascii_strcasecmp(enum_value->value_name, value) == 0
      || g_ascii_strcasecmp(enum_value->value_nick, value) == 0)
  {
    return true;
  }

  const char* suffix = std::strrchr(enum_value->value_name, '_');
  return suffix != nullptr && g_ascii_strcasecmp(suffix + 1, value) == 0;
}

bool set_enum_value_from_string(GValue* gvalue, const char* value)
{
  GEnumClass* enum_class = G_ENUM_CLASS(g_type_class_ref(G_VALUE_TYPE(gvalue)));
  if (enum_class == nullptr) {
    return false;
  }

  GEnumValue* enum_value = g_enum_get_value_by_name(enum_class, value);
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

bool set_string_value(GValue* gvalue, const char* value)
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

bool set_numeric_value(GValue* gvalue, double value)
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

bool set_child_proxy_string_if_present(GstElement* element, const char* name, const char* value)
{
  if (element == nullptr || name == nullptr || value == nullptr) {
    return false;
  }

  if (GParamSpec* pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
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

  GObject* target = nullptr;
  GParamSpec* pspec = nullptr;
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

bool set_child_proxy_double_if_present(GstElement* element, const char* name, double value)
{
  if (element == nullptr || name == nullptr) {
    return false;
  }

  if (GParamSpec* pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
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

  GObject* target = nullptr;
  GParamSpec* pspec = nullptr;
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

std::optional<double> get_numeric_value(const GValue* gvalue)
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

std::optional<std::string> get_string_value(const GValue* gvalue)
{
  if (G_VALUE_HOLDS_STRING(gvalue)) {
    const char* value = g_value_get_string(gvalue);
    if (value != nullptr && std::strlen(value) > 0U) {
      return std::string(value);
    }
    return std::nullopt;
  }
  if (G_VALUE_HOLDS_ENUM(gvalue)) {
    GEnumClass* enum_class = G_ENUM_CLASS(g_type_class_ref(G_VALUE_TYPE(gvalue)));
    if (enum_class != nullptr) {
      GEnumValue* enum_value = g_enum_get_value(enum_class, g_value_get_enum(gvalue));
      std::optional<std::string> result;
      if (enum_value != nullptr) {
        result = enum_value->value_nick != nullptr ? std::string(enum_value->value_nick)
                                                   : std::string(enum_value->value_name);
      }
      g_type_class_unref(enum_class);
      if (result.has_value() && !result->empty()) {
        return result;
      }
    }
  }

  gchar* serialized = gst_value_serialize(gvalue);
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

std::optional<double> get_child_proxy_numeric_if_present(GstElement* element, const char* name)
{
  if (element == nullptr || name == nullptr) {
    return std::nullopt;
  }

  if (GParamSpec* pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
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

  GObject* target = nullptr;
  GParamSpec* pspec = nullptr;
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

std::optional<std::string> get_child_proxy_string_if_present(GstElement* element, const char* name)
{
  if (element == nullptr || name == nullptr) {
    return std::nullopt;
  }

  if (GParamSpec* pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name)) {
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

  GObject* target = nullptr;
  GParamSpec* pspec = nullptr;
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

std::string caps_to_string(GstCaps* caps)
{
  if (caps == nullptr) {
    return "none";
  }
  gchar* text = gst_caps_to_string(caps);
  if (text == nullptr) {
    return "unknown";
  }
  std::string result(text);
  g_free(text);
  return result;
}

std::string current_caps_for_pad(GstElement* element, const char* pad_name)
{
  if (element == nullptr || pad_name == nullptr) {
    return "none";
  }
  GstPad* pad = gst_element_get_static_pad(element, pad_name);
  if (pad == nullptr) {
    return "none";
  }
  GstCaps* caps = gst_pad_get_current_caps(pad);
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

std::string collect_bus_diagnostics(GstBus* bus)
{
  if (bus == nullptr) {
    return "";
  }

  std::ostringstream diagnostics;
  for (int i = 0; i < 8; ++i) {
    GstMessage* message = gst_bus_timed_pop_filtered(
      bus, 100 * GST_MSECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING));
    if (message == nullptr) {
      break;
    }

    GstObject* source = GST_MESSAGE_SRC(message);
    const char* source_name = source != nullptr ? GST_OBJECT_NAME(source) : nullptr;
    GError* error = nullptr;
    gchar* debug = nullptr;
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
      : logger(std::move(logger_in)), config(std::move(config_in)), callback(std::move(callback_in))
  {
  }

  rclcpp::Logger logger;
  Config config;
  ResultCallback callback;
  std::string backend_name{"deepstream"};

#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
  GstElement* pipeline{nullptr};
  GstElement* source{nullptr};
  GstElement* source_caps{nullptr};
  GstElement* source_bayer_convert{nullptr};
  GstElement* source_bayer_caps{nullptr};
  GstElement* source_flip{nullptr};
  GstElement* tee{nullptr};
  GstElement* infer_queue{nullptr};
  GstElement* infer_convert{nullptr};
  GstElement* infer_cpu_caps{nullptr};
  GstElement* infer_nvconvert{nullptr};
  GstElement* infer_nvmm_caps{nullptr};
  GstElement* streammux{nullptr};
  GstElement* nvinfer{nullptr};
  GstElement* infer_post_queue{nullptr};
  GstElement* infer_sink_elem{nullptr};
  GstElement* color_queue{nullptr};
  GstElement* color_convert{nullptr};
  GstElement* color_caps{nullptr};
  GstElement* color_sink_elem{nullptr};
  GstBus* bus{nullptr};
  GstAppSink* infer_sink{nullptr};
  GstAppSink* color_sink{nullptr};
  GstAppSrc* app_source{nullptr};
  GstPad* tee_infer_pad{nullptr};
  GstPad* streammux_sink_pad{nullptr};
  GstPad* tee_color_pad{nullptr};
  std::filesystem::path runtime_config_path;
  std::thread source_thread;
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
  std::string actual_bayer_conversion_backend{"none"};
  bool cuda_bayer_resizes_source{false};
  GstClockTime next_source_pts{0};
  GstClockTime source_frame_duration{GST_CLOCK_TIME_NONE};
  std::uint64_t source_frame_index{0};
#if defined(HAVE_LIBREALSENSE2)
  rs2::pipeline realsense_pipeline;
  rs2::pipeline_profile realsense_profile;
  rs2_format realsense_stream_format{RS2_FORMAT_YUYV};
#endif
  enum class StageProbePoint : std::size_t
  {
    kSourceCaps = 0,
    kBayerConvert,
    kInferConvertSink,
    kInferConvertSrc,
    kNvConvertSrc,
    kInferNvmmSrc,
    kStreammuxSink,
    kStreammuxSrc,
    kNvInferSrc,
    kCount
  };

  struct StageProbeContext
  {
    DeepStreamRuntime::Impl* runtime{nullptr};
    StageProbePoint point{StageProbePoint::kSourceCaps};
  };

  struct StageProbeFrame
  {
    std::array<std::optional<std::chrono::steady_clock::time_point>,
      static_cast<std::size_t>(StageProbePoint::kCount)>
      times;
  };

  struct StageProfileMetric
  {
    double total_ms{0.0};
    std::size_t count{0};
  };

  struct StageProfileTotals
  {
    StageProfileMetric source_to_bayer;
    StageProfileMetric source_or_bayer_to_queue;
    StageProfileMetric videoconvert;
    StageProfileMetric nvvideoconvert;
    StageProfileMetric nvmm_to_streammux;
    StageProfileMetric nvinfer;
  };

  std::mutex stage_profile_mutex;
  std::map<std::uint64_t, StageProbeFrame> stage_profile_frames;
  StageProfileTotals stage_profile_totals;
  std::vector<std::unique_ptr<StageProbeContext>> stage_probe_contexts;
  std::size_t stage_profile_completed_frames{0};
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
      "DeepStream runtime unavailable: build missing GStreamer or DeepStream development "
      "headers/libs");
#else
    prepare_source_config();
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

    if (app_source != nullptr) {
      gst_app_src_end_of_stream(app_source);
    }
#if defined(HAVE_LIBREALSENSE2)
    try {
      realsense_pipeline.stop();
    } catch (...) {
    }
#endif

    if (source_thread.joinable()) {
      source_thread.join();
    }
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
    app_source = nullptr;

    if (!runtime_config_path.empty()) {
      std::error_code ec;
      std::filesystem::remove(runtime_config_path, ec);
    }
#endif
  }

#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
  bool wants_cuda_bayer_backend() const
  {
    const auto backend = normalize_format_token(config.bayer_conversion_backend);
    return backend == "cuda" || backend == "gpu" || backend == "cudabayerresize"
           || backend == "yolocudabayerresize";
  }

  bool wants_cpu_bayer_backend() const
  {
    const auto backend = normalize_format_token(config.bayer_conversion_backend);
    return backend == "cpu" || backend == "bayer2rgb";
  }

  bool is_realsense_source() const
  {
    return normalize_format_token(config.source_kind) == "realsense";
  }

  bool is_basler_source() const
  {
    const auto source_kind = normalize_format_token(config.source_kind);
    return source_kind.empty() || source_kind == "basler";
  }

  void prepare_source_config()
  {
    if (is_realsense_source()) {
      prepare_realsense_source_config();
      return;
    }
    if (!is_basler_source()) {
      throw std::runtime_error(
        "Unsupported DeepStream source_kind '" + config.source_kind + "'. Use basler or realsense.");
    }
  }

#if defined(HAVE_LIBREALSENSE2)
  static std::string realsense_format_label(rs2_format format)
  {
    switch (format) {
      case RS2_FORMAT_YUYV:
        return "yuy2";
      case RS2_FORMAT_BGR8:
        return "bgr8";
      case RS2_FORMAT_RGB8:
        return "rgb8";
      default:
        return "unknown";
    }
  }

  static int bytes_per_pixel_for_realsense_format(rs2_format format)
  {
    switch (format) {
      case RS2_FORMAT_YUYV:
        return 2;
      case RS2_FORMAT_BGR8:
      case RS2_FORMAT_RGB8:
        return 3;
      default:
        throw std::runtime_error("Unsupported RealSense color format for DeepStream appsrc");
    }
  }

  void prepare_realsense_source_config()
  {
    config.source_kind = "realsense";
    config.source_element = "appsrc";
    config.source_id = "realsense";
    config.source_use_full_sensor = true;
    config.flip_source = false;

    const int requested_width = std::max(1, config.source_width);
    const int requested_height = std::max(1, config.source_height);
    const int requested_fps = std::max(1, config.source_fps);
    struct PreferredFormat
    {
      rs2_format format;
      const char* label;
    };
    const std::array<PreferredFormat, 3> preferred_formats{{
      {RS2_FORMAT_YUYV, "yuy2"},
      {RS2_FORMAT_BGR8, "bgr8"},
      {RS2_FORMAT_RGB8, "rgb8"},
    }};

    rs2::context ctx;
    auto devices = ctx.query_devices();
    if (devices.size() == 0) {
      throw std::runtime_error("DeepStream RealSense source requested, but no RealSense camera was found");
    }

    int best_fps = -1;
    int best_rank = static_cast<int>(preferred_formats.size());
    rs2_format best_format = RS2_FORMAT_YUYV;

    for (auto&& device : devices) {
      for (auto&& sensor : device.query_sensors()) {
        if (!sensor.is<rs2::color_sensor>()) {
          continue;
        }
        for (auto&& profile : sensor.get_stream_profiles()) {
          auto video = profile.as<rs2::video_stream_profile>();
          if (video.stream_type() != RS2_STREAM_COLOR || video.width() != requested_width
              || video.height() != requested_height)
          {
            continue;
          }
          const int fps = video.fps();
          if (!config.source_use_max_native_fps && fps < requested_fps) {
            continue;
          }
          for (std::size_t rank = 0; rank < preferred_formats.size(); ++rank) {
            if (video.format() != preferred_formats[rank].format) {
              continue;
            }
            if (fps > best_fps || (fps == best_fps && static_cast<int>(rank) < best_rank)) {
              best_fps = fps;
              best_rank = static_cast<int>(rank);
              best_format = video.format();
            }
          }
        }
      }
    }

    if (best_fps <= 0) {
      throw std::runtime_error(
        "No compatible RealSense color profile found for " + std::to_string(requested_width) + "x"
        + std::to_string(requested_height) + " @ " + std::to_string(requested_fps)
        + " FPS using YUYV/BGR8/RGB8");
    }

    config.source_width = requested_width;
    config.source_height = requested_height;
    config.source_fps = best_fps;
    config.source_pixel_format = realsense_format_label(best_format);
    realsense_stream_format = best_format;
    source_actual_fps = static_cast<double>(best_fps);
    source_frame_duration = gst_util_uint64_scale_int(GST_SECOND, 1, best_fps);

    rs2::config rs_config;
    try {
      rs_config.disable_all_streams();
    } catch (...) {
    }
    rs_config.enable_stream(
      RS2_STREAM_COLOR, config.source_width, config.source_height, realsense_stream_format,
      config.source_fps);
    realsense_profile = realsense_pipeline.start(rs_config);
    configure_realsense_sensor();

    auto stream_profile =
      realsense_profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
    config.source_width = stream_profile.width();
    config.source_height = stream_profile.height();
    config.source_fps = stream_profile.fps();
    realsense_stream_format = stream_profile.format();
    config.source_pixel_format = realsense_format_label(realsense_stream_format);
    source_actual_fps = static_cast<double>(std::max(1, stream_profile.fps()));
    source_frame_duration =
      gst_util_uint64_scale_int(GST_SECOND, 1, std::max(1, stream_profile.fps()));

    RCLCPP_INFO(logger, "RealSense color-only DeepStream source selected: %dx%d %s @ %d FPS",
      config.source_width, config.source_height, config.source_pixel_format.c_str(),
      config.source_fps);
  }

  void configure_realsense_sensor()
  {
    auto device = realsense_profile.get_device();
    for (auto&& sensor : device.query_sensors()) {
      if (!sensor.is<rs2::color_sensor>()) {
        continue;
      }
      if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
        sensor.set_option(
          RS2_OPTION_ENABLE_AUTO_EXPOSURE, config.source_auto_exposure ? 1.0F : 0.0F);
      }
      if (!config.source_auto_exposure) {
        if (sensor.supports(RS2_OPTION_EXPOSURE)) {
          sensor.set_option(RS2_OPTION_EXPOSURE, static_cast<float>(config.source_exposure_us));
        }
        if (sensor.supports(RS2_OPTION_GAIN)) {
          sensor.set_option(RS2_OPTION_GAIN, static_cast<float>(config.source_gain_db));
        }
      }
      if (sensor.supports(RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE)) {
        sensor.set_option(
          RS2_OPTION_ENABLE_AUTO_WHITE_BALANCE, config.source_auto_white_balance ? 1.0F : 0.0F);
      }
      if (!config.source_auto_white_balance && sensor.supports(RS2_OPTION_WHITE_BALANCE)) {
        sensor.set_option(
          RS2_OPTION_WHITE_BALANCE, static_cast<float>(config.source_white_balance));
      }
    }
  }

  GstBuffer* make_realsense_appsrc_buffer(const rs2::video_frame& frame)
  {
    const int width = frame.get_width();
    const int height = frame.get_height();
    const int bytes_per_pixel = bytes_per_pixel_for_realsense_format(realsense_stream_format);
    const int dst_stride = width * bytes_per_pixel;
    const int src_stride = frame.get_stride_in_bytes();
    const auto* src = static_cast<const std::uint8_t*>(frame.get_data());
    if (src == nullptr || width <= 0 || height <= 0 || src_stride <= 0) {
      return nullptr;
    }

    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, static_cast<gsize>(dst_stride * height), nullptr);
    if (buffer == nullptr) {
      throw std::runtime_error("Failed to allocate RealSense appsrc GstBuffer");
    }

    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, GST_MAP_WRITE)) {
      gst_buffer_unref(buffer);
      throw std::runtime_error("Failed to map RealSense appsrc GstBuffer");
    }
    for (int row = 0; row < height; ++row) {
      std::memcpy(map_info.data + row * dst_stride, src + row * src_stride,
        static_cast<std::size_t>(dst_stride));
    }
    gst_buffer_unmap(buffer, &map_info);

    const GstClockTime duration =
      source_frame_duration != GST_CLOCK_TIME_NONE
        ? source_frame_duration
        : gst_util_uint64_scale_int(GST_SECOND, 1, std::max(1, config.source_fps));
    GST_BUFFER_PTS(buffer) = next_source_pts;
    GST_BUFFER_DTS(buffer) = next_source_pts;
    GST_BUFFER_DURATION(buffer) = duration;
    GST_BUFFER_OFFSET(buffer) = source_frame_index++;
    next_source_pts += duration;
    return buffer;
  }

  void realsense_source_loop()
  {
    while (running) {
      try {
        rs2::frameset frames;
        if (config.source_use_poll) {
          if (!realsense_pipeline.poll_for_frames(&frames)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
          }
        } else {
          frames = realsense_pipeline.wait_for_frames(1000);
        }

        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) {
          continue;
        }
        GstBuffer* buffer = make_realsense_appsrc_buffer(color_frame);
        if (buffer == nullptr) {
          continue;
        }
        const GstFlowReturn flow = gst_app_src_push_buffer(app_source, buffer);
        if (flow != GST_FLOW_OK) {
          if (running) {
            RCLCPP_WARN(logger, "RealSense appsrc push stopped with flow return %d",
              static_cast<int>(flow));
          }
          break;
        }
      } catch (const rs2::error& exc) {
        if (running) {
          RCLCPP_ERROR(logger, "RealSense capture error: %s", exc.what());
        }
      } catch (const std::exception& exc) {
        if (running) {
          RCLCPP_ERROR(logger, "RealSense appsrc error: %s", exc.what());
        }
      }
    }

    if (app_source != nullptr) {
      gst_app_src_end_of_stream(app_source);
    }
  }
#else
  void prepare_realsense_source_config()
  {
    throw std::runtime_error(
      "DeepStream RealSense source requested, but this yolo_cpp build did not find "
      "librealsense2 headers/libs.");
  }
#endif

  /// Verify GStreamer/DeepStream elements and read model metadata needed for parsing.
  void validate_runtime()
  {
    GError* error = nullptr;
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
      RCLCPP_WARN(logger,
        "Model metadata did not expose class names from %s; assuming 2 classes for OBB armor model",
        config.model_path.c_str());
    }

    const auto source_format = resolve_source_format(config.source_pixel_format);
    const bool need_full_color_branch =
      config.enable_color_verification || config.debug_publish_image_raw;
    const bool can_try_cuda_bayer = source_format.needs_bayer_convert && wants_cuda_bayer_backend()
                                    && config.pre_mux_resize && !need_full_color_branch
                                    && cuda_bayer_resize_available()
                                    && register_cuda_bayer_resize_element();
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
    if (source_format.needs_bayer_convert && !can_try_cuda_bayer) {
      required_elements.push_back("bayer2rgb");
    }
    for (const auto& element_name : required_elements) {
      GstElementFactory* factory = gst_element_factory_find(element_name.c_str());
      if (factory == nullptr) {
        throw std::runtime_error("Required GStreamer element not found: " + element_name
                                 + ". Check DeepStream, GStreamer, and camera source plugin "
                                   "installation.");
      }
      gst_object_unref(factory);
    }
  }

  /// Render the nvinfer config template into a temporary runtime config file.
  void materialize_config()
  {
    std::string config_text = read_text_file(config.config_template_path);
    const auto model_source_config = path_has_extension(config.model_path, ".onnx")
                                       ? ("onnx-file=" + config.model_path
                                           + "\nmodel-engine-file=" + config.model_path + ".engine")
                                       : ("model-engine-file=" + config.model_path);
    replace_all(config_text, "__MODEL_SOURCE_CONFIG__", model_source_config);
    replace_all(config_text, "__MODEL_ENGINE_FILE__", config.model_path);
    replace_all(config_text, "__INPUT_WIDTH__", std::to_string(config.imgsz_width));
    replace_all(config_text, "__INPUT_HEIGHT__", std::to_string(config.imgsz_height));
    replace_all(config_text, "__NUM_CLASSES__", std::to_string(std::max(1, num_classes)));
    replace_all(config_text, "__CONF_THRESHOLD__", std::to_string(config.conf_threshold));
    replace_all(config_text, "__NETWORK_MODE__", config.half ? "2" : "0");
    replace_all(config_text, "__OUTPUT_BLOB_NAMES__", "output0");

    runtime_config_path =
      std::filesystem::temp_directory_path() / "yolo_cpp_deepstream_runtime_config.txt";
    std::ofstream stream(runtime_config_path);
    if (!stream) {
      throw std::runtime_error(
        "Failed to create runtime DeepStream config: " + runtime_config_path.string());
    }
    stream << config_text;
  }

  /// Create a required GStreamer element or throw with the missing factory name.
  GstElement* make_element(const char* factory, const char* name)
  {
    GstElement* element = gst_element_factory_make(factory, name);
    if (element == nullptr) {
      throw std::runtime_error("Failed to create GStreamer element '" + std::string(factory) + "'");
    }
    return element;
  }

  GstElement* make_bayer_converter(const SourceFormatSpec& source_format,
    const std::optional<std::pair<int, int>>& pre_mux_resize, bool need_color_branch)
  {
    actual_bayer_conversion_backend = "none";
    cuda_bayer_resizes_source = false;
    if (!source_format.needs_bayer_convert) {
      return nullptr;
    }

    const auto backend = normalize_format_token(config.bayer_conversion_backend);
    if (wants_cuda_bayer_backend()) {
      if (need_color_branch) {
        RCLCPP_WARN(logger,
          "CUDA Bayer resize backend requested, but full-resolution color/debug output is enabled; "
          "falling back to bayer2rgb");
      } else if (!pre_mux_resize.has_value()) {
        RCLCPP_WARN(logger,
          "CUDA Bayer resize backend requested without pre_mux_resize; falling back to bayer2rgb");
      } else if (!cuda_bayer_resize_available()) {
        RCLCPP_WARN(logger,
          "CUDA Bayer resize backend was not compiled into this build; falling back to bayer2rgb");
      } else if (!register_cuda_bayer_resize_element()) {
        RCLCPP_WARN(logger,
          "Failed to register CUDA Bayer resize GStreamer element; falling back to bayer2rgb");
      } else {
        GstElement* element = make_element("yolocudabayerresize", "source_bayer_convert");
        g_object_set(G_OBJECT(element), "output-width", pre_mux_resize->first, "output-height",
          pre_mux_resize->second, "flip-method", config.flip_source ? 2 : 0, nullptr);
        actual_bayer_conversion_backend = "cuda";
        cuda_bayer_resizes_source = true;
        RCLCPP_INFO(logger, "Using CUDA Bayer resize backend: %dx%d Bayer -> %dx%d RGBA%s",
          config.source_width, config.source_height, pre_mux_resize->first, pre_mux_resize->second,
          config.flip_source ? " with 180-degree flip" : "");
        return element;
      }
    } else if (!wants_cpu_bayer_backend()) {
      throw std::runtime_error("Unsupported Bayer conversion backend '"
                               + config.bayer_conversion_backend
                               + "'. Use cuda/gpu or cpu/bayer2rgb.");
    }

    if (backend == "cpu" || backend == "bayer2rgb" || wants_cuda_bayer_backend()) {
      actual_bayer_conversion_backend = "cpu";
      return make_element("bayer2rgb", "source_bayer_convert");
    }

    actual_bayer_conversion_backend = "cpu";
    return make_element("bayer2rgb", "source_bayer_convert");
  }

  bool set_camera_enum_off(const char* name)
  {
    return set_child_proxy_string_if_present(source, name, "Off")
           || set_child_proxy_double_if_present(source, name, 0.0);
  }

  void configure_low_latency_queue(GstElement* queue)
  {
    if (queue == nullptr) {
      return;
    }
    g_object_set(G_OBJECT(queue), "max-size-buffers", 1, "max-size-bytes", 0, "max-size-time", 0,
      "leaky", 2, nullptr);
  }

  void configure_basler_source(const SourceFormatSpec& source_format)
  {
    RCLCPP_INFO(logger, "Configuring Basler source format/size/framerate through caps: %s",
      source_format.label.c_str());
    set_child_proxy_string_if_present(source, "cam::ExposureMode", "Timed");
    if (!set_camera_enum_off("cam::ExposureAuto")) {
      RCLCPP_WARN(logger, "Could not set Basler cam::ExposureAuto=Off through pylonsrc");
    }
    if (!set_child_proxy_double_if_present(source, "cam::ExposureTime", config.source_exposure_us))
    {
      RCLCPP_WARN(logger, "Could not set Basler cam::ExposureTime=%.2f through pylonsrc",
        config.source_exposure_us);
    }

    if (!set_camera_enum_off("cam::GainAuto")) {
      RCLCPP_WARN(logger, "Could not set Basler cam::GainAuto=Off through pylonsrc");
    }
    set_child_proxy_string_if_present(source, "cam::GainSelector", "All");
    if (!set_child_proxy_double_if_present(source, "cam::Gain", config.source_gain_db)) {
      RCLCPP_WARN(
        logger, "Could not set Basler cam::Gain=%.2f through pylonsrc", config.source_gain_db);
    }
  }

  std::optional<double> read_source_actual_fps() const
  {
    if (auto fps =
          get_child_proxy_numeric_if_present(source, "cam::BslResultingAcquisitionFrameRate");
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
    if (is_realsense_source()) {
      source_actual_fps =
        source_actual_fps > 0.0 ? source_actual_fps : static_cast<double>(std::max(1, config.source_fps));
      RCLCPP_INFO(logger,
        "DeepStream RealSense appsrc source: %dx%d pixel_format=%s @ %.2f FPS "
        "(pre_mux_resize=%s, flip_source=false)",
        config.source_width, config.source_height, config.source_pixel_format.c_str(),
        source_actual_fps, config.pre_mux_resize ? "true" : "false");
      RCLCPP_INFO(logger,
        "DeepStream RealSense caps requested: source='%s' infer_cpu='%s' infer_nvmm='%s' color='%s'",
        requested_source_caps.c_str(), requested_infer_cpu_caps.c_str(),
        requested_infer_nvmm_caps.c_str(),
        requested_color_caps.empty() ? "disabled" : requested_color_caps.c_str());
      RCLCPP_INFO(logger,
        "DeepStream RealSense caps negotiated: source='%s' infer_nvmm='%s' color='%s'",
        current_caps_for_pad(source_caps, "src").c_str(),
        current_caps_for_pad(infer_nvmm_caps, "src").c_str(),
        color_caps != nullptr ? current_caps_for_pad(color_caps, "src").c_str() : "disabled");
      return;
    }

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

    RCLCPP_INFO(logger,
      "DeepStream Basler source %s: %dx%d pixel_format=%s @ %.2f FPS "
      "(full_sensor=%s, pre_mux_resize=%s, bayer_backend=%s, fps_cap=%s)",
      fps.has_value() ? "reported" : "fallback", config.source_width, config.source_height,
      pixel_format.value_or("unknown").c_str(), source_actual_fps,
      config.source_use_full_sensor ? "true" : "false", config.pre_mux_resize ? "true" : "false",
      actual_bayer_conversion_backend.c_str(), fps_cap_status.c_str());
    RCLCPP_INFO(logger,
      "DeepStream Basler caps requested: source='%s' infer_cpu='%s' infer_nvmm='%s' color='%s'",
      requested_source_caps.c_str(), requested_infer_cpu_caps.c_str(),
      requested_infer_nvmm_caps.c_str(),
      requested_color_caps.empty() ? "disabled" : requested_color_caps.c_str());
    RCLCPP_INFO(logger, "DeepStream Basler caps negotiated: source='%s' infer_nvmm='%s' color='%s'",
      current_caps_for_pad(source_caps, "src").c_str(),
      current_caps_for_pad(infer_nvmm_caps, "src").c_str(),
      color_caps != nullptr ? current_caps_for_pad(color_caps, "src").c_str() : "disabled");
  }

  static GstPadProbeReturn stage_profile_pad_probe(
    GstPad*, GstPadProbeInfo* info, gpointer user_data)
  {
    auto* context = static_cast<StageProbeContext*>(user_data);
    if (context == nullptr || context->runtime == nullptr || info == nullptr) {
      return GST_PAD_PROBE_OK;
    }
    if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) == 0) {
      return GST_PAD_PROBE_OK;
    }

    GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    if (buffer == nullptr) {
      return GST_PAD_PROBE_OK;
    }
    context->runtime->record_stage_profile_point(buffer, context->point);
    return GST_PAD_PROBE_OK;
  }

  static std::uint64_t stage_profile_key(GstBuffer* buffer)
  {
    const GstClockTime pts = GST_BUFFER_PTS(buffer);
    if (pts != GST_CLOCK_TIME_NONE) {
      return static_cast<std::uint64_t>(pts);
    }
    const GstClockTime dts = GST_BUFFER_DTS(buffer);
    if (dts != GST_CLOCK_TIME_NONE) {
      return static_cast<std::uint64_t>(dts);
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(buffer));
  }

  static void add_stage_delta(const StageProbeFrame& frame, StageProbePoint from,
    StageProbePoint to, StageProfileMetric& metric)
  {
    const auto from_index = static_cast<std::size_t>(from);
    const auto to_index = static_cast<std::size_t>(to);
    if (!frame.times[from_index].has_value() || !frame.times[to_index].has_value()) {
      return;
    }
    metric.total_ms +=
      std::chrono::duration<double, std::milli>(*frame.times[to_index] - *frame.times[from_index])
        .count();
    ++metric.count;
  }

  static std::string average_text(const StageProfileMetric& metric)
  {
    if (metric.count == 0U) {
      return "n/a";
    }
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3)
           << metric.total_ms / static_cast<double>(metric.count);
    return stream.str();
  }

  void maybe_log_stage_profile()
  {
    const auto interval = static_cast<std::size_t>(std::max(1, config.profile_stage_interval));
    if (stage_profile_completed_frames == 0U || (stage_profile_completed_frames % interval) != 0U) {
      return;
    }

    RCLCPP_INFO(logger,
      "DeepStream stage profile frames=%zu avg_ms source_to_bayer=%s "
      "bayer_or_source_to_queue=%s videoconvert=%s nvvideoconvert=%s "
      "nvmm_to_streammux=%s nvinfer=%s",
      stage_profile_completed_frames, average_text(stage_profile_totals.source_to_bayer).c_str(),
      average_text(stage_profile_totals.source_or_bayer_to_queue).c_str(),
      average_text(stage_profile_totals.videoconvert).c_str(),
      average_text(stage_profile_totals.nvvideoconvert).c_str(),
      average_text(stage_profile_totals.nvmm_to_streammux).c_str(),
      average_text(stage_profile_totals.nvinfer).c_str());
  }

  void record_stage_profile_point(GstBuffer* buffer, StageProbePoint point)
  {
    if (!config.profile_stages) {
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto key = stage_profile_key(buffer);
    std::lock_guard<std::mutex> lock(stage_profile_mutex);
    auto& frame = stage_profile_frames[key];
    frame.times[static_cast<std::size_t>(point)] = now;

    switch (point) {
      case StageProbePoint::kBayerConvert:
        add_stage_delta(frame, StageProbePoint::kSourceCaps, StageProbePoint::kBayerConvert,
          stage_profile_totals.source_to_bayer);
        break;
      case StageProbePoint::kInferConvertSink:
        add_stage_delta(frame,
          frame.times[static_cast<std::size_t>(StageProbePoint::kBayerConvert)].has_value()
            ? StageProbePoint::kBayerConvert
            : StageProbePoint::kSourceCaps,
          StageProbePoint::kInferConvertSink, stage_profile_totals.source_or_bayer_to_queue);
        break;
      case StageProbePoint::kInferConvertSrc:
        add_stage_delta(frame, StageProbePoint::kInferConvertSink,
          StageProbePoint::kInferConvertSrc, stage_profile_totals.videoconvert);
        break;
      case StageProbePoint::kNvConvertSrc:
        add_stage_delta(frame, StageProbePoint::kInferConvertSrc, StageProbePoint::kNvConvertSrc,
          stage_profile_totals.nvvideoconvert);
        break;
      case StageProbePoint::kStreammuxSrc:
        break;
      case StageProbePoint::kStreammuxSink:
        add_stage_delta(frame, StageProbePoint::kInferNvmmSrc, StageProbePoint::kStreammuxSink,
          stage_profile_totals.nvmm_to_streammux);
        break;
      case StageProbePoint::kNvInferSrc:
        add_stage_delta(frame, StageProbePoint::kStreammuxSrc, StageProbePoint::kNvInferSrc,
          stage_profile_totals.nvinfer);
        stage_profile_frames.erase(key);
        break;
      case StageProbePoint::kInferNvmmSrc:
        ++stage_profile_completed_frames;
        maybe_log_stage_profile();
        break;
      case StageProbePoint::kSourceCaps:
      case StageProbePoint::kCount:
        break;
    }

    while (stage_profile_frames.size() > 2048U) {
      stage_profile_frames.erase(stage_profile_frames.begin());
    }
  }

  void add_stage_profile_probe(
    GstElement* element, const char* pad_name, StageProbePoint point, const char* label)
  {
    if (!config.profile_stages || element == nullptr || pad_name == nullptr) {
      return;
    }
    GstPad* pad = gst_element_get_static_pad(element, pad_name);
    if (pad == nullptr) {
      RCLCPP_WARN(logger, "Could not attach DeepStream stage profile probe: %s", label);
      return;
    }
    add_stage_profile_probe_to_pad(pad, point, label);
    gst_object_unref(pad);
  }

  void add_stage_profile_probe_to_pad(GstPad* pad, StageProbePoint point, const char* label)
  {
    if (!config.profile_stages || pad == nullptr) {
      return;
    }
    auto context = std::make_unique<StageProbeContext>();
    context->runtime = this;
    context->point = point;
    gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER,
      &DeepStreamRuntime::Impl::stage_profile_pad_probe, context.get(), nullptr);
    stage_probe_contexts.push_back(std::move(context));
    (void)label;
  }

  void attach_stage_profile_probes()
  {
    if (!config.profile_stages) {
      return;
    }
    add_stage_profile_probe(source_caps, "src", StageProbePoint::kSourceCaps, "source_caps:src");
    if (source_bayer_convert != nullptr) {
      add_stage_profile_probe(
        source_bayer_convert, "src", StageProbePoint::kBayerConvert, "source_bayer_convert:src");
    }
    add_stage_profile_probe(
      infer_convert, "sink", StageProbePoint::kInferConvertSink, "videoconvert:sink");
    add_stage_profile_probe(
      infer_convert, "src", StageProbePoint::kInferConvertSrc, "videoconvert:src");
    add_stage_profile_probe(
      infer_nvconvert, "src", StageProbePoint::kNvConvertSrc, "nvvideoconvert:src");
    add_stage_profile_probe(
      infer_nvmm_caps, "src", StageProbePoint::kInferNvmmSrc, "infer_nvmm_caps:src");
    add_stage_profile_probe_to_pad(
      streammux_sink_pad, StageProbePoint::kStreammuxSink, "nvstreammux:sink_0");
    add_stage_profile_probe(streammux, "src", StageProbePoint::kStreammuxSrc, "nvstreammux:src");
    add_stage_profile_probe(nvinfer, "src", StageProbePoint::kNvInferSrc, "nvinfer:src");
    RCLCPP_INFO(logger, "DeepStream stage profiling enabled; logging every %d frames",
      std::max(1, config.profile_stage_interval));
  }

  void build_pipeline()
  {
    pipeline = gst_pipeline_new("yolo-deepstream-pipeline");
    if (pipeline == nullptr) {
      throw std::runtime_error("Failed to create DeepStream pipeline");
    }

    const bool source_is_realsense = is_realsense_source();
    const auto source_format = resolve_source_format(config.source_pixel_format);
    const std::optional<int> source_width = positive_dimension(config.source_width);
    const std::optional<int> source_height = positive_dimension(config.source_height);
    const auto pre_mux_resize = resolve_pre_mux_resize(config);
    const bool need_full_color_branch =
      config.enable_color_verification || config.debug_publish_image_raw;
    const bool need_preview_branch = config.preview_frame && !need_full_color_branch;
    const bool need_color_branch = need_full_color_branch || need_preview_branch;
    source = make_element(config.source_element.c_str(), "source");
    if (source_is_realsense) {
      app_source = GST_APP_SRC(source);
      gst_app_src_set_stream_type(app_source, GST_APP_STREAM_TYPE_STREAM);
      g_object_set(G_OBJECT(source), "is-live", TRUE, "format", GST_FORMAT_TIME, "block", FALSE,
        "do-timestamp", FALSE, nullptr);
    }
    source_caps = make_element("capsfilter", "source_caps");
    source_bayer_convert =
      make_bayer_converter(source_format, pre_mux_resize, need_full_color_branch);
    if (cuda_bayer_resizes_source) {
      source_bayer_caps = make_element("capsfilter", "source_bayer_caps");
    }
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

    const bool nvconvert_flip_supported =
      g_object_class_find_property(G_OBJECT_GET_CLASS(infer_nvconvert), "flip-method") != nullptr;
    const bool use_infer_nvconvert_flip = config.flip_source && !cuda_bayer_resizes_source
                                          && !need_full_color_branch && nvconvert_flip_supported;
    const bool use_source_flip =
      config.flip_source && !cuda_bayer_resizes_source && !use_infer_nvconvert_flip;
    if (use_source_flip) {
      source_flip = make_element("videoflip", "source_flip");
    }
    if (need_color_branch) {
      color_queue = make_element("queue", "color_queue");
      color_convert = make_element("videoconvert", "color_convert");
      color_caps = make_element("capsfilter", "color_caps");
      color_sink_elem = make_element("appsink", "color_sink");
      color_sink = GST_APP_SINK(color_sink_elem);
    }

    gst_bin_add_many(GST_BIN(pipeline), source, source_caps, tee, infer_queue, infer_convert,
      infer_cpu_caps, infer_nvconvert, infer_nvmm_caps, streammux, nvinfer, infer_post_queue,
      infer_sink_elem, nullptr);
    if (source_bayer_convert != nullptr) {
      gst_bin_add(GST_BIN(pipeline), source_bayer_convert);
    }
    if (source_bayer_caps != nullptr) {
      gst_bin_add(GST_BIN(pipeline), source_bayer_caps);
    }
    if (source_flip != nullptr) {
      gst_bin_add(GST_BIN(pipeline), source_flip);
    }
    if (need_color_branch) {
      gst_bin_add_many(
        GST_BIN(pipeline), color_queue, color_convert, color_caps, color_sink_elem, nullptr);
    }

    const std::optional<int> processing_width =
      config.source_use_full_sensor ? std::nullopt : source_width;
    const std::optional<int> processing_height =
      config.source_use_full_sensor ? std::nullopt : source_height;

    if (!source_is_realsense
        && g_object_class_find_property(G_OBJECT_GET_CLASS(source), "do-timestamp") != nullptr)
    {
      g_object_set(G_OBJECT(source), "do-timestamp", TRUE, nullptr);
    }
    if (!source_is_realsense) {
      configure_basler_source(source_format);
    }
    configure_low_latency_queue(infer_queue);
    configure_low_latency_queue(infer_post_queue);

    bool uncapped_fps_applied = false;
    if (!source_is_realsense) {
      if (config.source_fps_uncapped
          && set_child_proxy_bool_if_present(source, "cam::AcquisitionFrameRateEnable", false))
      {
        uncapped_fps_applied = true;
        fps_cap_status = "disabled camera AcquisitionFrameRate limit";
      } else if (config.source_fps_uncapped) {
        RCLCPP_INFO(logger,
          "Basler AcquisitionFrameRateEnable is not exposed by pylonsrc; requesting %d "
          "FPS through caps",
          config.source_fps);
        fps_cap_status = "uncapped requested, falling back to caps framerate";
      }
    }

    std::optional<int> capped_fps;
    if (source_is_realsense) {
      capped_fps = std::max(1, config.source_fps);
      fps_cap_status = "RealSense selected color FPS";
    } else if (!config.source_fps_uncapped) {
      set_child_proxy_bool_if_present(source, "cam::AcquisitionFrameRateEnable", true);
      if (!set_child_proxy_double_if_present(
            source, "cam::AcquisitionFrameRate", static_cast<double>(config.source_fps)))
      {
        RCLCPP_WARN(logger, "Could not set Basler cam::AcquisitionFrameRate=%d through pylonsrc",
          config.source_fps);
      }
      capped_fps = config.source_fps;
      fps_cap_status = "camera AcquisitionFrameRate cap requested";
    } else if (!uncapped_fps_applied) {
      capped_fps = config.source_fps;
      fps_cap_status = "caps framerate fallback";
    }
    requested_source_caps = make_video_caps_text(
      source_format.media_type, source_format.caps_format, source_width, source_height, capped_fps);
    GstCaps* source_caps_value = gst_caps_from_string(requested_source_caps.c_str());
    if (source_caps_value == nullptr) {
      throw std::runtime_error("Failed to create DeepStream source caps: " + requested_source_caps);
    }
    g_object_set(G_OBJECT(source_caps), "caps", source_caps_value, nullptr);
    if (source_is_realsense) {
      gst_app_src_set_caps(app_source, source_caps_value);
    }
    gst_caps_unref(source_caps_value);
    if (source_flip != nullptr) {
      g_object_set(G_OBJECT(source_flip), "method", 2, nullptr);
    }
    if (use_infer_nvconvert_flip) {
      g_object_set(G_OBJECT(infer_nvconvert), "flip-method", 2, nullptr);
      RCLCPP_INFO(logger,
        "Moving Basler 180-degree flip to nvvideoconvert after inference-branch conversion");
    } else if (config.flip_source && !cuda_bayer_resizes_source && !need_full_color_branch
               && !nvconvert_flip_supported)
    {
      RCLCPP_WARN(
        logger, "nvvideoconvert does not expose flip-method; keeping full-resolution videoflip");
    }

    const std::optional<int> infer_width =
      pre_mux_resize.has_value()
        ? std::optional<int>(pre_mux_resize->first)
        : (config.source_use_full_sensor ? std::nullopt : std::optional<int>(config.imgsz_width));
    const std::optional<int> infer_height =
      pre_mux_resize.has_value()
        ? std::optional<int>(pre_mux_resize->second)
        : (config.source_use_full_sensor ? std::nullopt : std::optional<int>(config.imgsz_height));
    const std::optional<int> infer_cpu_width =
      cuda_bayer_resizes_source ? infer_width : processing_width;
    const std::optional<int> infer_cpu_height =
      cuda_bayer_resizes_source ? infer_height : processing_height;
    requested_infer_cpu_caps =
      make_video_caps_text("video/x-raw", "RGBA", infer_cpu_width, infer_cpu_height, capped_fps);
    GstCaps* infer_cpu_caps_value = gst_caps_from_string(requested_infer_cpu_caps.c_str());
    if (infer_cpu_caps_value == nullptr) {
      throw std::runtime_error(
        "Failed to create DeepStream CPU conversion caps: " + requested_infer_cpu_caps);
    }
    g_object_set(G_OBJECT(infer_cpu_caps), "caps", infer_cpu_caps_value, nullptr);
    gst_caps_unref(infer_cpu_caps_value);
    if (source_bayer_caps != nullptr) {
      GstCaps* source_bayer_caps_value = gst_caps_from_string(requested_infer_cpu_caps.c_str());
      if (source_bayer_caps_value == nullptr) {
        throw std::runtime_error(
          "Failed to create CUDA Bayer output caps: " + requested_infer_cpu_caps);
      }
      g_object_set(G_OBJECT(source_bayer_caps), "caps", source_bayer_caps_value, nullptr);
      gst_caps_unref(source_bayer_caps_value);
    }

    requested_infer_nvmm_caps = make_video_caps_text(
      "video/x-raw(memory:NVMM)", "NV12", infer_width, infer_height, capped_fps);
    GstCaps* infer_caps_value = gst_caps_from_string(requested_infer_nvmm_caps.c_str());
    if (infer_caps_value == nullptr) {
      throw std::runtime_error(
        "Failed to create DeepStream NVMM caps: " + requested_infer_nvmm_caps);
    }
    g_object_set(G_OBJECT(infer_nvmm_caps), "caps", infer_caps_value, nullptr);
    gst_caps_unref(infer_caps_value);

    g_object_set(G_OBJECT(streammux), "batch-size", 1, "width", config.imgsz_width, "height",
      config.imgsz_height, "live-source", TRUE, "enable-padding", TRUE, nullptr);
    g_object_set(G_OBJECT(nvinfer), "config-file-path", runtime_config_path.c_str(), nullptr);
    g_object_set(G_OBJECT(infer_sink_elem), "emit-signals", FALSE, "sync", FALSE, "max-buffers", 1,
      "drop", TRUE, nullptr);

    if (need_color_branch) {
      configure_low_latency_queue(color_queue);
      const std::optional<int> color_width =
        (need_preview_branch && cuda_bayer_resizes_source) ? infer_width : processing_width;
      const std::optional<int> color_height =
        (need_preview_branch && cuda_bayer_resizes_source) ? infer_height : processing_height;
      requested_color_caps =
        make_video_caps_text("video/x-raw", "BGR", color_width, color_height, capped_fps);
      GstCaps* color_caps_value = gst_caps_from_string(requested_color_caps.c_str());
      if (color_caps_value == nullptr) {
        throw std::runtime_error("Failed to create DeepStream color caps: " + requested_color_caps);
      }
      g_object_set(G_OBJECT(color_caps), "caps", color_caps_value, nullptr);
      gst_caps_unref(color_caps_value);
      g_object_set(G_OBJECT(color_sink_elem), "emit-signals", FALSE, "sync", FALSE, "max-buffers",
        4, "drop", TRUE, nullptr);
    }

    gboolean source_linked = FALSE;
    if (source_bayer_convert != nullptr && source_bayer_caps != nullptr && source_flip != nullptr) {
      source_linked = gst_element_link_many(
        source, source_caps, source_bayer_convert, source_bayer_caps, source_flip, tee, nullptr);
    } else if (source_bayer_convert != nullptr && source_bayer_caps != nullptr) {
      source_linked = gst_element_link_many(
        source, source_caps, source_bayer_convert, source_bayer_caps, tee, nullptr);
    } else if (source_bayer_convert != nullptr && source_flip != nullptr) {
      source_linked =
        gst_element_link_many(source, source_caps, source_bayer_convert, source_flip, tee, nullptr);
    } else if (source_bayer_convert != nullptr) {
      source_linked =
        gst_element_link_many(source, source_caps, source_bayer_convert, tee, nullptr);
    } else if (source_flip != nullptr) {
      source_linked = gst_element_link_many(source, source_caps, source_flip, tee, nullptr);
    } else {
      source_linked = gst_element_link_many(source, source_caps, tee, nullptr);
    }
    if (!source_linked) {
      throw std::runtime_error(source_bayer_convert != nullptr
                                 ? "Failed to link DeepStream Bayer source path"
                                 : "Failed to link DeepStream source path");
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
    GstPad* infer_queue_sink_pad = gst_element_get_static_pad(infer_queue, "sink");
    if (tee_infer_pad == nullptr || infer_queue_sink_pad == nullptr
        || gst_pad_link(tee_infer_pad, infer_queue_sink_pad) != GST_PAD_LINK_OK)
    {
      release_request_pad(tee, tee_infer_pad);
      if (infer_queue_sink_pad != nullptr) {
        gst_object_unref(infer_queue_sink_pad);
      }
      throw std::runtime_error("Failed to link tee to inference queue");
    }
    gst_object_unref(infer_queue_sink_pad);

    GstPad* infer_src_pad = gst_element_get_static_pad(infer_nvmm_caps, "src");
    streammux_sink_pad = gst_element_request_pad_simple(streammux, "sink_0");
    if (infer_src_pad == nullptr || streammux_sink_pad == nullptr
        || gst_pad_link(infer_src_pad, streammux_sink_pad) != GST_PAD_LINK_OK)
    {
      if (infer_src_pad != nullptr) {
        gst_object_unref(infer_src_pad);
      }
      release_request_pad(streammux, streammux_sink_pad);
      throw std::runtime_error("Failed to link inference branch to nvstreammux");
    }
    gst_object_unref(infer_src_pad);

    if (need_color_branch) {
      if (!gst_element_link_many(color_queue, color_convert, color_caps, color_sink_elem, nullptr))
      {
        throw std::runtime_error("Failed to link DeepStream color branch");
      }
      tee_color_pad = gst_element_request_pad_simple(tee, "src_%u");
      GstPad* color_queue_sink_pad = gst_element_get_static_pad(color_queue, "sink");
      if (tee_color_pad == nullptr || color_queue_sink_pad == nullptr
          || gst_pad_link(tee_color_pad, color_queue_sink_pad) != GST_PAD_LINK_OK)
      {
        release_request_pad(tee, tee_color_pad);
        if (color_queue_sink_pad != nullptr) {
          gst_object_unref(color_queue_sink_pad);
        }
        throw std::runtime_error("Failed to link tee to color queue");
      }
      gst_object_unref(color_queue_sink_pad);
    }

    attach_stage_profile_probes();

    bus = gst_element_get_bus(pipeline);
    if (bus == nullptr) {
      throw std::runtime_error("Failed to acquire pipeline bus");
    }

    const GstStateChangeReturn state_ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (state_ret == GST_STATE_CHANGE_FAILURE) {
      const std::string diagnostics = collect_bus_diagnostics(bus);
      throw std::runtime_error("Failed to set DeepStream pipeline to PLAYING"
                               + (diagnostics.empty() ? std::string{} : (": " + diagnostics)));
    }
    if (state_ret == GST_STATE_CHANGE_ASYNC) {
      const GstStateChangeReturn wait_ret =
        gst_element_get_state(pipeline, nullptr, nullptr, 2 * GST_SECOND);
      if (wait_ret == GST_STATE_CHANGE_FAILURE) {
        const std::string diagnostics = collect_bus_diagnostics(bus);
        throw std::runtime_error("DeepStream pipeline failed while waiting for PLAYING"
                                 + (diagnostics.empty() ? std::string{} : (": " + diagnostics)));
      }
    }
    query_source_actuals();
  }

  void start_threads()
  {
    running = true;
    if (is_realsense_source()) {
#if defined(HAVE_LIBREALSENSE2)
      source_thread = std::thread([this]() { this->realsense_source_loop(); });
#endif
    }
    infer_thread = std::thread([this]() { this->infer_loop(); });
    if (color_sink != nullptr) {
      color_thread = std::thread([this]() { this->color_loop(); });
    }
    bus_thread = std::thread([this]() { this->bus_loop(); });
  }

  void bus_loop()
  {
    while (running) {
      GstMessage* message = gst_bus_timed_pop_filtered(bus, 100 * GST_MSECOND,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_WARNING));
      if (message == nullptr) {
        continue;
      }

      switch (GST_MESSAGE_TYPE(message)) {
        case GST_MESSAGE_ERROR: {
          GError* error = nullptr;
          gchar* debug = nullptr;
          gst_message_parse_error(message, &error, &debug);
          RCLCPP_ERROR(logger, "DeepStream pipeline error: %s debug=%s",
            error != nullptr ? error->message : "unknown", debug != nullptr ? debug : "");
          if (error != nullptr) {
            g_error_free(error);
          }
          if (debug != nullptr) {
            g_free(debug);
          }
          break;
        }
        case GST_MESSAGE_WARNING: {
          GError* error = nullptr;
          gchar* debug = nullptr;
          gst_message_parse_warning(message, &error, &debug);
          RCLCPP_WARN(logger, "DeepStream pipeline warning: %s debug=%s",
            error != nullptr ? error->message : "unknown", debug != nullptr ? debug : "");
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

  FramePacket sample_to_bgr_packet(GstSample* sample)
  {
    FramePacket packet{};
    if (sample == nullptr) {
      return packet;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
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

  std::vector<int64_t> dims_to_shape(const NvDsInferDims& dims) const
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
    const NvDsInferTensorMeta* tensor_meta, const NvDsInferLayerInfo& layer) const
  {
    const auto shape = dims_to_shape(layer.inferDims);
    std::size_t volume = 1;
    for (const auto dim : shape) {
      volume *= static_cast<std::size_t>(std::max<int64_t>(1, dim));
    }

    const int output_index = static_cast<int>(&layer - tensor_meta->output_layers_info);
    const void* host_ptr = nullptr;
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
      const auto* floats = static_cast<const float*>(host_ptr);
      std::copy(floats, floats + volume, values.begin());
      return values;
    }
    if (layer.dataType == HALF) {
      const auto* halves = static_cast<const std::uint16_t*>(host_ptr);
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

  std::optional<NvDsInferTensorMeta*> find_tensor_meta(NvDsFrameMeta* frame_meta) const
  {
    if (frame_meta == nullptr) {
      return std::nullopt;
    }
    for (NvDsMetaList* user_meta = frame_meta->frame_user_meta_list; user_meta != nullptr;
      user_meta = user_meta->next)
    {
      auto* meta = static_cast<NvDsUserMeta*>(user_meta->data);
      if (meta == nullptr || meta->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) {
        continue;
      }
      return static_cast<NvDsInferTensorMeta*>(meta->user_meta_data);
    }
    return std::nullopt;
  }

  void infer_loop()
  {
    while (running) {
      GstSample* sample = gst_app_sink_try_pull_sample(infer_sink, 100 * GST_MSECOND);
      if (sample == nullptr) {
        continue;
      }

      try {
        process_infer_sample(sample);
      } catch (const std::exception& exc) {
        RCLCPP_ERROR(logger, "Failed processing DeepStream inference sample: %s", exc.what());
      }

      gst_sample_unref(sample);
    }
  }

  void color_loop()
  {
    while (running) {
      GstSample* sample = gst_app_sink_try_pull_sample(color_sink, 100 * GST_MSECOND);
      if (sample == nullptr) {
        continue;
      }

      try {
        store_frame(sample_to_bgr_packet(sample));
      } catch (const std::exception& exc) {
        RCLCPP_ERROR(logger, "Failed processing DeepStream color sample: %s", exc.what());
      }

      gst_sample_unref(sample);
    }
  }

  void process_infer_sample(GstSample* sample)
  {
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (buffer == nullptr) {
      return;
    }

    NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
    if (batch_meta == nullptr) {
      throw std::runtime_error("DeepStream buffer missing batch meta");
    }

    for (NvDsMetaList* frame_list = batch_meta->frame_meta_list; frame_list != nullptr;
      frame_list = frame_list->next)
    {
      auto* frame_meta = static_cast<NvDsFrameMeta*>(frame_list->data);
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
      const auto& output_layer = (*tensor_meta)->output_layers_info[0];
      const auto tensor_values = tensor_to_float_vector(*tensor_meta, output_layer);
      const auto output_shape = dims_to_shape(output_layer.inferDims);

      Result result{};
      if (config.enable_color_verification || config.debug_publish_image_raw
          || config.preview_frame)
      {
        const auto pts =
          frame_meta->buf_pts != GST_CLOCK_TIME_NONE ? frame_meta->buf_pts : GST_BUFFER_PTS(buffer);
        if (pts != GST_CLOCK_TIME_NONE) {
          result.source_packet = lookup_frame(static_cast<std::uint64_t>(pts));
        } else {
          RCLCPP_WARN(logger,
            "DeepStream inference sample missing valid PTS; color/debug frame lookup skipped (%s)",
            gst_clock_to_string(pts).c_str());
        }
      }

      const bool have_source_packet =
        result.source_packet.has_value() && !result.source_packet->empty();
      const bool source_packet_matches_source_geometry =
        have_source_packet && !cuda_bayer_resizes_source;
      const int frame_meta_source_width = frame_meta->source_frame_width > 0
                                            ? static_cast<int>(frame_meta->source_frame_width)
                                            : config.source_width;
      const int frame_meta_source_height = frame_meta->source_frame_height > 0
                                             ? static_cast<int>(frame_meta->source_frame_height)
                                             : config.source_height;
      const int src_w = source_packet_matches_source_geometry
                          ? result.source_packet->width()
                          : (config.pre_mux_resize ? config.source_width : frame_meta_source_width);
      const int src_h =
        source_packet_matches_source_geometry
          ? result.source_packet->height()
          : (config.pre_mux_resize ? config.source_height : frame_meta_source_height);
      result.batch.letterbox =
        make_letterbox_metadata(src_w, src_h, config.imgsz_width, config.imgsz_height);
      result.batch.detections = parse_yolo_output_tensor(tensor_values.data(), output_shape,
        config.imgsz_width, config.imgsz_height, std::max(1, num_classes), config.conf_threshold,
        task_obb, end2end, export_includes_nms, &logger);
      const auto parse_done = std::chrono::steady_clock::now();
      result.timings.parse_ms =
        std::chrono::duration<double, std::milli>(parse_done - parse_start).count();

      callback(std::move(result));
    }
  }
#endif
};

DeepStreamRuntime::DeepStreamRuntime(rclcpp::Logger logger, Config config, ResultCallback callback)
    : impl_(std::make_unique<Impl>(std::move(logger), std::move(config), std::move(callback)))
{
  try {
    impl_->start();
  } catch (...) {
    impl_->stop();
    impl_.reset();
    throw;
  }
}

DeepStreamRuntime::~DeepStreamRuntime() = default;

const std::string& DeepStreamRuntime::backend_name() const
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
