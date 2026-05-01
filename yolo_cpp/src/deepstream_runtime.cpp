#include "yolo_cpp/deepstream_runtime.hpp"

#include "yolo_cpp/detection_postprocess.hpp"
#include "yolo_cpp/utils.hpp"

#include <rclcpp/logging.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <gstnvdsinfer.h>
#include <gstnvdsmeta.h>
#include <nvdsmeta.h>
#endif

namespace yolo_cpp
{

namespace
{

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

void replace_all(std::string & text, const std::string & needle, const std::string & replacement)
{
  std::size_t pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    text.replace(pos, needle.size(), replacement);
    pos += replacement.size();
  }
}

#if defined(HAVE_GSTREAMER) && defined(HAVE_DEEPSTREAM)
std::string gst_clock_to_string(GstClockTime value)
{
  if (value == GST_CLOCK_TIME_NONE) {
    return "GST_CLOCK_TIME_NONE";
  }
  return std::to_string(static_cast<std::uint64_t>(value));
}

void release_request_pad(GstElement * element, GstPad *& pad)
{
  if (element == nullptr || pad == nullptr) {
    return;
  }
  gst_element_release_request_pad(element, pad);
  gst_object_unref(pad);
  pad = nullptr;
}
#endif

}  // namespace

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
  GstElement * tee{nullptr};
  GstElement * infer_queue{nullptr};
  GstElement * infer_convert{nullptr};
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
#endif

  ~Impl()
  {
    stop();
  }

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

    const std::vector<std::string> required_elements = {
      config.source_element,
      "tee",
      "queue",
      "videoconvert",
      "nvvideoconvert",
      "nvstreammux",
      "nvinfer",
      "appsink",
    };
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

  void materialize_config()
  {
    std::string config_text = read_text_file(config.config_template_path);
    replace_all(config_text, "__MODEL_ENGINE_FILE__", config.model_path);
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

  GstElement * make_element(const char * factory, const char * name)
  {
    GstElement * element = gst_element_factory_make(factory, name);
    if (element == nullptr) {
      throw std::runtime_error(
              "Failed to create GStreamer element '" + std::string(factory) + "'");
    }
    return element;
  }

  void build_pipeline()
  {
    pipeline = gst_pipeline_new("yolo-deepstream-pipeline");
    if (pipeline == nullptr) {
      throw std::runtime_error("Failed to create DeepStream pipeline");
    }

    source = make_element(config.source_element.c_str(), "source");
    source_caps = make_element("capsfilter", "source_caps");
    tee = make_element("tee", "tee");
    infer_queue = make_element("queue", "infer_queue");
    infer_convert = make_element("videoconvert", "infer_convert");
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
      tee,
      infer_queue,
      infer_convert,
      infer_nvconvert,
      infer_nvmm_caps,
      streammux,
      nvinfer,
      infer_post_queue,
      infer_sink_elem,
      nullptr);
    if (need_color_branch) {
      gst_bin_add_many(
        GST_BIN(pipeline),
        color_queue,
        color_convert,
        color_caps,
        color_sink_elem,
        nullptr);
    }

    GstCaps * source_caps_value = gst_caps_new_simple(
      "video/x-raw",
      "width", G_TYPE_INT, config.source_width,
      "height", G_TYPE_INT, config.source_height,
      "framerate", GST_TYPE_FRACTION, config.source_fps, 1,
      nullptr);
    g_object_set(G_OBJECT(source_caps), "caps", source_caps_value, nullptr);
    gst_caps_unref(source_caps_value);

    GstCaps * infer_caps_value = gst_caps_new_simple(
      "video/x-raw(memory:NVMM)",
      "format", G_TYPE_STRING, "NV12",
      "width", G_TYPE_INT, config.imgsz_width,
      "height", G_TYPE_INT, config.imgsz_height,
      "framerate", GST_TYPE_FRACTION, config.source_fps, 1,
      nullptr);
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
      GstCaps * color_caps_value = gst_caps_new_simple(
        "video/x-raw",
        "format", G_TYPE_STRING, "BGR",
        "width", G_TYPE_INT, config.source_width,
        "height", G_TYPE_INT, config.source_height,
        "framerate", GST_TYPE_FRACTION, config.source_fps, 1,
        nullptr);
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

    if (!gst_element_link_many(source, source_caps, tee, nullptr)) {
      throw std::runtime_error("Failed to link DeepStream source path");
    }
    if (!gst_element_link_many(
        infer_queue, infer_convert, infer_nvconvert, infer_nvmm_caps, nullptr))
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
      throw std::runtime_error("Failed to set DeepStream pipeline to PLAYING");
    }
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
      const int src_w = frame_meta->source_frame_width > 0 ?
        static_cast<int>(frame_meta->source_frame_width) : config.source_width;
      const int src_h = frame_meta->source_frame_height > 0 ?
        static_cast<int>(frame_meta->source_frame_height) : config.source_height;
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
        &logger);
      const auto parse_done = std::chrono::steady_clock::now();
      result.timings.parse_ms = std::chrono::duration<double, std::milli>(
        parse_done - parse_start).count();

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

}  // namespace yolo_cpp
