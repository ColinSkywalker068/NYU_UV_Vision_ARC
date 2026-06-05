#include "yolo_cpp/gst_cuda_bayer_resize.hpp"

#if defined(HAVE_GSTREAMER)
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#endif

#if defined(HAVE_GSTREAMER) && defined(HAVE_YOLO_CUDA_BAYER_RESIZE)
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

#include "cuda_bayer_resize.hpp"
#endif

namespace yolo_cpp
{

#if defined(HAVE_GSTREAMER) && defined(HAVE_YOLO_CUDA_BAYER_RESIZE)
namespace
{

constexpr const char* kElementName = "yolocudabayerresize";

enum BayerPattern
{
  kRggb = 0,
  kBggr = 1,
  kGrbg = 2,
  kGbrg = 3,
};

struct _GstYoloCudaBayerResize
{
  GstBaseTransform parent;
  gint output_width;
  gint output_height;
  gint flip_method;
  gint input_width;
  gint input_height;
  gint output_stride;
  BayerPattern pattern;
  void* cuda_context;
  guint64 frame_count;
  gboolean logged_input_memory;
};

struct _GstYoloCudaBayerResizeClass
{
  GstBaseTransformClass parent_class;
};

using GstYoloCudaBayerResize = _GstYoloCudaBayerResize;
using GstYoloCudaBayerResizeClass = _GstYoloCudaBayerResizeClass;

G_DEFINE_TYPE(GstYoloCudaBayerResize, gst_yolo_cuda_bayer_resize, GST_TYPE_BASE_TRANSFORM)

enum
{
  kProp0,
  kPropOutputWidth,
  kPropOutputHeight,
  kPropFlipMethod,
};

BayerPattern pattern_from_caps(const gchar* format)
{
  if (format == nullptr) {
    return kRggb;
  }
  const std::string value(format);
  if (value == "bggr") {
    return kBggr;
  }
  if (value == "grbg") {
    return kGrbg;
  }
  if (value == "gbrg") {
    return kGbrg;
  }
  return kRggb;
}

GstCaps* fixed_output_caps(const GstYoloCudaBayerResize* self)
{
  const gint width = std::max(2, self->output_width);
  const gint height = std::max(2, self->output_height);
  return gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "RGBA", "width", G_TYPE_INT,
    width, "height", G_TYPE_INT, height, nullptr);
}

GstCaps* bayer_sink_caps()
{
  return gst_caps_from_string(
    "video/x-bayer,format=(string){rggb,bggr,grbg,gbrg},"
    "width=(int)[2,MAX],height=(int)[2,MAX],framerate=(fraction)[0/1,2147483647/1]");
}

void copy_caps_field(GstCaps* destination, const GstCaps* source, const char* field)
{
  if (destination == nullptr || source == nullptr || field == nullptr || gst_caps_is_empty(source)
      || gst_caps_is_any(source))
  {
    return;
  }
  const GstStructure* source_structure = gst_caps_get_structure(source, 0);
  const GValue* value = gst_structure_get_value(source_structure, field);
  if (value == nullptr) {
    return;
  }
  for (guint i = 0; i < gst_caps_get_size(destination); ++i) {
    GstStructure* destination_structure = gst_caps_get_structure(destination, i);
    gst_structure_set_value(destination_structure, field, value);
  }
}

void copy_timing_caps_fields(GstCaps* destination, const GstCaps* source)
{
  copy_caps_field(destination, source, "framerate");
  copy_caps_field(destination, source, "pixel-aspect-ratio");
}

void gst_yolo_cuda_bayer_resize_set_property(
  GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec)
{
  auto* self = reinterpret_cast<GstYoloCudaBayerResize*>(object);
  switch (prop_id) {
    case kPropOutputWidth:
      self->output_width = g_value_get_int(value);
      break;
    case kPropOutputHeight:
      self->output_height = g_value_get_int(value);
      break;
    case kPropFlipMethod:
      self->flip_method = g_value_get_int(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

void gst_yolo_cuda_bayer_resize_get_property(
  GObject* object, guint prop_id, GValue* value, GParamSpec* pspec)
{
  auto* self = reinterpret_cast<GstYoloCudaBayerResize*>(object);
  switch (prop_id) {
    case kPropOutputWidth:
      g_value_set_int(value, self->output_width);
      break;
    case kPropOutputHeight:
      g_value_set_int(value, self->output_height);
      break;
    case kPropFlipMethod:
      g_value_set_int(value, self->flip_method);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

GstCaps* gst_yolo_cuda_bayer_resize_transform_caps(
  GstBaseTransform* transform, GstPadDirection direction, GstCaps* caps, GstCaps* filter)
{
  auto* self = reinterpret_cast<GstYoloCudaBayerResize*>(transform);
  GstCaps* result = nullptr;
  if (direction == GST_PAD_SINK) {
    result = fixed_output_caps(self);
    copy_timing_caps_fields(result, caps);
  } else {
    result = bayer_sink_caps();
    copy_timing_caps_fields(result, caps);
  }
  if (filter != nullptr) {
    GstCaps* intersection = gst_caps_intersect_full(filter, result, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref(result);
    result = intersection;
  }
  (void)caps;
  return result;
}

GstCaps* gst_yolo_cuda_bayer_resize_fixate_caps(
  GstBaseTransform* transform, GstPadDirection direction, GstCaps*, GstCaps* othercaps)
{
  auto* self = reinterpret_cast<GstYoloCudaBayerResize*>(transform);
  if (direction == GST_PAD_SINK) {
    GstCaps* fixed = fixed_output_caps(self);
    copy_timing_caps_fields(fixed, othercaps);
    gst_caps_unref(othercaps);
    return fixed;
  }
  return gst_caps_fixate(othercaps);
}

gboolean gst_yolo_cuda_bayer_resize_set_caps(
  GstBaseTransform* transform, GstCaps* incaps, GstCaps* outcaps)
{
  auto* self = reinterpret_cast<GstYoloCudaBayerResize*>(transform);
  GstStructure* in_structure = gst_caps_get_structure(incaps, 0);
  GstStructure* out_structure = gst_caps_get_structure(outcaps, 0);
  const gchar* format = gst_structure_get_string(in_structure, "format");
  self->pattern = pattern_from_caps(format);
  if (!gst_structure_get_int(in_structure, "width", &self->input_width)
      || !gst_structure_get_int(in_structure, "height", &self->input_height))
  {
    GST_ERROR_OBJECT(self, "Input caps missing width/height");
    return FALSE;
  }
  if (!gst_structure_get_int(out_structure, "width", &self->output_width)
      || !gst_structure_get_int(out_structure, "height", &self->output_height))
  {
    GST_ERROR_OBJECT(self, "Output caps missing width/height");
    return FALSE;
  }
  self->output_stride = self->output_width * 4;
  return TRUE;
}

gboolean gst_yolo_cuda_bayer_resize_transform_size(GstBaseTransform* transform,
  GstPadDirection direction, GstCaps*, gsize size, GstCaps*, gsize* othersize)
{
  auto* self = reinterpret_cast<GstYoloCudaBayerResize*>(transform);
  if (direction == GST_PAD_SINK) {
    *othersize = static_cast<gsize>(std::max(2, self->output_width))
                 * static_cast<gsize>(std::max(2, self->output_height)) * 4U;
  } else {
    if (self->input_width <= 0 || self->input_height <= 0) {
      *othersize = size;
    } else {
      *othersize = static_cast<gsize>(self->input_width) * static_cast<gsize>(self->input_height);
    }
  }
  return TRUE;
}

GstFlowReturn gst_yolo_cuda_bayer_resize_transform(
  GstBaseTransform* transform, GstBuffer* inbuf, GstBuffer* outbuf)
{
  auto* self = reinterpret_cast<GstYoloCudaBayerResize*>(transform);
  const guint64 frame_id = ++self->frame_count;
  const cudaError_t prior_cuda_error = cudaGetLastError();
  if (prior_cuda_error != cudaSuccess) {
    GST_WARNING_OBJECT(self, "CUDA Bayer frame=%" G_GUINT64_FORMAT
                             " prior CUDA error before transform: %s",
      frame_id, cudaGetErrorString(prior_cuda_error));
  }
  if (self->cuda_context == nullptr) {
    self->cuda_context = yolo_cuda_bayer_resize_create();
  }
  if (self->cuda_context == nullptr) {
    GST_ELEMENT_ERROR(GST_ELEMENT(self), RESOURCE, FAILED,
      ("Failed to create CUDA Bayer resize context"), (nullptr));
    return GST_FLOW_ERROR;
  }

  GstMapInfo in_map{};
  GstMapInfo out_map{};
  if (!gst_buffer_map(inbuf, &in_map, GST_MAP_READ)) {
    GST_ELEMENT_ERROR(
      GST_ELEMENT(self), RESOURCE, READ, ("Failed to map Bayer input buffer"), (nullptr));
    return GST_FLOW_ERROR;
  }
  if (!gst_buffer_map(outbuf, &out_map, GST_MAP_WRITE)) {
    gst_buffer_unmap(inbuf, &in_map);
    GST_ELEMENT_ERROR(
      GST_ELEMENT(self), RESOURCE, WRITE, ("Failed to map RGBA output buffer"), (nullptr));
    return GST_FLOW_ERROR;
  }

  GstMemory* input_memory = gst_buffer_peek_memory(inbuf, 0);
  const char* input_allocator = "none";
  guint input_memory_flags = 0U;
  if (input_memory != nullptr) {
    input_memory_flags = static_cast<guint>(GST_MINI_OBJECT_FLAGS(input_memory));
    if (input_memory->allocator != nullptr) {
      const char* memory_type = gst_allocator_get_memory_type(input_memory->allocator);
      input_allocator = memory_type != nullptr ? memory_type : "unknown";
    } else {
      input_allocator = "null-allocator";
    }
  }
  if (!self->logged_input_memory || frame_id <= 3U || frame_id % 300U == 0U) {
    GST_WARNING_OBJECT(self,
      "CUDA Bayer frame=%" G_GUINT64_FORMAT
      " input_allocator=%s input_flags=0x%x input_size=%" G_GSIZE_FORMAT
      " output_size=%" G_GSIZE_FORMAT " sync_memcpy=%s",
      frame_id, input_allocator, input_memory_flags, in_map.size, out_map.size,
      g_getenv("YOLO_CUDA_BAYER_SYNC_MEMCPY") != nullptr ? g_getenv("YOLO_CUDA_BAYER_SYNC_MEMCPY")
                                                         : "false");
    self->logged_input_memory = TRUE;
  }

  std::array<char, 256> error_message{};
  if (self->input_width <= 0 || self->input_height <= 0 || self->output_width <= 0
      || self->output_height <= 0)
  {
    gst_buffer_unmap(outbuf, &out_map);
    gst_buffer_unmap(inbuf, &in_map);
    GST_ELEMENT_ERROR(GST_ELEMENT(self), RESOURCE, FAILED,
      ("Invalid CUDA Bayer resize caps input=%dx%d output=%dx%d", self->input_width,
        self->input_height, self->output_width, self->output_height),
      (nullptr));
    return GST_FLOW_ERROR;
  }

  const gsize minimum_input_size =
    static_cast<gsize>(self->input_width) * static_cast<gsize>(self->input_height);
  const gsize minimum_output_size = static_cast<gsize>(self->output_width)
                                    * static_cast<gsize>(self->output_height) * 4U;
  if (in_map.size < minimum_input_size || out_map.size < minimum_output_size) {
    gst_buffer_unmap(outbuf, &out_map);
    gst_buffer_unmap(inbuf, &in_map);
    GST_ELEMENT_ERROR(GST_ELEMENT(self), RESOURCE, FAILED,
      ("CUDA Bayer resize buffer too small: input_size=%" G_GSIZE_FORMAT
       " required_input=%" G_GSIZE_FORMAT " output_size=%" G_GSIZE_FORMAT
       " required_output=%" G_GSIZE_FORMAT,
        in_map.size, minimum_input_size, out_map.size, minimum_output_size),
      (nullptr));
    return GST_FLOW_ERROR;
  }

  const int input_stride = std::max(self->input_width,
    self->input_height > 0 ? static_cast<int>(in_map.size / static_cast<gsize>(self->input_height))
                           : self->input_width);
  const int output_stride = std::max(self->output_stride,
    self->output_height > 0
      ? static_cast<int>(out_map.size / static_cast<gsize>(self->output_height))
      : self->output_stride);
  const gsize required_input_size =
    static_cast<gsize>(input_stride) * static_cast<gsize>(self->input_height);
  const gsize required_output_size =
    static_cast<gsize>(output_stride) * static_cast<gsize>(self->output_height);
  if (required_input_size > in_map.size || required_output_size > out_map.size) {
    gst_buffer_unmap(outbuf, &out_map);
    gst_buffer_unmap(inbuf, &in_map);
    GST_ELEMENT_ERROR(GST_ELEMENT(self), RESOURCE, FAILED,
      ("CUDA Bayer resize stride exceeds mapped buffer: input_stride=%d input_size=%"
       G_GSIZE_FORMAT " required_input=%" G_GSIZE_FORMAT
       " output_stride=%d output_size=%" G_GSIZE_FORMAT " required_output=%" G_GSIZE_FORMAT,
        input_stride, in_map.size, required_input_size, output_stride, out_map.size,
        required_output_size),
      (nullptr));
    return GST_FLOW_ERROR;
  }

  const bool ok = yolo_cuda_bayer_resize_rgba(self->cuda_context, in_map.data, self->input_width,
    self->input_height, input_stride, out_map.data, self->output_width, self->output_height,
    output_stride, static_cast<int>(self->pattern), self->flip_method, error_message.data(),
    error_message.size());
  gst_buffer_unmap(outbuf, &out_map);
  gst_buffer_unmap(inbuf, &in_map);
  if (!ok) {
    GST_ELEMENT_ERROR(GST_ELEMENT(self), RESOURCE, FAILED,
      ("CUDA Bayer frame=%" G_GUINT64_FORMAT
       " failed: %s input_allocator=%s input_flags=0x%x input_stride=%d output_stride=%d "
       "input=%dx%d output=%dx%d",
        frame_id, error_message.data(), input_allocator, input_memory_flags, input_stride,
        output_stride, self->input_width, self->input_height, self->output_width,
        self->output_height),
      (nullptr));
    return GST_FLOW_ERROR;
  }
  return GST_FLOW_OK;
}

void gst_yolo_cuda_bayer_resize_finalize(GObject* object)
{
  auto* self = reinterpret_cast<GstYoloCudaBayerResize*>(object);
  if (self->cuda_context != nullptr) {
    yolo_cuda_bayer_resize_destroy(self->cuda_context);
    self->cuda_context = nullptr;
  }
  G_OBJECT_CLASS(gst_yolo_cuda_bayer_resize_parent_class)->finalize(object);
}

void gst_yolo_cuda_bayer_resize_class_init(GstYoloCudaBayerResizeClass* klass)
{
  auto* gobject_class = G_OBJECT_CLASS(klass);
  auto* transform_class = GST_BASE_TRANSFORM_CLASS(klass);
  gobject_class->set_property = gst_yolo_cuda_bayer_resize_set_property;
  gobject_class->get_property = gst_yolo_cuda_bayer_resize_get_property;
  gobject_class->finalize = gst_yolo_cuda_bayer_resize_finalize;

  g_object_class_install_property(gobject_class, kPropOutputWidth,
    g_param_spec_int("output-width", "Output width", "RGBA output width", 2, G_MAXINT, 640,
      static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(gobject_class, kPropOutputHeight,
    g_param_spec_int("output-height", "Output height", "RGBA output height", 2, G_MAXINT, 400,
      static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(gobject_class, kPropFlipMethod,
    g_param_spec_int("flip-method", "Flip method", "0 for none, 2 for 180-degree rotation", 0, 2, 0,
      static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass), "YOLO CUDA Bayer resize",
    "Filter/Converter/Video", "CUDA Bayer demosaic plus resize to RGBA", "CV Detection");

  GstCaps* sink_caps = bayer_sink_caps();
  GstCaps* src_caps =
    gst_caps_from_string("video/x-raw,format=(string)RGBA,width=(int)[2,MAX],height=(int)[2,MAX],"
                         "framerate=(fraction)[0/1,2147483647/1]");
  gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
    gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, sink_caps));
  gst_element_class_add_pad_template(
    GST_ELEMENT_CLASS(klass), gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, src_caps));

  transform_class->transform_caps = gst_yolo_cuda_bayer_resize_transform_caps;
  transform_class->fixate_caps = gst_yolo_cuda_bayer_resize_fixate_caps;
  transform_class->set_caps = gst_yolo_cuda_bayer_resize_set_caps;
  transform_class->transform_size = gst_yolo_cuda_bayer_resize_transform_size;
  transform_class->transform = gst_yolo_cuda_bayer_resize_transform;
}

void gst_yolo_cuda_bayer_resize_init(GstYoloCudaBayerResize* self)
{
  self->output_width = 640;
  self->output_height = 400;
  self->flip_method = 0;
  self->input_width = 0;
  self->input_height = 0;
  self->output_stride = 0;
  self->pattern = kRggb;
  self->cuda_context = nullptr;
  self->frame_count = 0U;
  self->logged_input_memory = FALSE;
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), FALSE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

std::once_flag g_register_once;
bool g_registered = false;

}  // namespace

bool cuda_bayer_resize_available()
{
  return true;
}

bool register_cuda_bayer_resize_element()
{
  std::call_once(g_register_once, []() {
    g_registered = gst_element_register(
      nullptr, kElementName, GST_RANK_NONE, gst_yolo_cuda_bayer_resize_get_type());
  });
  return g_registered;
}

#else

bool cuda_bayer_resize_available()
{
  return false;
}

bool register_cuda_bayer_resize_element()
{
  return false;
}

#endif

}  // namespace yolo_cpp
