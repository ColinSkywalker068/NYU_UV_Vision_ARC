#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "cuda_bayer_resize.hpp"

namespace
{

enum BayerPattern
{
  kRggb = 0,
  kBggr = 1,
  kGrbg = 2,
  kGbrg = 3,
};

enum CudaPreprocessOutputType
{
  kOutputFloat32 = 0,
  kOutputFloat16 = 1,
};

struct CudaBayerResizeContext
{
  std::uint8_t* device_input{nullptr};
  std::uint8_t* device_output{nullptr};
  std::size_t input_bytes{0};
  std::size_t output_bytes{0};
  cudaStream_t stream{};
  bool sync_memcpy{false};
};

void set_error(char* message, std::size_t size, const char* prefix, cudaError_t status)
{
  if (message == nullptr || size == 0U) {
    return;
  }
  std::snprintf(message, size, "%s: %s", prefix, cudaGetErrorString(status));
}

bool ensure_device_buffer(std::uint8_t** buffer, std::size_t* current_size,
  std::size_t required_size, char* error_message, std::size_t error_message_size)
{
  if (*buffer != nullptr && *current_size >= required_size) {
    return true;
  }
  if (*buffer != nullptr) {
    cudaFree(*buffer);
    *buffer = nullptr;
    *current_size = 0U;
  }
  const cudaError_t status = cudaMalloc(reinterpret_cast<void**>(buffer), required_size);
  if (status != cudaSuccess) {
    set_error(error_message, error_message_size, "cudaMalloc failed", status);
    return false;
  }
  *current_size = required_size;
  return true;
}

bool env_flag_enabled(const char* name)
{
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return false;
  }
  return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0
         || std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0
         || std::strcmp(value, "ON") == 0 || std::strcmp(value, "yes") == 0
         || std::strcmp(value, "YES") == 0;
}

__device__ int clamp_int(int value, int low, int high)
{
  return max(low, min(high, value));
}

__device__ int bayer_channel(int x, int y, int pattern)
{
  const int xp = x & 1;
  const int yp = y & 1;
  switch (pattern) {
    case kBggr:
      if (yp == 0 && xp == 0) {
        return 2;
      }
      if (yp == 1 && xp == 1) {
        return 0;
      }
      return 1;
    case kGrbg:
      if (yp == 0 && xp == 1) {
        return 0;
      }
      if (yp == 1 && xp == 0) {
        return 2;
      }
      return 1;
    case kGbrg:
      if (yp == 0 && xp == 0) {
        return 1;
      }
      if (yp == 0 && xp == 1) {
        return 2;
      }
      if (yp == 1 && xp == 0) {
        return 0;
      }
      return 1;
    case kRggb:
    default:
      if (yp == 0 && xp == 0) {
        return 0;
      }
      if (yp == 1 && xp == 1) {
        return 2;
      }
      return 1;
  }
}

__device__ unsigned char sample_channel(const std::uint8_t* input, int width, int height,
  int stride, int sx, int sy, int pattern, int target_channel)
{
  int sum = 0;
  int count = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    const int y = clamp_int(sy + dy, 0, height - 1);
    for (int dx = -1; dx <= 1; ++dx) {
      const int x = clamp_int(sx + dx, 0, width - 1);
      if (bayer_channel(x, y, pattern) == target_channel) {
        sum += input[(y * stride) + x];
        ++count;
      }
    }
  }
  if (count == 0) {
    return input[(clamp_int(sy, 0, height - 1) * stride) + clamp_int(sx, 0, width - 1)];
  }
  return static_cast<unsigned char>(sum / count);
}

__global__ void bayer_resize_rgba_kernel(const std::uint8_t* input, int input_width,
  int input_height, int input_stride, std::uint8_t* output, int output_width, int output_height,
  int output_stride, int pattern, int flip_method)
{
  const int ox = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int oy = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (ox >= output_width || oy >= output_height) {
    return;
  }

  const int logical_x = flip_method == 2 ? output_width - 1 - ox : ox;
  const int logical_y = flip_method == 2 ? output_height - 1 - oy : oy;
  const float scale_x = static_cast<float>(input_width) / static_cast<float>(output_width);
  const float scale_y = static_cast<float>(input_height) / static_cast<float>(output_height);
  const int sx = clamp_int(
    static_cast<int>((static_cast<float>(logical_x) + 0.5F) * scale_x), 0, input_width - 1);
  const int sy = clamp_int(
    static_cast<int>((static_cast<float>(logical_y) + 0.5F) * scale_y), 0, input_height - 1);

  std::uint8_t* pixel = output + (oy * output_stride) + (ox * 4);
  pixel[0] = sample_channel(input, input_width, input_height, input_stride, sx, sy, pattern, 0);
  pixel[1] = sample_channel(input, input_width, input_height, input_stride, sx, sy, pattern, 1);
  pixel[2] = sample_channel(input, input_width, input_height, input_stride, sx, sy, pattern, 2);
  pixel[3] = 255U;
}

template <typename OutputT>
__device__ void write_normalized(OutputT* output, int offset, float value);

template <>
__device__ void write_normalized<float>(float* output, int offset, float value)
{
  output[offset] = value;
}

template <>
__device__ void write_normalized<__half>(__half* output, int offset, float value)
{
  output[offset] = __float2half(value);
}

template <typename OutputT>
__global__ void bayer_resize_chw_kernel(const std::uint8_t* input, int input_width,
  int input_height, int input_stride, OutputT* output, int output_width, int output_height,
  int pattern, int flip_method)
{
  const int ox = (blockIdx.x * blockDim.x) + threadIdx.x;
  const int oy = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (ox >= output_width || oy >= output_height) {
    return;
  }

  const float scale =
    fminf(static_cast<float>(output_width) / static_cast<float>(input_width),
      static_cast<float>(output_height) / static_cast<float>(input_height));
  const int resized_width =
    max(1, min(output_width, static_cast<int>(static_cast<float>(input_width) * scale)));
  const int resized_height =
    max(1, min(output_height, static_cast<int>(static_cast<float>(input_height) * scale)));
  const int pad_x = (output_width - resized_width) / 2;
  const int pad_y = (output_height - resized_height) / 2;
  const int plane = output_width * output_height;
  const int output_offset = (oy * output_width) + ox;

  if (ox < pad_x || ox >= pad_x + resized_width || oy < pad_y || oy >= pad_y + resized_height) {
    constexpr float kPadValue = 114.0F / 255.0F;
    write_normalized(output, output_offset, kPadValue);
    write_normalized(output, plane + output_offset, kPadValue);
    write_normalized(output, (2 * plane) + output_offset, kPadValue);
    return;
  }

  const int resized_x = ox - pad_x;
  const int resized_y = oy - pad_y;
  const int logical_x = flip_method == 2 ? resized_width - 1 - resized_x : resized_x;
  const int logical_y = flip_method == 2 ? resized_height - 1 - resized_y : resized_y;
  const float scale_x = static_cast<float>(input_width) / static_cast<float>(resized_width);
  const float scale_y = static_cast<float>(input_height) / static_cast<float>(resized_height);
  const int sx = clamp_int(
    static_cast<int>((static_cast<float>(logical_x) + 0.5F) * scale_x), 0, input_width - 1);
  const int sy = clamp_int(
    static_cast<int>((static_cast<float>(logical_y) + 0.5F) * scale_y), 0, input_height - 1);

  const float red = static_cast<float>(
                      sample_channel(input, input_width, input_height, input_stride, sx, sy,
                        pattern, 0))
                    / 255.0F;
  const float green = static_cast<float>(
                        sample_channel(input, input_width, input_height, input_stride, sx, sy,
                          pattern, 1))
                      / 255.0F;
  const float blue = static_cast<float>(
                       sample_channel(input, input_width, input_height, input_stride, sx, sy,
                         pattern, 2))
                     / 255.0F;
  write_normalized(output, output_offset, red);
  write_normalized(output, plane + output_offset, green);
  write_normalized(output, (2 * plane) + output_offset, blue);
}

}  // namespace

extern "C" void* yolo_cuda_bayer_resize_create()
{
  auto* context = new CudaBayerResizeContext();
  context->sync_memcpy = env_flag_enabled("YOLO_CUDA_BAYER_SYNC_MEMCPY");
  if (cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking) != cudaSuccess) {
    delete context;
    return nullptr;
  }
  return context;
}

extern "C" void yolo_cuda_bayer_resize_destroy(void* raw_context)
{
  auto* context = static_cast<CudaBayerResizeContext*>(raw_context);
  if (context == nullptr) {
    return;
  }
  if (context->device_input != nullptr) {
    cudaFree(context->device_input);
  }
  if (context->device_output != nullptr) {
    cudaFree(context->device_output);
  }
  if (context->stream != nullptr) {
    cudaStreamDestroy(context->stream);
  }
  delete context;
}

extern "C" bool yolo_cuda_bayer_resize_rgba(void* raw_context, const std::uint8_t* input,
  int input_width, int input_height, int input_stride, std::uint8_t* output, int output_width,
  int output_height, int output_stride, int bayer_pattern, int flip_method, char* error_message,
  std::size_t error_message_size)
{
  auto* context = static_cast<CudaBayerResizeContext*>(raw_context);
  if (context == nullptr || input == nullptr || output == nullptr || input_width <= 0
      || input_height <= 0 || output_width <= 0 || output_height <= 0)
  {
    if (error_message != nullptr && error_message_size > 0U) {
      std::snprintf(error_message, error_message_size, "invalid CUDA Bayer resize arguments");
    }
    return false;
  }
  if (input_stride < input_width || output_stride < output_width * 4) {
    if (error_message != nullptr && error_message_size > 0U) {
      std::snprintf(error_message, error_message_size,
        "invalid CUDA Bayer resize stride input=%d width=%d output=%d required_output=%d",
        input_stride, input_width, output_stride, output_width * 4);
    }
    return false;
  }

  const std::size_t input_bytes =
    static_cast<std::size_t>(input_stride) * static_cast<std::size_t>(input_height);
  const std::size_t output_bytes =
    static_cast<std::size_t>(output_stride) * static_cast<std::size_t>(output_height);
  if (!ensure_device_buffer(&context->device_input, &context->input_bytes, input_bytes,
        error_message, error_message_size)
      || !ensure_device_buffer(&context->device_output, &context->output_bytes, output_bytes,
        error_message, error_message_size))
  {
    return false;
  }

  cudaError_t status =
    context->sync_memcpy
      ? cudaMemcpy(context->device_input, input, input_bytes, cudaMemcpyHostToDevice)
      : cudaMemcpyAsync(
          context->device_input, input, input_bytes, cudaMemcpyHostToDevice, context->stream);
  if (status != cudaSuccess) {
    set_error(error_message, error_message_size,
      context->sync_memcpy ? "cudaMemcpy input failed" : "cudaMemcpyAsync input failed", status);
    return false;
  }

  const dim3 block(16, 16);
  const dim3 grid(static_cast<unsigned int>((output_width + block.x - 1) / block.x),
    static_cast<unsigned int>((output_height + block.y - 1) / block.y));
  bayer_resize_rgba_kernel<<<grid, block, 0, context->stream>>>(context->device_input, input_width,
    input_height, input_stride, context->device_output, output_width, output_height, output_stride,
    bayer_pattern, flip_method);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    set_error(error_message, error_message_size, "Bayer resize kernel launch failed", status);
    return false;
  }

  status =
    context->sync_memcpy
      ? cudaMemcpy(output, context->device_output, output_bytes, cudaMemcpyDeviceToHost)
      : cudaMemcpyAsync(
          output, context->device_output, output_bytes, cudaMemcpyDeviceToHost, context->stream);
  if (status != cudaSuccess) {
    set_error(error_message, error_message_size,
      context->sync_memcpy ? "cudaMemcpy output failed" : "cudaMemcpyAsync output failed", status);
    return false;
  }

  status = cudaStreamSynchronize(context->stream);
  if (status != cudaSuccess) {
    set_error(error_message, error_message_size, "cudaStreamSynchronize failed", status);
    return false;
  }
  return true;
}

extern "C" bool yolo_cuda_bayer_resize_chw(void* raw_context, const std::uint8_t* input,
  int input_width, int input_height, int input_stride, void* device_output, int output_width,
  int output_height, int bayer_pattern, int flip_method, int output_data_type, char* error_message,
  std::size_t error_message_size)
{
  auto* context = static_cast<CudaBayerResizeContext*>(raw_context);
  if (context == nullptr || input == nullptr || device_output == nullptr || input_width <= 0
      || input_height <= 0 || output_width <= 0 || output_height <= 0)
  {
    if (error_message != nullptr && error_message_size > 0U) {
      std::snprintf(error_message, error_message_size, "invalid CUDA Bayer CHW arguments");
    }
    return false;
  }
  if (input_stride < input_width) {
    if (error_message != nullptr && error_message_size > 0U) {
      std::snprintf(error_message, error_message_size,
        "invalid CUDA Bayer CHW input stride input=%d width=%d", input_stride, input_width);
    }
    return false;
  }

  const std::size_t input_bytes =
    static_cast<std::size_t>(input_stride) * static_cast<std::size_t>(input_height);
  if (!ensure_device_buffer(
        &context->device_input, &context->input_bytes, input_bytes, error_message,
        error_message_size))
  {
    return false;
  }

  cudaError_t status =
    context->sync_memcpy
      ? cudaMemcpy(context->device_input, input, input_bytes, cudaMemcpyHostToDevice)
      : cudaMemcpyAsync(
          context->device_input, input, input_bytes, cudaMemcpyHostToDevice, context->stream);
  if (status != cudaSuccess) {
    set_error(error_message, error_message_size,
      context->sync_memcpy ? "cudaMemcpy CHW input failed" : "cudaMemcpyAsync CHW input failed",
      status);
    return false;
  }

  const dim3 block(16, 16);
  const dim3 grid(static_cast<unsigned int>((output_width + block.x - 1) / block.x),
    static_cast<unsigned int>((output_height + block.y - 1) / block.y));
  if (output_data_type == kOutputFloat16) {
    bayer_resize_chw_kernel<<<grid, block, 0, context->stream>>>(context->device_input,
      input_width, input_height, input_stride, static_cast<__half*>(device_output), output_width,
      output_height, bayer_pattern, flip_method);
  } else {
    bayer_resize_chw_kernel<<<grid, block, 0, context->stream>>>(context->device_input,
      input_width, input_height, input_stride, static_cast<float*>(device_output), output_width,
      output_height, bayer_pattern, flip_method);
  }
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    set_error(error_message, error_message_size, "Bayer CHW kernel launch failed", status);
    return false;
  }

  status = cudaStreamSynchronize(context->stream);
  if (status != cudaSuccess) {
    set_error(error_message, error_message_size, "cudaStreamSynchronize CHW failed", status);
    return false;
  }
  return true;
}
