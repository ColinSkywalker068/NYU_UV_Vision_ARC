#ifndef YOLO_CPP__CUDA_BAYER_RESIZE_INTERNAL_HPP_
#define YOLO_CPP__CUDA_BAYER_RESIZE_INTERNAL_HPP_

#include <cstddef>
#include <cstdint>

extern "C"
{

void * yolo_cuda_bayer_resize_create();

void yolo_cuda_bayer_resize_destroy(void * context);

bool yolo_cuda_bayer_resize_rgba(
  void * context,
  const std::uint8_t * input,
  int input_width,
  int input_height,
  int input_stride,
  std::uint8_t * output,
  int output_width,
  int output_height,
  int output_stride,
  int bayer_pattern,
  int flip_method,
  char * error_message,
  std::size_t error_message_size);

}  // extern "C"

#endif
