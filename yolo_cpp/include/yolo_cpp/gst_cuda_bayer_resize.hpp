#ifndef YOLO_CPP__GST_CUDA_BAYER_RESIZE_HPP_
#define YOLO_CPP__GST_CUDA_BAYER_RESIZE_HPP_

namespace yolo_cpp
{

/// Register the in-process CUDA Bayer resize GStreamer element when compiled in.
bool register_cuda_bayer_resize_element();

/// Return true when the CUDA Bayer resize element was built into this binary.
bool cuda_bayer_resize_available();

}  // namespace yolo_cpp

#endif
