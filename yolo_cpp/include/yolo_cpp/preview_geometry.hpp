#ifndef YOLO_CPP__PREVIEW_GEOMETRY_HPP_
#define YOLO_CPP__PREVIEW_GEOMETRY_HPP_

#include <cstdint>
#include <opencv2/core.hpp>

namespace yolo_cpp
{
// Inference-only nvvideoconvert rotation happens after the preview tee.
// Bring those pixels into the already-upright detection coordinate frame.
// Source videoflip and CUDA Bayer paths have already rotated both branches.
inline void copy_deepstream_preview_frame(
  const cv::Mat & bgr, cv::Mat & frame, bool flip_source,
  bool cuda_bayer_resizes_source, bool source_flip_present)
{
  if (flip_source && !cuda_bayer_resizes_source && !source_flip_present) {
    cv::flip(bgr, frame, -1);
  } else {
    frame = bgr.clone();
  }
}

// Message time, not subscription delivery time: queued data must not look live.
inline bool preview_sample_fresh(std::int64_t stamp_ns, std::int64_t now_ns)
{
  return stamp_ns > 0 && now_ns >= stamp_ns && now_ns - stamp_ns <= 250000000;
}
}  // namespace yolo_cpp
#endif
