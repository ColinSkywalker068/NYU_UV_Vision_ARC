#ifndef YOLO_CPP__PREVIEW_GEOMETRY_HPP_
#define YOLO_CPP__PREVIEW_GEOMETRY_HPP_

#include <cstdint>

namespace yolo_cpp
{
// Message time, not subscription delivery time: queued data must not look live.
inline bool preview_sample_fresh(std::int64_t stamp_ns, std::int64_t now_ns)
{
  return stamp_ns > 0 && now_ns >= stamp_ns && now_ns - stamp_ns <= 250000000;
}
}  // namespace yolo_cpp
#endif
