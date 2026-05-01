#ifndef YOLO_CPP__FRAME_PACKET_HPP_
#define YOLO_CPP__FRAME_PACKET_HPP_

#include <opencv2/core.hpp>

#include <cstdint>
#include <memory>
#include <string>

namespace yolo_cpp
{

enum class PixelFormat
{
  kBgr8,
  kRgb8,
  kYuyv,
  kUnknown
};

enum class BackingStoreKind
{
  kHost,
  kPinnedHost,
  kGpu,
  kNvmm
};

struct FramePacket
{
  cv::Mat frame;
  PixelFormat pixel_format{PixelFormat::kUnknown};
  BackingStoreKind storage_kind{BackingStoreKind::kHost};
  std::int64_t capture_time_ns{0};
  std::string source_id;
  std::shared_ptr<void> owner;

  int width() const {return frame.cols;}
  int height() const {return frame.rows;}
  bool empty() const {return frame.empty();}
};

}  // namespace yolo_cpp

#endif
