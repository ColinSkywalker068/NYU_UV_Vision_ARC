#ifndef YOLO_CPP__FRAME_PACKET_HPP_
#define YOLO_CPP__FRAME_PACKET_HPP_

#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <string>

namespace yolo_cpp
{

enum class PixelFormat
{
  /// OpenCV-style 8-bit BGR image data.
  kBgr8,
  /// 8-bit RGB image data.
  kRgb8,
  /// Packed YUYV camera data.
  kYuyv,
  /// 8-bit raw Bayer mosaic formats.
  kBayerRggb8,
  kBayerBggr8,
  kBayerGrbg8,
  kBayerGbrg8,
  /// Unknown or backend-specific format.
  kUnknown
};

enum class BackingStoreKind
{
  /// Standard host memory.
  kHost,
  /// Page-locked host memory suitable for faster device transfer.
  kPinnedHost,
  /// Device memory.
  kGpu,
  /// NVIDIA multimedia buffer memory.
  kNvmm
};

/// Captured frame plus ownership and transport metadata.
struct FramePacket
{
  cv::Mat frame;
  PixelFormat pixel_format{PixelFormat::kUnknown};
  BackingStoreKind storage_kind{BackingStoreKind::kHost};
  /// Camera or pipeline timestamp used to associate inference metadata.
  std::int64_t capture_time_ns{0};
  /// Local steady-clock time when this frame entered the processing pipeline.
  std::int64_t capture_steady_time_ns{0};
  std::string source_id;
  std::shared_ptr<void> owner;
  bool flip_180{false};

  /// Return frame width in pixels.
  int width() const
  {
    return frame.cols;
  }

  /// Return frame height in pixels.
  int height() const
  {
    return frame.rows;
  }

  /// Return true when no frame data is attached.
  bool empty() const
  {
    return frame.empty();
  }
};

}  // namespace yolo_cpp

#endif
