#ifndef YOLO_CPP__MJPEG_PREVIEW_SERVER_HPP_
#define YOLO_CPP__MJPEG_PREVIEW_SERVER_HPP_

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "yolo_cpp/visibility_control.hpp"
#include "yolo_cpp/controller_auto_aim_status.hpp"

namespace yolo_cpp {

/// Tiny HTTP server that exposes the detector preview as MJPEG plus JSON stats.
class YOLO_CPP_PUBLIC MjpegPreviewServer {
public:
  struct Config {
    std::string bind_address{"0.0.0.0"};
    std::uint16_t port{8080};
    std::string title{"CV Detection Preview"};
  };

  MjpegPreviewServer() = default;
  ~MjpegPreviewServer();

  MjpegPreviewServer(const MjpegPreviewServer &) = delete;
  MjpegPreviewServer &operator=(const MjpegPreviewServer &) = delete;

  /// Start listening for HTTP clients. Throws std::runtime_error on bind/listen
  /// errors.
  void start(Config config);

  /// Stop accepting clients and close active connections.
  void stop();

  /// Encode and publish one JPEG frame for future HTTP clients.
  bool update_frame(const cv::Mat &frame_bgr, const std::string &stats_json,
                    int jpeg_quality);

  /// Update independently of camera frames so HTTP status also expires when video stops.
  void update_controller_status(int state);

  /// Return the browser URL operators can open from the same robot network.
  std::string url() const;

private:
  void accept_loop();
  void handle_client(int client_fd);
  void remove_client(int client_fd);
  void close_active_clients();

  bool send_index(int client_fd);
  bool send_stats(int client_fd);
  bool send_snapshot(int client_fd);
  bool send_stream(int client_fd);

  static bool send_all(int fd, const char *data, std::size_t size);
  static std::string content_length_header(const std::string &content_type,
                                           std::size_t size);
  static std::string index_html(const std::string &title);

  Config config_{};
  std::atomic<bool> running_{false};
  int listen_fd_{-1};
  std::thread accept_thread_;

  mutable std::mutex frame_mutex_;
  std::condition_variable frame_cv_;
  std::vector<unsigned char> latest_jpeg_;
  std::string latest_stats_json_{"{}"};
  ControllerAutoAimStatus controller_status_;
  std::uint64_t frame_sequence_{0};

  std::mutex clients_mutex_;
  std::condition_variable clients_cv_;
  std::vector<int> active_clients_;
};

} // namespace yolo_cpp

#endif
