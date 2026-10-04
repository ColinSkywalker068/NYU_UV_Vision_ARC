#include "yolo_cpp/mjpeg_preview_server.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <mutex>
#include <netinet/in.h>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <stdexcept>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

#include <arpa/inet.h>
#include <signal.h>

namespace yolo_cpp {

namespace {

constexpr char kBoundary[] = "cv-detection-preview";

void ignore_sigpipe_once() {
  static std::once_flag once;
  std::call_once(once, []() { ::signal(SIGPIPE, SIG_IGN); });
}

std::string http_date_now() {
  std::time_t now = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&now, &tm);
  char buffer[64]{};
  std::strftime(buffer, sizeof(buffer), "%a, %d %b %Y %H:%M:%S GMT", &tm);
  return buffer;
}

std::string request_path(const std::string &request) {
  const auto method_end = request.find(' ');
  if (method_end == std::string::npos) {
    return "/";
  }
  const auto path_start = method_end + 1;
  const auto path_end = request.find(' ', path_start);
  if (path_end == std::string::npos || path_end <= path_start) {
    return "/";
  }
  auto path = request.substr(path_start, path_end - path_start);
  const auto query = path.find('?');
  if (query != std::string::npos) {
    path.resize(query);
  }
  return path.empty() ? "/" : path;
}

} // namespace

MjpegPreviewServer::~MjpegPreviewServer() { stop(); }

void MjpegPreviewServer::start(Config config) {
  if (running_.load()) {
    return;
  }
  ignore_sigpipe_once();
  config_ = std::move(config);

  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd_ < 0) {
    throw std::runtime_error("socket() failed: " +
                             std::string(std::strerror(errno)));
  }

  int yes = 1;
  ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(config_.port);
  if (config_.bind_address.empty() || config_.bind_address == "0.0.0.0") {
    address.sin_addr.s_addr = htonl(INADDR_ANY);
  } else if (::inet_pton(AF_INET, config_.bind_address.c_str(),
                         &address.sin_addr) != 1) {
    ::close(listen_fd_);
    listen_fd_ = -1;
    throw std::runtime_error("Invalid preview_http_bind_address: " +
                             config_.bind_address);
  }

  if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) < 0) {
    const std::string error = std::strerror(errno);
    ::close(listen_fd_);
    listen_fd_ = -1;
    throw std::runtime_error("bind() failed for MJPEG preview server: " +
                             error);
  }
  if (::listen(listen_fd_, 8) < 0) {
    const std::string error = std::strerror(errno);
    ::close(listen_fd_);
    listen_fd_ = -1;
    throw std::runtime_error("listen() failed for MJPEG preview server: " +
                             error);
  }

  running_.store(true);
  accept_thread_ = std::thread(&MjpegPreviewServer::accept_loop, this);
}

void MjpegPreviewServer::stop() {
  const bool was_running = running_.exchange(false);
  frame_cv_.notify_all();
  if (listen_fd_ >= 0) {
    ::shutdown(listen_fd_, SHUT_RDWR);
    ::close(listen_fd_);
    listen_fd_ = -1;
  }
  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }
  close_active_clients();
  {
    std::unique_lock<std::mutex> lock(clients_mutex_);
    clients_cv_.wait(lock, [this]() { return active_clients_.empty(); });
  }
  if (!was_running) {
    return;
  }
}

bool MjpegPreviewServer::update_frame(const cv::Mat &frame_bgr,
                                      const std::string &stats_json,
                                      int jpeg_quality) {
  if (frame_bgr.empty()) {
    return false;
  }
  std::vector<unsigned char> encoded;
  const std::vector<int> params{cv::IMWRITE_JPEG_QUALITY,
                                std::clamp(jpeg_quality, 1, 100)};
  if (!cv::imencode(".jpg", frame_bgr, encoded, params)) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    latest_jpeg_ = std::move(encoded);
    latest_stats_json_ = stats_json.empty() ? "{}" : stats_json;
    ++frame_sequence_;
  }
  frame_cv_.notify_all();
  return true;
}

void MjpegPreviewServer::update_controller_status(int state) {
  std::lock_guard<std::mutex> lock(frame_mutex_);
  controller_status_.update(state, controller_status_now_ns());
}

std::string MjpegPreviewServer::url() const {
  const std::string host =
      config_.bind_address.empty() || config_.bind_address == "0.0.0.0"
          ? "<robot-host>"
          : config_.bind_address;
  return "http://" + host + ":" + std::to_string(config_.port) + "/";
}

void MjpegPreviewServer::accept_loop() {
  while (running_.load()) {
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);
    const int client_fd = ::accept(
        listen_fd_, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
    if (client_fd < 0) {
      if (!running_.load()) {
        break;
      }
      continue;
    }
    {
      std::lock_guard<std::mutex> lock(clients_mutex_);
      active_clients_.push_back(client_fd);
    }
    std::thread(&MjpegPreviewServer::handle_client, this, client_fd).detach();
  }
}

void MjpegPreviewServer::handle_client(int client_fd) {
  char buffer[4096]{};
  const ssize_t received = ::recv(client_fd, buffer, sizeof(buffer) - 1, 0);
  if (received <= 0) {
    remove_client(client_fd);
    ::close(client_fd);
    return;
  }
  const std::string request(buffer, static_cast<std::size_t>(received));
  const std::string path = request_path(request);

  bool ok = false;
  if (path == "/" || path == "/index.html") {
    ok = send_index(client_fd);
  } else if (path == "/stream.mjpg") {
    ok = send_stream(client_fd);
  } else if (path == "/snapshot.jpg") {
    ok = send_snapshot(client_fd);
  } else if (path == "/stats.json") {
    ok = send_stats(client_fd);
  } else {
    const std::string body = "Not found\n";
    const std::string header = "HTTP/1.1 404 Not Found\r\n" +
                               content_length_header("text/plain", body.size());
    ok = send_all(client_fd, header.c_str(), header.size()) &&
         send_all(client_fd, body.c_str(), body.size());
  }

  (void)ok;
  remove_client(client_fd);
  ::shutdown(client_fd, SHUT_RDWR);
  ::close(client_fd);
}

void MjpegPreviewServer::remove_client(int client_fd) {
  std::lock_guard<std::mutex> lock(clients_mutex_);
  active_clients_.erase(
      std::remove(active_clients_.begin(), active_clients_.end(), client_fd),
      active_clients_.end());
  clients_cv_.notify_all();
}

void MjpegPreviewServer::close_active_clients() {
  std::vector<int> clients;
  {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    clients = active_clients_;
  }
  for (const int client_fd : clients) {
    ::shutdown(client_fd, SHUT_RDWR);
  }
}

bool MjpegPreviewServer::send_index(int client_fd) {
  const std::string body = index_html(config_.title);
  const std::string header =
      "HTTP/1.1 200 OK\r\n" +
      content_length_header("text/html; charset=utf-8", body.size());
  return send_all(client_fd, header.c_str(), header.size()) &&
         send_all(client_fd, body.c_str(), body.size());
}

bool MjpegPreviewServer::send_stats(int client_fd) {
  std::string body;
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    body = latest_stats_json_;
    // The detector supplies an object; reject invalid outer framing cheaply.
    if (body.size() < 2 || body.front() != '{' || body.back() != '}') {body = "{}";}
    body.pop_back();  // Append live status fields, even when no frame has arrived.
    if (body.size() > 1) {body += ",";}
    body += controller_status_.json_fields(controller_status_now_ns()) + "}";
  }
  const std::string header =
      "HTTP/1.1 200 OK\r\n" +
      content_length_header("application/json", body.size());
  return send_all(client_fd, header.c_str(), header.size()) &&
         send_all(client_fd, body.c_str(), body.size());
}

bool MjpegPreviewServer::send_snapshot(int client_fd) {
  std::vector<unsigned char> jpeg;
  {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    jpeg = latest_jpeg_;
  }
  if (jpeg.empty()) {
    const std::string body = "No preview frame has been rendered yet.\n";
    const std::string header = "HTTP/1.1 503 Service Unavailable\r\n" +
                               content_length_header("text/plain", body.size());
    return send_all(client_fd, header.c_str(), header.size()) &&
           send_all(client_fd, body.c_str(), body.size());
  }
  const std::string header =
      "HTTP/1.1 200 OK\r\n" + content_length_header("image/jpeg", jpeg.size());
  return send_all(client_fd, header.c_str(), header.size()) &&
         send_all(client_fd, reinterpret_cast<const char *>(jpeg.data()),
                  jpeg.size());
}

bool MjpegPreviewServer::send_stream(int client_fd) {
  const std::string header =
      "HTTP/1.1 200 OK\r\n"
      "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
      "Pragma: no-cache\r\n"
      "Connection: close\r\n"
      "Content-Type: multipart/x-mixed-replace; boundary=" +
      std::string(kBoundary) + "\r\n\r\n";
  if (!send_all(client_fd, header.c_str(), header.size())) {
    return false;
  }

  std::uint64_t last_sequence = 0;
  while (running_.load()) {
    std::vector<unsigned char> jpeg;
    std::uint64_t sequence = 0;
    {
      std::unique_lock<std::mutex> lock(frame_mutex_);
      frame_cv_.wait(lock, [this, last_sequence]() {
        return !running_.load() || frame_sequence_ > last_sequence;
      });
      if (!running_.load()) {
        return false;
      }
      jpeg = latest_jpeg_;
      sequence = frame_sequence_;
    }
    if (jpeg.empty()) {
      continue;
    }
    last_sequence = sequence;
    std::ostringstream part;
    part << "--" << kBoundary << "\r\n"
         << "Content-Type: image/jpeg\r\n"
         << "Content-Length: " << jpeg.size() << "\r\n"
         << "Date: " << http_date_now() << "\r\n\r\n";
    const std::string part_header = part.str();
    if (!send_all(client_fd, part_header.c_str(), part_header.size()) ||
        !send_all(client_fd, reinterpret_cast<const char *>(jpeg.data()),
                  jpeg.size()) ||
        !send_all(client_fd, "\r\n", 2)) {
      return false;
    }
  }
  return true;
}

bool MjpegPreviewServer::send_all(int fd, const char *data, std::size_t size) {
  std::size_t sent_total = 0;
  while (sent_total < size) {
#ifdef MSG_NOSIGNAL
    const int flags = MSG_NOSIGNAL;
#else
    const int flags = 0;
#endif
    const ssize_t sent =
        ::send(fd, data + sent_total, size - sent_total, flags);
    if (sent <= 0) {
      return false;
    }
    sent_total += static_cast<std::size_t>(sent);
  }
  return true;
}

std::string
MjpegPreviewServer::content_length_header(const std::string &content_type,
                                          std::size_t size) {
  std::ostringstream out;
  out << "Content-Type: " << content_type << "\r\n"
      << "Content-Length: " << size << "\r\n"
      << "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n"
      << "Connection: close\r\n\r\n";
  return out.str();
}

std::string MjpegPreviewServer::index_html(const std::string &title) {
  std::ostringstream html;
  html
      << "<!doctype html><html><head><meta charset=\"utf-8\">"
      << "<meta name=\"viewport\" "
         "content=\"width=device-width,initial-scale=1\">"
      << "<title>" << title << "</title>"
      << "<style>"
      << "body{margin:0;background:#111;color:#f4f4f4;font-family:ui-sans-"
         "serif,system-ui,sans-serif;}"
      << "main{display:grid;grid-template-columns:minmax(0,1fr) "
         "280px;min-height:100vh;}"
      << ".frame{display:grid;place-items:center;background:#050505;overflow:"
         "hidden;}"
      << "img{display:block;max-width:100%;max-height:100vh;object-fit:contain;"
         "}"
      << "aside{padding:16px;background:#1d1d1d;border-left:1px solid #333;}"
      << "h1{font-size:18px;margin:0 0 "
         "16px;}dl{display:grid;grid-template-columns:1fr auto;gap:8px 12px;}"
      << "dt{color:#aaa;}dd{margin:0;font-variant-numeric:tabular-nums;}code{"
         "color:#b7f7c7;}"
      << "@media(max-width:900px){main{grid-template-columns:1fr;}aside{border-"
         "left:0;border-top:1px solid #333;}}"
      << "</style></head><body><main><section class=\"frame\">"
      << "<img src=\"/stream.mjpg\" alt=\"CV preview stream\"></section><aside>"
      << "<h1>" << title << "</h1><dl id=\"stats\"></dl>"
      << "<p><code>/stream.mjpg</code><br><code>/snapshot.jpg</code><br><code>/"
         "stats.json</code></p>"
      << "</aside></main><script>"
      << "const stats=document.getElementById('stats');"
      << "function row(k,v){return `<dt>${k}</dt><dd>${v}</dd>`;}"
      << "async function tick(){try{const r=await "
         "fetch('/stats.json',{cache:'no-store'});"
      << "const s=await r.json();stats.innerHTML="
      << "row('fps',(s.fps??0).toFixed?.(1)??s.fps)+row('backend',s.backend??''"
         ")+"
      << "row('camera',s.camera??'')+row('frame',`${s.width??0}x${s.height??0}`"
         ")+"
      << "row('candidates',s.candidates??0)+row('enemy',s.enemy_candidates??0)+"
      << "row('target',s.has_target?`#${s.target_track_id}`:'none')+"
      << "row('auto aim (MCU)',(s.auto_aim_status??'unknown').toUpperCase())+"
      << "row('TX configuration',s.transmit_enabled===true?'ENABLED':(s.transmit_enabled===false?'DISABLED':'UNKNOWN'))+"
      << "row('latency ms',(s.latency_ms??0).toFixed?.(1)??s.latency_ms);}"
      << "catch(e){stats.innerHTML=row('status','waiting');}}"
      << "tick();setInterval(tick,500);</script></body></html>";
  return html.str();
}

} // namespace yolo_cpp
