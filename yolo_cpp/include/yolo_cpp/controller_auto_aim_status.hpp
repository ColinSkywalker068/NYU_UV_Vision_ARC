#ifndef YOLO_CPP__CONTROLLER_AUTO_AIM_STATUS_HPP_
#define YOLO_CPP__CONTROLLER_AUTO_AIM_STATUS_HPP_

#include <chrono>
#include <cstdint>
#include <string>

namespace yolo_cpp
{
// Only ST robot_status advertises this contract. SP mode is never acceptance.
inline int decode_controller_auto_aim(std::uint8_t robot_status)
{
  return (robot_status & 0x80U) ? ((robot_status & 0x08U) ? 1 : 0) : -1;
}

inline std::int64_t controller_status_now_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

// Caller owns synchronization. Receipt time is monotonic, not ROS/wall time.
class ControllerAutoAimStatus
{
public:
  void update(int state, std::int64_t received_ns)
  {
    state_ = state == 0 || state == 1 ? state : -1;
    received_ns_ = received_ns;
  }
  int value(std::int64_t now_ns) const
  {
    return received_ns_ > 0 && now_ns >= received_ns_ && now_ns - received_ns_ <= 500000000 ? state_
                                                                                            : -1;
  }
  const char * name(std::int64_t now_ns) const
  {
    const int state = value(now_ns);
    return state == 1 ? "on" : (state == 0 ? "off" : "unknown");
  }
  double age_ms(std::int64_t now_ns) const
  {
    return received_ns_ > 0 && now_ns >= received_ns_ ? (now_ns - received_ns_) * 1e-6 : -1.0;
  }
  std::string json_fields(std::int64_t now_ns) const
  {
    const auto age = age_ms(now_ns);
    return std::string("\"auto_aim_status\":\"") + name(now_ns) +
           "\",\"auto_aim_status_age_ms\":" + (age < 0 ? "null" : std::to_string(age));
  }

private:
  int state_{-1};
  std::int64_t received_ns_{0};
};
}  // namespace yolo_cpp
#endif
