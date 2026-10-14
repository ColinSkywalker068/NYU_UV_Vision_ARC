#ifndef YOLO_CPP__SERIAL_PORT_SELECTION_HPP_
#define YOLO_CPP__SERIAL_PORT_SELECTION_HPP_

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace yolo_cpp
{
// Match the controller's udev USB identity, never a generic ACM device or
// ST-LINK debug VCP. The serial number may change when the board is replaced.
inline bool is_vision_comm_name(const std::string & name)
{
  const std::string prefix = "usb-NYUSH_Vision_Comm_port_";
  const std::string suffix = "-if00";
  if (name.size() <= prefix.size() + suffix.size() ||
    name.compare(0, prefix.size(), prefix) != 0 ||
    name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
  {
    return false;
  }
  return std::all_of(
    name.begin() + prefix.size(), name.end() - suffix.size(), [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    });
}

struct SerialPortSelection
{
  std::string path;
  std::string error;
};

inline SerialPortSelection select_serial_port(
  const std::string & requested, const std::filesystem::path & by_id_directory)
{
  std::error_code error;
  if (requested != "auto") {
    // An explicit available path always wins. Only stale Vision Comm by-id
    // paths may fall back; missing custom/UART paths remain explicit failures.
    const bool present = std::filesystem::exists(requested, error);
    if (present || error ||
      std::filesystem::path(requested).parent_path() != by_id_directory ||
      !is_vision_comm_name(std::filesystem::path(requested).filename().string()))
    {
      return {requested, {}};
    }
  }
  std::vector<std::string> candidates;
  std::filesystem::directory_iterator entry(by_id_directory, error), end;
  while (!error && entry != end) {
    std::error_code candidate_error;
    if (is_vision_comm_name(entry->path().filename().string()) &&
      entry->is_symlink(candidate_error) && !candidate_error &&
      entry->is_character_file(candidate_error) && !candidate_error)
    {
      candidates.push_back(entry->path().string());
    }
    entry.increment(error);
  }
  if (error) {
    return {{}, "cannot scan Vision Comm by-id directory " + by_id_directory.string() +
      ": " + error.message()};
  }
  if (candidates.empty()) {
    return {{}, "no eligible NYUSH Vision Comm controller in " + by_id_directory.string()};
  }
  if (candidates.size() != 1) {
    std::sort(candidates.begin(), candidates.end());
    std::string detail = "ambiguous Vision Comm controllers; set an explicit serial_port:";
    for (const auto & candidate : candidates) {
      detail += " " + candidate;
    }
    return {{}, detail};
  }
  return {candidates.front(), {}};
}
}  // namespace yolo_cpp

#endif  // YOLO_CPP__SERIAL_PORT_SELECTION_HPP_
