/// @file model_metadata.cpp
/// @brief Reads Ultralytics/TensorRT metadata embedded in serialized engine files.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "yolo_cpp/detection_postprocess.hpp"

namespace yolo_cpp
{

namespace
{

/// Read an engine/model file as binary data.
std::string read_file(const std::string& path)
{
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Failed to open model file: " + path);
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  return contents.str();
}

/// Extract Ultralytics-style JSON metadata prefix from an engine file.
std::optional<std::string> extract_engine_metadata(const std::string& data)
{
  if (data.size() < sizeof(std::uint32_t)) {
    return std::nullopt;
  }

  std::uint32_t metadata_len = 0;
  std::memcpy(&metadata_len, data.data(), sizeof(metadata_len));
  const std::size_t json_offset = sizeof(metadata_len);
  if (metadata_len == 0U || json_offset + metadata_len > data.size()) {
    return std::nullopt;
  }

  const std::string metadata = data.substr(json_offset, metadata_len);
  if (metadata.empty() || metadata.front() != '{' || metadata.back() != '}') {
    return std::nullopt;
  }
  return metadata;
}

/// Return the offset where serialized TensorRT plan bytes begin after metadata.
std::size_t engine_plan_offset(const std::string& data)
{
  if (data.size() < sizeof(std::uint32_t)) {
    return 0U;
  }

  std::uint32_t metadata_len = 0;
  std::memcpy(&metadata_len, data.data(), sizeof(metadata_len));
  const std::size_t json_offset = sizeof(metadata_len);
  const std::size_t plan_offset = json_offset + metadata_len;
  if (metadata_len == 0U || plan_offset > data.size()) {
    return 0U;
  }

  const std::string metadata = data.substr(json_offset, metadata_len);
  if (metadata.empty() || metadata.front() != '{' || metadata.back() != '}') {
    return 0U;
  }
  return plan_offset;
}

/// Extract a simple JSON string value without pulling in a JSON dependency.
std::optional<std::string> extract_json_string_value(
  const std::string& json, const std::string& key)
{
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }

  const auto colon = json.find(':', key_pos + needle.size());
  const auto first_quote = json.find('"', colon + 1);
  if (colon == std::string::npos || first_quote == std::string::npos) {
    return std::nullopt;
  }

  const auto second_quote = json.find('"', first_quote + 1);
  if (second_quote == std::string::npos) {
    return std::nullopt;
  }
  return json.substr(first_quote + 1, second_quote - first_quote - 1);
}

/// Parse the model class-name map from lightweight metadata JSON.
std::unordered_map<int, std::string> extract_names(const std::string& json)
{
  std::unordered_map<int, std::string> names;
  const auto names_key = json.find("\"names\"");
  if (names_key == std::string::npos) {
    return names;
  }

  const auto block_begin = json.find('{', names_key);
  if (block_begin == std::string::npos) {
    return names;
  }

  int depth = 0;
  bool in_string = false;
  bool escaping = false;
  std::size_t block_end = std::string::npos;
  for (std::size_t i = block_begin; i < json.size(); ++i) {
    const char c = json[i];
    if (escaping) {
      escaping = false;
      continue;
    }
    if (c == '\\') {
      escaping = true;
      continue;
    }
    if (c == '"') {
      in_string = !in_string;
      continue;
    }
    if (in_string) {
      continue;
    }
    if (c == '{') {
      ++depth;
    } else if (c == '}') {
      --depth;
      if (depth == 0) {
        block_end = i;
        break;
      }
    }
  }

  if (block_end == std::string::npos) {
    return names;
  }

  const std::string block = json.substr(block_begin + 1, block_end - block_begin - 1);
  std::size_t pos = 0;
  while (true) {
    const auto key_q1 = block.find('"', pos);
    if (key_q1 == std::string::npos) {
      break;
    }
    const auto key_q2 = block.find('"', key_q1 + 1);
    const auto val_q1 = block.find('"', key_q2 + 1);
    const auto val_q2 = block.find('"', val_q1 + 1);
    if (key_q2 == std::string::npos || val_q1 == std::string::npos || val_q2 == std::string::npos) {
      break;
    }

    names[std::stoi(block.substr(key_q1 + 1, key_q2 - key_q1 - 1))] =
      block.substr(val_q1 + 1, val_q2 - val_q1 - 1);
    pos = val_q2 + 1;
  }
  return names;
}

/// Extract a simple JSON bool value without pulling in a JSON dependency.
std::optional<bool> extract_json_bool_value(const std::string& json, const std::string& key)
{
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }

  const auto colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }

  const auto value_pos = json.find_first_not_of(" \t\r\n", colon + 1);
  if (value_pos == std::string::npos) {
    return std::nullopt;
  }
  if (json.compare(value_pos, 4, "true") == 0) {
    return true;
  }
  if (json.compare(value_pos, 5, "false") == 0) {
    return false;
  }
  return std::nullopt;
}

}  // namespace

EngineMetadata read_engine_metadata_from_file(const std::string& model_path)
{
  EngineMetadata metadata{};
  const auto file_data = read_file(model_path);
  const auto json = extract_engine_metadata(file_data);
  if (!json.has_value()) {
    return metadata;
  }

  metadata.class_names = extract_names(*json);
  metadata.task = extract_json_string_value(*json, "task");
  metadata.version = extract_json_string_value(*json, "version");
  metadata.end2end = extract_json_bool_value(*json, "end2end").value_or(false);
  metadata.int8_export = extract_json_bool_value(*json, "int8").value_or(false);
  metadata.export_includes_nms = extract_json_bool_value(*json, "nms");
  return metadata;
}

std::string read_engine_plan_from_file(const std::string& model_path)
{
  const auto file_data = read_file(model_path);
  const std::size_t plan_offset = engine_plan_offset(file_data);
  if (plan_offset == 0U) {
    return file_data;
  }
  return file_data.substr(plan_offset);
}

}  // namespace yolo_cpp
