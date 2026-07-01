#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "yolo_cpp/tracker.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

struct DetectionInput {
  float center_x{0.0F};
  float center_y{0.0F};
  float width{40.0F};
  float height{20.0F};
  float confidence{0.90F};
};

struct FrameInput {
  double dt{1.0 / 60.0};
  std::vector<DetectionInput> detections;
};

struct Options {
  std::string scenario{"dropout_hold"};
  std::string input_path;
  int image_width{640};
  int image_height{480};
  double hfov_degrees{79.2};
  double vfov_degrees{63.78};
  double angle_gain{1.0};
  bool trace_control{false};
  bool publish_zero_angles_when_no_target{true};
  yolo_cpp::TrackConfig config;
};

std::string track_state_name(yolo_cpp::TrackState state) {
  switch (state) {
  case yolo_cpp::TrackState::kDetecting:
    return "detecting";
  case yolo_cpp::TrackState::kTracking:
    return "tracking";
  case yolo_cpp::TrackState::kCoasting:
    return "coasting";
  }
  return "unknown";
}

double degrees_to_radians(double degrees) { return degrees * kPi / 180.0; }

std::pair<double, double> calculate_angles(const cv::Rect &bbox,
                                           int image_width, int image_height,
                                           double hfov_rad, double vfov_rad) {
  const double center_x =
      static_cast<double>(bbox.x) + (static_cast<double>(bbox.width) / 2.0);
  const double center_y =
      static_cast<double>(bbox.y) + (static_cast<double>(bbox.height) / 2.0);
  const double pixel_offset_x =
      (center_x - (static_cast<double>(image_width) / 2.0)) /
      (static_cast<double>(image_width) / 2.0);
  const double pixel_offset_y =
      (center_y - (static_cast<double>(image_height) / 2.0)) /
      (static_cast<double>(image_height) / 2.0);
  return {-std::atan(pixel_offset_x * std::tan(hfov_rad / 2.0)),
          std::atan(pixel_offset_y * std::tan(vfov_rad / 2.0))};
}

std::string sign_name(double value) {
  if (value > 1e-6) {
    return "+";
  }
  if (value < -1e-6) {
    return "-";
  }
  return "0";
}

std::string bool_name(bool value) { return value ? "true" : "false"; }

bool parse_bool(const std::string &text) {
  if (text == "true" || text == "1" || text == "yes" || text == "on") {
    return true;
  }
  if (text == "false" || text == "0" || text == "no" || text == "off") {
    return false;
  }
  throw std::runtime_error("expected boolean value, got: " + text);
}

yolo_cpp::TrackObservation make_observation(const DetectionInput &input,
                                            int image_width, int image_height) {
  yolo_cpp::TrackObservation observation;
  observation.center = cv::Point2f(input.center_x, input.center_y);
  observation.bbox = cv::Rect(
      static_cast<int>(std::round(input.center_x - (input.width / 2.0F))),
      static_cast<int>(std::round(input.center_y - (input.height / 2.0F))),
      std::max(1, static_cast<int>(std::round(input.width))),
      std::max(1, static_cast<int>(std::round(input.height))));
  observation.confidence = input.confidence;
  observation.class_id = 0;
  observation.is_enemy = true;
  observation.image_width = image_width;
  observation.image_height = image_height;
  observation.distance_to_image_center = static_cast<float>(
      std::hypot(input.center_x - (static_cast<float>(image_width) / 2.0F),
                 input.center_y - (static_cast<float>(image_height) / 2.0F)));
  return observation;
}

std::vector<FrameInput> dropout_hold_scenario() {
  return {
      {1.0 / 60.0, {{190.0F, 365.0F}}},
      {1.0 / 60.0, {{198.0F, 358.0F}}},
      {1.0 / 60.0, {}},
      {1.0 / 60.0, {{210.0F, 350.0F}}},
      {1.0 / 60.0, {}},
      {1.0 / 60.0, {{225.0F, 338.0F}}},
      {1.0 / 60.0, {{245.0F, 325.0F}}},
      {1.0 / 60.0, {{270.0F, 305.0F}}},
      {1.0 / 60.0, {{300.0F, 275.0F}}},
      {1.0 / 60.0, {{318.0F, 248.0F}}},
  };
}

std::vector<FrameInput> long_dropout_scenario() {
  return {
      {1.0 / 60.0, {{190.0F, 365.0F}}},
      {1.0 / 60.0, {{198.0F, 358.0F}}},
      {1.0 / 60.0, {}},
      {1.0 / 60.0, {}},
      {1.0 / 60.0, {}},
      {1.0 / 60.0, {{210.0F, 350.0F}}},
      {1.0 / 60.0, {{225.0F, 338.0F}}},
  };
}

std::vector<FrameInput> corners_scenario() {
  return {
      {1.0 / 60.0, {{160.0F, 120.0F}}},
      {1.0 / 60.0, {{480.0F, 120.0F}}},
      {1.0 / 60.0, {{160.0F, 360.0F}}},
      {1.0 / 60.0, {{480.0F, 360.0F}}},
  };
}

std::optional<double> parse_double(const std::string &text) {
  char *end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (end == text.c_str() || *end != '\0') {
    return std::nullopt;
  }
  return value;
}

std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> parts;
  std::stringstream stream(line);
  std::string part;
  while (std::getline(stream, part, ',')) {
    part.erase(part.begin(),
               std::find_if(part.begin(), part.end(), [](unsigned char ch) {
                 return !std::isspace(ch);
               }));
    part.erase(std::find_if(part.rbegin(), part.rend(),
                            [](unsigned char ch) { return !std::isspace(ch); })
                   .base(),
               part.end());
    parts.push_back(part);
  }
  return parts;
}

std::vector<FrameInput> load_csv_frames(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("could not open input CSV: " + path);
  }

  std::vector<FrameInput> frames;
  std::string line;
  int previous_frame = -1;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto parts = split_csv_line(line);
    if (!parts.empty() && parts.front() == "frame") {
      continue;
    }
    if (parts.size() != 6U) {
      throw std::runtime_error(
          "CSV rows must be: frame,dt,center_x,center_y,width,height; got: " +
          line);
    }
    const auto frame_value = parse_double(parts[0]);
    const auto dt = parse_double(parts[1]);
    if (!frame_value.has_value() || !dt.has_value()) {
      throw std::runtime_error("invalid frame or dt in CSV row: " + line);
    }
    const int frame_index = static_cast<int>(*frame_value);
    if (frame_index != previous_frame) {
      frames.push_back(FrameInput{*dt, {}});
      previous_frame = frame_index;
    }
    if (parts[2].empty() || parts[2] == "none") {
      continue;
    }
    const auto center_x = parse_double(parts[2]);
    const auto center_y = parse_double(parts[3]);
    const auto width = parse_double(parts[4]);
    const auto height = parse_double(parts[5]);
    if (!center_x.has_value() || !center_y.has_value() || !width.has_value() ||
        !height.has_value()) {
      throw std::runtime_error("invalid detection values in CSV row: " + line);
    }
    frames.back().detections.push_back(DetectionInput{
        static_cast<float>(*center_x), static_cast<float>(*center_y),
        static_cast<float>(*width), static_cast<float>(*height), 0.90F});
  }
  return frames;
}

void print_usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n\n"
            << "Options:\n"
            << "  --scenario dropout_hold|long_dropout|corners|csv\n"
            << "  --input PATH                  CSV input for --scenario csv\n"
            << "  --image-width N               Default: 640\n"
            << "  --image-height N              Default: 480\n"
            << "  --angle-gain FLOAT            Default: 1.0\n"
            << "  --trace-control               Print "
               "tracker->processor->solver->bridge columns\n"
            << "  --publish-zero-angles-when-no-target BOOL\n"
            << "  --track-max-coast-frames N    Default: 2 for this harness\n"
            << "  --track-drop-frames N         Default: 4\n"
            << "  --association-ratio FLOAT     Default: 0.30\n"
            << "  --help\n\n"
            << "CSV format: frame,dt,center_x,center_y,width,height. Use "
               "center_x=none for a dropout.\n";
}

Options parse_options(int argc, char **argv) {
  Options options;
  options.config.track_min_hits = 1;
  options.config.track_max_coast_frames = 2;
  options.config.track_drop_frames = 4;
  options.config.association_max_center_distance_ratio = 0.30;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };
    if (arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--scenario") {
      options.scenario = require_value(arg);
    } else if (arg == "--input") {
      options.input_path = require_value(arg);
    } else if (arg == "--image-width") {
      options.image_width = std::stoi(require_value(arg));
    } else if (arg == "--image-height") {
      options.image_height = std::stoi(require_value(arg));
    } else if (arg == "--angle-gain") {
      options.angle_gain = std::stod(require_value(arg));
    } else if (arg == "--trace-control") {
      options.trace_control = true;
    } else if (arg == "--publish-zero-angles-when-no-target") {
      options.publish_zero_angles_when_no_target =
          parse_bool(require_value(arg));
    } else if (arg == "--track-max-coast-frames") {
      options.config.track_max_coast_frames = std::stoi(require_value(arg));
    } else if (arg == "--track-drop-frames") {
      options.config.track_drop_frames = std::stoi(require_value(arg));
    } else if (arg == "--association-ratio") {
      options.config.association_max_center_distance_ratio =
          std::stod(require_value(arg));
    } else {
      throw std::runtime_error("unknown option: " + arg);
    }
  }
  return options;
}

std::vector<FrameInput> frames_for_options(const Options &options) {
  if (options.scenario == "dropout_hold") {
    return dropout_hold_scenario();
  }
  if (options.scenario == "long_dropout") {
    return long_dropout_scenario();
  }
  if (options.scenario == "corners") {
    return corners_scenario();
  }
  if (options.scenario == "csv") {
    if (options.input_path.empty()) {
      throw std::runtime_error("--scenario csv requires --input PATH");
    }
    return load_csv_frames(options.input_path);
  }
  throw std::runtime_error("unknown scenario: " + options.scenario);
}

bool validate_expected_sign(const yolo_cpp::TrackSnapshot &target, double yaw,
                            double pitch, std::ostream &error_stream) {
  const double image_center_x =
      static_cast<double>(target.observation.image_width) / 2.0;
  const double image_center_y =
      static_cast<double>(target.observation.image_height) / 2.0;
  const double target_x = static_cast<double>(target.observation.center.x);
  const double target_y = static_cast<double>(target.observation.center.y);
  bool ok = true;

  if (target_x < image_center_x - 1.0 && yaw <= 0.0) {
    error_stream << "expected positive yaw for left-of-center target, got "
                 << yaw << "\n";
    ok = false;
  }
  if (target_x > image_center_x + 1.0 && yaw >= 0.0) {
    error_stream << "expected negative yaw for right-of-center target, got "
                 << yaw << "\n";
    ok = false;
  }
  if (target_y < image_center_y - 1.0 && pitch >= 0.0) {
    error_stream << "expected negative pitch for above-center target, got "
                 << pitch << "\n";
    ok = false;
  }
  if (target_y > image_center_y + 1.0 && pitch <= 0.0) {
    error_stream << "expected positive pitch for below-center target, got "
                 << pitch << "\n";
    ok = false;
  }
  return ok;
}

int run_control_trace(const Options &options) {
  const auto frames = frames_for_options(options);
  yolo_cpp::TrackManager manager(options.config);
  const double hfov_rad = degrees_to_radians(options.hfov_degrees);
  const double vfov_rad = degrees_to_radians(options.vfov_degrees);
  const double effective_gain = std::max(1.0, options.angle_gain);
  double bridge_latest_yaw = 0.0;
  double bridge_latest_pitch = 0.0;
  bool bridge_has_target = false;
  bool ok = true;

  std::cout
      << "frame,detections,tracker_selected,tracker_track_id,tracker_state,"
         "tracker_lost,tracker_center_x,tracker_center_y,processor_has_target,"
         "processor_is_coasting,processor_has_pose,solver_input,solver_has_"
         "solution,"
         "solver_angle_published,solver_yaw,solver_pitch,override,bridge_has_"
         "target,"
         "bridge_mode,bridge_latest_yaw,bridge_latest_pitch\n";

  for (std::size_t frame_index = 0; frame_index < frames.size();
       ++frame_index) {
    std::vector<yolo_cpp::TrackObservation> observations;
    observations.reserve(frames[frame_index].detections.size());
    for (const auto &detection : frames[frame_index].detections) {
      observations.push_back(make_observation(detection, options.image_width,
                                              options.image_height));
    }

    const auto result = manager.update(observations, frames[frame_index].dt);
    const bool tracker_selected = result.selected_target.has_value();
    const yolo_cpp::TrackSnapshot *target =
        tracker_selected ? &(*result.selected_target) : nullptr;
    const bool processor_has_target =
        target != nullptr &&
        (target->state == yolo_cpp::TrackState::kTracking ||
         target->state == yolo_cpp::TrackState::kCoasting);
    const bool processor_is_coasting =
        target != nullptr && target->state == yolo_cpp::TrackState::kCoasting;
    const bool processor_has_pose =
        processor_has_target && target != nullptr && target->has_pose;

    bool solver_has_solution = processor_has_target;
    bool solver_angle_published = false;
    double solver_yaw = 0.0;
    double solver_pitch = 0.0;
    std::string solver_input = "none";
    if (processor_has_target && target != nullptr) {
      solver_input = processor_has_pose ? "3d" : "2d_bbox";
      const auto [yaw_raw, pitch_raw] =
          calculate_angles(target->observation.bbox, options.image_width,
                           options.image_height, hfov_rad, vfov_rad);
      solver_yaw = yaw_raw * effective_gain;
      solver_pitch = pitch_raw * effective_gain;
      solver_angle_published = true;
      ok = validate_expected_sign(*target, solver_yaw, solver_pitch,
                                  std::cerr) &&
           ok;
    } else if (options.publish_zero_angles_when_no_target) {
      solver_angle_published = true;
      solver_has_solution = false;
      solver_input = "no_target_zero";
      solver_yaw = 0.0;
      solver_pitch = 0.0;
    }

    const bool override = processor_has_target;
    if (solver_angle_published) {
      bridge_latest_yaw = solver_yaw;
      bridge_latest_pitch = solver_pitch;
    }
    bridge_has_target = override;
    const std::string bridge_mode = bridge_has_target ? "active" : "hold";

    std::cout << frame_index << "," << observations.size() << ","
              << bool_name(tracker_selected) << ","
              << (target != nullptr ? target->track_id : -1) << ","
              << (target != nullptr ? track_state_name(target->state) : "none")
              << "," << (target != nullptr ? target->lost_frames : 0) << ","
              << std::fixed << std::setprecision(3)
              << (target != nullptr ? target->observation.center.x : 0.0F)
              << ","
              << (target != nullptr ? target->observation.center.y : 0.0F)
              << "," << bool_name(processor_has_target) << ","
              << bool_name(processor_is_coasting) << ","
              << bool_name(processor_has_pose) << "," << solver_input << ","
              << bool_name(solver_has_solution) << ","
              << bool_name(solver_angle_published) << ","
              << std::setprecision(6) << solver_yaw << "," << solver_pitch
              << "," << bool_name(override) << ","
              << bool_name(bridge_has_target) << "," << bridge_mode << ","
              << bridge_latest_yaw << "," << bridge_latest_pitch << "\n";
  }
  return ok ? 0 : 2;
}

int run_replay(const Options &options) {
  if (options.trace_control) {
    return run_control_trace(options);
  }

  const auto frames = frames_for_options(options);
  yolo_cpp::TrackManager manager(options.config);
  const double hfov_rad = degrees_to_radians(options.hfov_degrees);
  const double vfov_rad = degrees_to_radians(options.vfov_degrees);
  const double effective_gain = std::max(1.0, options.angle_gain);
  bool ok = true;

  std::cout << "frame,dt,detections,selected,track_id,state,lost,center_x,"
               "center_y,pred_x,pred_y,"
               "effective_gain,yaw_rad,pitch_rad,yaw_deg,pitch_deg,yaw_sign,"
               "pitch_sign\n";
  for (std::size_t frame_index = 0; frame_index < frames.size();
       ++frame_index) {
    std::vector<yolo_cpp::TrackObservation> observations;
    observations.reserve(frames[frame_index].detections.size());
    for (const auto &detection : frames[frame_index].detections) {
      observations.push_back(make_observation(detection, options.image_width,
                                              options.image_height));
    }

    const auto result = manager.update(observations, frames[frame_index].dt);
    if (!result.selected_target.has_value()) {
      std::cout << frame_index << "," << frames[frame_index].dt << ","
                << observations.size() << ",false,-1,none,0,,,,,"
                << effective_gain << ",0,0,0,0,0,0\n";
      continue;
    }

    const auto &target = *result.selected_target;
    const auto [yaw_raw, pitch_raw] =
        calculate_angles(target.observation.bbox, options.image_width,
                         options.image_height, hfov_rad, vfov_rad);
    const double yaw = yaw_raw * effective_gain;
    const double pitch = pitch_raw * effective_gain;
    ok = validate_expected_sign(target, yaw, pitch, std::cerr) && ok;

    std::cout << frame_index << "," << frames[frame_index].dt << ","
              << observations.size() << ",true," << target.track_id << ","
              << track_state_name(target.state) << "," << target.lost_frames
              << "," << std::fixed << std::setprecision(3)
              << target.observation.center.x << ","
              << target.observation.center.y << "," << target.predicted_center.x
              << "," << target.predicted_center.y << "," << std::setprecision(3)
              << effective_gain << "," << std::setprecision(6) << yaw << ","
              << pitch << "," << (yaw * 180.0 / kPi) << ","
              << (pitch * 180.0 / kPi) << "," << sign_name(yaw) << ","
              << sign_name(pitch) << "\n";
  }

  if (options.scenario == "dropout_hold") {
    const bool saw_dropout =
        std::any_of(frames.begin(), frames.end(), [](const FrameInput &frame) {
          return frame.detections.empty();
        });
    if (!saw_dropout) {
      std::cerr << "dropout_hold scenario did not contain a dropout\n";
      ok = false;
    }
  }
  return ok ? 0 : 2;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run_replay(parse_options(argc, argv));
  } catch (const std::exception &error) {
    std::cerr << "tracker_replay: " << error.what() << "\n";
    return 1;
  }
}
