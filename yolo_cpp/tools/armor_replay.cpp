// Offline ARUW tensors -> production decoder and image-space tracker.
// No ROS nodes, serial port, 3D pose inference or controller output is started.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "yolo_cpp/detection_postprocess.hpp"
#include "yolo_cpp/tracker.hpp"
#include "yolo_cpp/utils.hpp"

int main(int argc, char ** argv)
{
  using namespace yolo_cpp;
  if (argc != 4) {
    std::cerr << "Usage: armor_replay tensors.txt tracks.jsonl confidence\n";
    return 2;
  }
  try {
    const float confidence = std::stof(argv[3]);
    if (!std::isfinite(confidence) || confidence <= 0 || confidence >= 1) {
      throw std::runtime_error("Confidence must be in (0, 1)");
    }
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    if (!input || !output) {
      throw std::runtime_error("Cannot open replay files");
    }
    output << std::setprecision(9);
    TrackManager
      tracker;  // Exact production defaults; both colors selectable for offline inspection.
    int frame, width, height, count;
    double timestamp, previous_time = -1;
    while (input >> frame >> timestamp >> width >> height >> count) {
      if (
        frame < 0 || !std::isfinite(timestamp) || timestamp <= previous_time || width < 1 ||
        width > 16384 || height < 1 || height > 16384 || count < 0 || count > 300) {
        throw std::runtime_error("Invalid replay frame header");
      }
      std::vector<float> raw(static_cast<std::size_t>(count) * 14);
      for (auto & value : raw) {
        if (!(input >> value)) {
          throw std::runtime_error("Incomplete tensor row");
        }
      }
      const auto detections =
        parse_aruw_pose_tensor(raw.data(), {1, count, 14}, 640, 384, confidence);
      const auto resize = aruw_resize_metadata(width, height);
      std::vector<TrackObservation> observations;
      for (const auto & d : detections) {
        TrackObservation o;
        o.bbox = unletterbox_rect(d.bbox_model, resize);
        o.center = {o.bbox.x + o.bbox.width * 0.5F, o.bbox.y + o.bbox.height * 0.5F};
        o.confidence = d.confidence;
        o.class_id = d.class_id;
        o.armor_number = d.armor_number;
        o.armor_type = d.armor_type;
        o.image_width = width;
        o.image_height = height;
        o.has_corners = o.corners_are_keypoints = true;
        o.measurement_stamp_ns = static_cast<std::int64_t>((timestamp + 1) * 1e9);
        for (std::size_t i = 0; i < 4; ++i) {
          o.corners_source[i] = unletterbox_point(d.corners_model[i], resize);
        }
        observations.push_back(o);
      }
      const double dt = previous_time < 0 ? 1.0 / 30 : timestamp - previous_time;
      previous_time = timestamp;
      const auto result = tracker.update(observations, dt);
      output << "{\"frame\":" << frame << ",\"time_s\":" << timestamp
             << ",\"detections\":" << detections.size()
             << ",\"selected_id\":" << result.selected_track_id.value_or(-1) << ",\"tracks\":[";
      bool first = true;
      for (const auto & t : result.tracks) {
        if (!first) {
          output << ',';
        }
        first = false;
        const auto & o = t.observation;
        output << "{\"id\":" << t.track_id << ",\"state\":" << static_cast<int>(t.state)
               << ",\"lost_frames\":" << t.lost_frames << ",\"age\":" << t.age
               << ",\"class_id\":" << o.class_id << ",\"number\":" << o.armor_number
               << ",\"confidence\":" << o.confidence << ",\"center\":[" << o.center.x << ','
               << o.center.y << ']' << ",\"prediction\":[" << t.predicted_center.x << ','
               << t.predicted_center.y << ']' << ",\"bbox\":[" << o.bbox.x << ',' << o.bbox.y << ','
               << o.bbox.width << ',' << o.bbox.height << ']' << ",\"corners\":[";
        for (std::size_t i = 0; i < 4; ++i) {
          if (i) {
            output << ',';
          }
          output << '[' << o.corners_source[i].x << ',' << o.corners_source[i].y << ']';
        }
        output << "]}";
      }
      output << "]}\n";
    }
    if (!input.eof()) {
      throw std::runtime_error("Malformed replay input");
    }
    if (!output) {
      throw std::runtime_error("Replay output write failed");
    }
  } catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
