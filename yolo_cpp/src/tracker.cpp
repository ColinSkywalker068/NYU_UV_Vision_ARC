/// @file tracker.cpp
/// @brief Implements Kalman tracking, Hungarian association, coasting, and target lock.

#include "yolo_cpp/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "tracking_math.hpp"

namespace yolo_cpp
{

namespace
{

using tracking_math::clamp_center;
using tracking_math::distance_3d;
using tracking_math::kInvalidAssignmentCost;
using tracking_math::kMinimumDt;
using tracking_math::multiply_6x6;
using tracking_math::normalized_center_distance;
using tracking_math::rect_from_center;
using tracking_math::rect_iou;
using tracking_math::solve_hungarian;
using tracking_math::state_priority;
using tracking_math::transpose_6x6;

/// Project a track center without mutating tracker state.
cv::Point2f project_center(const ArmorPlateTracker& tracker, double dt)
{
  const auto projected = tracker.project(dt);
  return cv::Point2f(static_cast<float>(projected.first), static_cast<float>(projected.second));
}

}  // namespace

ArmorPlateTracker::ArmorPlateTracker(double x, double y, double width, double height,
  double proc_noise_pos, double proc_noise_vel, double proc_noise_acc, double meas_noise,
  double init_cov)
    : state_{x, y, 0.0, 0.0, 0.0, 0.0},
      q_diag_{proc_noise_pos, proc_noise_pos, proc_noise_vel, proc_noise_vel, proc_noise_acc,
        proc_noise_acc},
      r_(meas_noise),
      width_(width),
      height_(height)
{
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      p_[i][j] = (i == j) ? init_cov : 0.0;
    }
  }
}

std::pair<double, double> ArmorPlateTracker::predict(double dt)
{
  const double dt2 = 0.5 * dt * dt;
  double f[6][6] = {{1, 0, dt, 0, dt2, 0}, {0, 1, 0, dt, 0, dt2}, {0, 0, 1, 0, dt, 0},
    {0, 0, 0, 1, 0, dt}, {0, 0, 0, 0, 1, 0}, {0, 0, 0, 0, 0, 1}};

  std::array<double, 6> new_state{};
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      new_state[r] += f[r][c] * state_[c];
    }
  }
  state_ = new_state;

  double ft[6][6]{};
  double fp[6][6]{};
  double fpf_t[6][6]{};
  transpose_6x6(f, ft);
  multiply_6x6(f, p_, fp);
  multiply_6x6(fp, ft, fpf_t);
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      p_[i][j] = fpf_t[i][j] + ((i == j) ? q_diag_[i] : 0.0);
    }
  }

  return {state_[0], state_[1]};
}

std::pair<double, double> ArmorPlateTracker::update(double x, double y, double width, double height)
{
  const double s00 = p_[0][0] + r_;
  const double s01 = p_[0][1];
  const double s10 = p_[1][0];
  const double s11 = p_[1][1] + r_;
  const double det = (s00 * s11) - (s01 * s10);
  if (det == 0.0) {
    return {state_[0], state_[1]};
  }

  const double inv00 = s11 / det;
  const double inv01 = -s01 / det;
  const double inv10 = -s10 / det;
  const double inv11 = s00 / det;

  double k[6][2]{};
  for (int i = 0; i < 6; ++i) {
    k[i][0] = (p_[i][0] * inv00) + (p_[i][1] * inv10);
    k[i][1] = (p_[i][0] * inv01) + (p_[i][1] * inv11);
  }

  const double innovation0 = x - state_[0];
  const double innovation1 = y - state_[1];
  for (int i = 0; i < 6; ++i) {
    state_[i] += (k[i][0] * innovation0) + (k[i][1] * innovation1);
  }

  double new_p[6][6]{};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      new_p[i][j] = p_[i][j] - (k[i][0] * p_[0][j]) - (k[i][1] * p_[1][j]);
    }
  }
  std::memcpy(p_, new_p, sizeof(p_));

  width_ = width;
  height_ = height;
  return {state_[0], state_[1]};
}

std::pair<double, double> ArmorPlateTracker::project(double dt) const
{
  const double dt_safe = std::max(dt, kMinimumDt);
  const double dt2 = 0.5 * dt_safe * dt_safe;
  return {state_[0] + (state_[2] * dt_safe) + (state_[4] * dt2),
    state_[1] + (state_[3] * dt_safe) + (state_[5] * dt2)};
}

const std::array<double, 6>& ArmorPlateTracker::state() const
{
  return state_;
}

double ArmorPlateTracker::width() const
{
  return width_;
}

double ArmorPlateTracker::height() const
{
  return height_;
}

TrackManager::TrackManager(TrackConfig config) {set_config(config);}

void TrackManager::set_config(const TrackConfig& config)
{
  if (!std::isfinite(config.track_max_coast_s) || config.track_max_coast_s < 0 ||
      !std::isfinite(config.track_drop_s) || config.track_drop_s < config.track_max_coast_s ||
      !std::isfinite(config.position_filter_tau_s) || config.position_filter_tau_s < 0 ||
      !std::isfinite(config.velocity_filter_tau_s) || config.velocity_filter_tau_s < 0) {
    throw std::invalid_argument("Invalid tracker coast/drop/filter time constants");
  }
  config_ = config;
}

const TrackConfig& TrackManager::config() const
{
  return config_;
}

void TrackManager::reset()
{
  tracks_.clear();
  locked_track_id_.reset();
  next_track_id_ = 1;
}

TrackingFrameResult TrackManager::update(const std::vector<TrackObservation>& detections, double dt)
{
  const double dt_safe = std::max(dt, kMinimumDt);
  std::vector<TrackObservation> enemy_detections;
  enemy_detections.reserve(detections.size());
  for (const auto& detection : detections) {
    if (detection.is_enemy) {
      enemy_detections.push_back(detection);
    }
  }

  for (auto& track : tracks_) {
    ++track.age;
    track.detection_age_s += dt_safe;
    track.pose_measurement_age_s += dt_safe;
    const auto predicted = track.tracker.predict(dt_safe);
    track.predicted_center = clamp_center(
      cv::Point2f(static_cast<float>(predicted.first), static_cast<float>(predicted.second)),
      track.observation.image_width, track.observation.image_height);
    if (track.has_3d_state) {
      track.predicted_position =
        cv::Point3d(track.position.x + (track.velocity[0] * dt_safe),
          track.position.y + (track.velocity[1] * dt_safe),
          track.position.z + (track.velocity[2] * dt_safe));
    }
  }

  std::vector<std::vector<double>> cost_matrix(
    tracks_.size(), std::vector<double>(enemy_detections.size(), kInvalidAssignmentCost));
  for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
    const auto predicted_bbox = rect_from_center(tracks_[track_index].predicted_center,
      tracks_[track_index].tracker.width(), tracks_[track_index].tracker.height(),
      tracks_[track_index].observation.image_width, tracks_[track_index].observation.image_height);
    for (std::size_t detection_index = 0; detection_index < enemy_detections.size();
      ++detection_index)
    {
      const auto& detection = enemy_detections[detection_index];
      const auto & prior_observation = tracks_[track_index].observation;
      if (prior_observation.class_id != detection.class_id ||
        (prior_observation.armor_number >= 0 && detection.armor_number >= 0 &&
         prior_observation.armor_number != detection.armor_number)) {continue;}
      if (tracks_[track_index].has_3d_state && detection.has_pose) {
        const double position_distance =
          distance_3d(tracks_[track_index].predicted_position, detection.position);
        if (position_distance > config_.association_max_3d_distance_m) {
          continue;
        }
        cost_matrix[track_index][detection_index] =
          position_distance / std::max(1e-6, config_.association_max_3d_distance_m);
      } else {
        const double center_ratio =
          normalized_center_distance(tracks_[track_index].predicted_center, detection.center,
            detection.image_width, detection.image_height);
        const double iou = rect_iou(predicted_bbox, detection.bbox);
        if (center_ratio > config_.association_max_center_distance_ratio
            || iou < config_.association_min_iou)
        {
          continue;
        }
        cost_matrix[track_index][detection_index] = center_ratio + (1.0 - iou);
      }
    }
  }

  const auto assignments = solve_hungarian(cost_matrix);
  std::vector<bool> detection_used(enemy_detections.size(), false);

  for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
    const int detection_index = track_index < assignments.size() ? assignments[track_index] : -1;
    if (detection_index >= 0 && static_cast<std::size_t>(detection_index) < enemy_detections.size()
        && cost_matrix[track_index][static_cast<std::size_t>(detection_index)]
             < kInvalidAssignmentCost)
    {
      auto& track = tracks_[track_index];
      const auto& detection = enemy_detections[static_cast<std::size_t>(detection_index)];
      track.tracker.update(static_cast<double>(detection.center.x),
        static_cast<double>(detection.center.y), static_cast<double>(detection.bbox.width),
        static_cast<double>(detection.bbox.height));
      track.hit_streak += 1U;
      track.lost_frames = 0;
      track.detection_age_s = 0.0;
      if (detection.has_pose) {
        const double pose_dt = std::max(track.pose_measurement_age_s, kMinimumDt);
        if (track.has_3d_state) {
          // Difference real measurements over their actual gap. Differencing
          // filtered/predicted positions over one frame produces reacquisition kicks.
          const cv::Vec3d measured_velocity(
            (detection.position.x - track.last_pose_measurement_position.x) / pose_dt,
            (detection.position.y - track.last_pose_measurement_position.y) / pose_dt,
            (detection.position.z - track.last_pose_measurement_position.z) / pose_dt);
          const double velocity_gain = config_.velocity_filter_tau_s > 0
            ? -std::expm1(-pose_dt / config_.velocity_filter_tau_s) : 1.0;
          const double position_gain = config_.position_filter_tau_s > 0
            ? -std::expm1(-pose_dt / config_.position_filter_tau_s) : 1.0;
          track.velocity = (1 - velocity_gain) * track.velocity + velocity_gain * measured_velocity;
          track.position = (1 - position_gain) * track.predicted_position +
            position_gain * detection.position;
        } else {
          track.position = detection.position;
          track.velocity = cv::Vec3d(0.0, 0.0, 0.0);
          track.has_3d_state = true;
        }
        track.pose_measurement_stamp_ns = detection.measurement_stamp_ns;
        track.last_pose_measurement_position = detection.position;
        track.pose_measurement_age_s = 0.0;
        track.predicted_position =
          cv::Point3d(track.position.x + (track.velocity[0] * dt_safe),
            track.position.y + (track.velocity[1] * dt_safe),
            track.position.z + (track.velocity[2] * dt_safe));
      } else if (track.has_3d_state) {
        track.position = track.predicted_position;
      }
      track.observation = detection;
      track.state = track.hit_streak >= static_cast<std::uint32_t>(config_.track_min_hits)
                      ? TrackState::kTracking
                      : TrackState::kDetecting;
      track.predicted_center = clamp_center(
        project_center(track.tracker, dt_safe), detection.image_width, detection.image_height);
      detection_used[static_cast<std::size_t>(detection_index)] = true;
    } else {
      auto& track = tracks_[track_index];
      track.lost_frames += 1U;
      if (track.has_3d_state) {
        track.position = track.predicted_position;
      }
      if (track.state == TrackState::kTracking || track.state == TrackState::kCoasting) {
        track.state = TrackState::kCoasting;
      }
    }
  }

  for (std::size_t detection_index = 0; detection_index < enemy_detections.size();
    ++detection_index)
  {
    if (detection_used[detection_index]) {
      continue;
    }
    const auto& detection = enemy_detections[detection_index];
    TrackRecord record;
    record.track_id = next_track_id_++;
    record.tracker = ArmorPlateTracker(static_cast<double>(detection.center.x),
      static_cast<double>(detection.center.y), static_cast<double>(detection.bbox.width),
      static_cast<double>(detection.bbox.height));
    record.state = TrackState::kDetecting;
    record.age = 1U;
    record.hit_streak = 1U;
    record.lost_frames = 0U;
    record.observation = detection;
    record.predicted_center = detection.center;
    if (detection.has_pose) {
      record.has_3d_state = true;
      record.pose_measurement_stamp_ns = detection.measurement_stamp_ns;
      record.position = detection.position;
      record.last_pose_measurement_position = detection.position;
      record.predicted_position = detection.position;
      record.velocity = cv::Vec3d(0.0, 0.0, 0.0);
    }
    tracks_.push_back(std::move(record));
  }

  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                  [this](const TrackRecord& track) {
                    return track.lost_frames
                           > static_cast<std::uint32_t>(config_.track_drop_frames) ||
                           track.detection_age_s > config_.track_drop_s;
                  }),
    tracks_.end());

  TrackingFrameResult result;
  result.tracks.reserve(tracks_.size());
  for (const auto& track : tracks_) {
    result.tracks.push_back(make_snapshot(track));
  }
  result.selected_track_id = choose_selected_track_id(result.tracks);
  if (result.selected_track_id.has_value()) {
    const auto selected_id = *result.selected_track_id;
    auto it = std::find_if(result.tracks.begin(), result.tracks.end(),
      [selected_id](const TrackSnapshot& track) { return track.track_id == selected_id; });
    if (it != result.tracks.end()) {
      result.selected_target = *it;
    }
  }
  return result;
}

TrackSnapshot TrackManager::make_snapshot(const TrackRecord& track) const
{
  TrackSnapshot snapshot;
  snapshot.track_id = track.track_id;
  snapshot.state = track.state;
  snapshot.age = track.age;
  snapshot.hit_streak = track.hit_streak;
  snapshot.lost_frames = track.lost_frames;
  snapshot.observation = track.observation;
  if (track.has_3d_state) {
    snapshot.has_pose = true;
    snapshot.observation.has_pose = true;
    // Fresh 2D boxes must not make a stale retained 3D pose look newly measured.
    snapshot.observation.measurement_stamp_ns = track.pose_measurement_stamp_ns;
    snapshot.observation.position = track.position;
    snapshot.predicted_position = track.predicted_position;
    snapshot.has_velocity = true;
    snapshot.velocity = track.velocity;
  }
  if (track.state == TrackState::kCoasting && track.lost_frames > 0U) {
    snapshot.observation.center = clamp_center(
      track.predicted_center, track.observation.image_width, track.observation.image_height);
    snapshot.observation.bbox = rect_from_center(snapshot.observation.center, track.tracker.width(),
      track.tracker.height(), track.observation.image_width, track.observation.image_height);
    if (track.has_3d_state) {
      snapshot.observation.position = track.position;
    }
  }
  snapshot.predicted_center = clamp_center(
    track.predicted_center, track.observation.image_width, track.observation.image_height);
  snapshot.selectable = is_selectable(track);
  return snapshot;
}

bool TrackManager::is_selectable(const TrackRecord& track) const
{
  switch (track.state) {
    case TrackState::kTracking:
      return track.lost_frames == 0U;
    case TrackState::kDetecting:
      return track.lost_frames == 0U;
    case TrackState::kCoasting:
      return track.lost_frames <= static_cast<std::uint32_t>(config_.track_max_coast_frames) &&
             track.detection_age_s <= config_.track_max_coast_s;
  }
  return false;
}

std::optional<int> TrackManager::choose_selected_track_id(const std::vector<TrackSnapshot>& tracks)
{
  std::vector<RankedTrack> ranked;
  ranked.reserve(tracks.size());
  std::unordered_map<int, const TrackSnapshot*> by_id;
  for (const auto& track : tracks) {
    by_id[track.track_id] = &track;
    if (!track.selectable) {
      continue;
    }
    const auto& center =
      track.state == TrackState::kCoasting ? track.predicted_center : track.observation.center;
    const double center_distance = normalized_center_distance(center,
      cv::Point2f(static_cast<float>(track.observation.image_width) / 2.0F,
        static_cast<float>(track.observation.image_height) / 2.0F),
      track.observation.image_width, track.observation.image_height);
    // Confidence and proximity both matter. Expressing the lock hysteresis in this same score
    // prevents a far-off stale armor from winning solely because its confidence is marginally
    // higher, while still avoiding switches for small frame-to-frame fluctuations.
    const double selection_score = track.observation.confidence - center_distance;
    ranked.push_back(RankedTrack{&track, center_distance, selection_score});
  }

  if (ranked.empty()) {
    locked_track_id_.reset();
    return std::nullopt;
  }

  std::sort(ranked.begin(), ranked.end(), [](const RankedTrack& lhs, const RankedTrack& rhs) {
    if (state_priority(lhs.snapshot->state) != state_priority(rhs.snapshot->state)) {
      return state_priority(lhs.snapshot->state) > state_priority(rhs.snapshot->state);
    }
    if (lhs.selection_score != rhs.selection_score) {
      return lhs.selection_score > rhs.selection_score;
    }
    if (lhs.center_distance != rhs.center_distance) {
      return lhs.center_distance < rhs.center_distance;
    }
    return lhs.snapshot->age > rhs.snapshot->age;
  });

  const TrackSnapshot* selected = ranked.front().snapshot;

  if (locked_track_id_.has_value()) {
    const auto locked_it = by_id.find(*locked_track_id_);
    if (locked_it != by_id.end() && locked_it->second->selectable) {
      const auto* locked = locked_it->second;
      if (locked->track_id != selected->track_id) {
        const cv::Point2f locked_center = locked->state == TrackState::kCoasting
                                            ? locked->predicted_center
                                            : locked->observation.center;
        const double locked_distance = normalized_center_distance(locked_center,
          cv::Point2f(static_cast<float>(locked->observation.image_width) / 2.0F,
            static_cast<float>(locked->observation.image_height) / 2.0F),
          locked->observation.image_width, locked->observation.image_height);
        const bool locked_exceeded_coast = locked->state == TrackState::kCoasting
                                           && locked->lost_frames > static_cast<std::uint32_t>(
                                                config_.target_lock_hysteresis_frames);
        const double locked_score = locked->observation.confidence - locked_distance;
        const bool challenger_is_better =
          ranked.front().selection_score >=
          locked_score + config_.target_switch_min_score_delta;
        if (!locked_exceeded_coast && !challenger_is_better) {
          selected = locked;
        }
      }
    }
  }

  locked_track_id_ = selected->track_id;
  return selected->track_id;
}

}  // namespace yolo_cpp
