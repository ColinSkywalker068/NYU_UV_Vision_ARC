#include "yolo_cpp/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace yolo_cpp
{

namespace
{

void multiply_6x6(const double a[6][6], const double b[6][6], double out[6][6])
{
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      double sum = 0.0;
      for (int k = 0; k < 6; ++k) {
        sum += a[r][k] * b[k][c];
      }
      out[r][c] = sum;
    }
  }
}

void transpose_6x6(const double in[6][6], double out[6][6])
{
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      out[c][r] = in[r][c];
    }
  }
}

constexpr double kMinimumDt = 1e-3;
constexpr double kInvalidAssignmentCost = 1e6;

cv::Point2f clamp_center(
  const cv::Point2f & center, int image_width, int image_height)
{
  const float max_x = static_cast<float>(std::max(0, image_width - 1));
  const float max_y = static_cast<float>(std::max(0, image_height - 1));
  return cv::Point2f(
    std::clamp(center.x, 0.0F, max_x),
    std::clamp(center.y, 0.0F, max_y));
}

cv::Rect rect_from_center(
  const cv::Point2f & center, double width, double height, int image_width, int image_height)
{
  const int max_w = std::max(1, image_width);
  const int max_h = std::max(1, image_height);
  int x1 = static_cast<int>(std::round(static_cast<double>(center.x) - (width / 2.0)));
  int y1 = static_cast<int>(std::round(static_cast<double>(center.y) - (height / 2.0)));
  int x2 = static_cast<int>(std::round(static_cast<double>(center.x) + (width / 2.0)));
  int y2 = static_cast<int>(std::round(static_cast<double>(center.y) + (height / 2.0)));
  x1 = std::clamp(x1, 0, max_w - 1);
  y1 = std::clamp(y1, 0, max_h - 1);
  x2 = std::clamp(x2, 0, max_w - 1);
  y2 = std::clamp(y2, 0, max_h - 1);
  if (x2 <= x1) {
    x2 = std::min(max_w, x1 + 1);
  }
  if (y2 <= y1) {
    y2 = std::min(max_h, y1 + 1);
  }
  return cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1));
}

double rect_iou(const cv::Rect & lhs, const cv::Rect & rhs)
{
  const int x1 = std::max(lhs.x, rhs.x);
  const int y1 = std::max(lhs.y, rhs.y);
  const int x2 = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
  const int y2 = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
  const int inter_w = std::max(0, x2 - x1);
  const int inter_h = std::max(0, y2 - y1);
  const double inter = static_cast<double>(inter_w) * static_cast<double>(inter_h);
  const double lhs_area = static_cast<double>(lhs.width) * static_cast<double>(lhs.height);
  const double rhs_area = static_cast<double>(rhs.width) * static_cast<double>(rhs.height);
  const double denom = lhs_area + rhs_area - inter;
  if (denom <= 0.0) {
    return 0.0;
  }
  return inter / denom;
}

double normalized_center_distance(
  const cv::Point2f & lhs, const cv::Point2f & rhs, int image_width, int image_height)
{
  const double dx = static_cast<double>(lhs.x - rhs.x);
  const double dy = static_cast<double>(lhs.y - rhs.y);
  const double diagonal = std::hypot(
    static_cast<double>(std::max(1, image_width)),
    static_cast<double>(std::max(1, image_height)));
  if (diagonal <= 0.0) {
    return 0.0;
  }
  return std::hypot(dx, dy) / diagonal;
}

double distance_3d(const cv::Point3d & lhs, const cv::Point3d & rhs)
{
  return std::sqrt(
    ((lhs.x - rhs.x) * (lhs.x - rhs.x)) +
    ((lhs.y - rhs.y) * (lhs.y - rhs.y)) +
    ((lhs.z - rhs.z) * (lhs.z - rhs.z)));
}

cv::Point2f project_center(const ArmorPlateTracker & tracker, double dt)
{
  const auto projected = tracker.project(dt);
  return cv::Point2f(
    static_cast<float>(projected.first),
    static_cast<float>(projected.second));
}

std::vector<int> solve_hungarian(const std::vector<std::vector<double>> & cost_matrix)
{
  if (cost_matrix.empty() || cost_matrix.front().empty()) {
    return {};
  }

  const int rows = static_cast<int>(cost_matrix.size());
  const int cols = static_cast<int>(cost_matrix.front().size());
  const int size = std::max(rows, cols);

  std::vector<std::vector<double>> padded(
    static_cast<std::size_t>(size),
    std::vector<double>(static_cast<std::size_t>(size), kInvalidAssignmentCost));
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      padded[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = cost_matrix[r][c];
    }
  }

  std::vector<double> u(static_cast<std::size_t>(size + 1), 0.0);
  std::vector<double> v(static_cast<std::size_t>(size + 1), 0.0);
  std::vector<int> p(static_cast<std::size_t>(size + 1), 0);
  std::vector<int> way(static_cast<std::size_t>(size + 1), 0);

  for (int i = 1; i <= size; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(static_cast<std::size_t>(size + 1), kInvalidAssignmentCost);
    std::vector<bool> used(static_cast<std::size_t>(size + 1), false);
    do {
      used[static_cast<std::size_t>(j0)] = true;
      const int i0 = p[static_cast<std::size_t>(j0)];
      int j1 = 0;
      double delta = kInvalidAssignmentCost;
      for (int j = 1; j <= size; ++j) {
        if (used[static_cast<std::size_t>(j)]) {
          continue;
        }
        const double cur = padded[static_cast<std::size_t>(i0 - 1)][static_cast<std::size_t>(j - 1)] -
          u[static_cast<std::size_t>(i0)] - v[static_cast<std::size_t>(j)];
        if (cur < minv[static_cast<std::size_t>(j)]) {
          minv[static_cast<std::size_t>(j)] = cur;
          way[static_cast<std::size_t>(j)] = j0;
        }
        if (minv[static_cast<std::size_t>(j)] < delta) {
          delta = minv[static_cast<std::size_t>(j)];
          j1 = j;
        }
      }
      for (int j = 0; j <= size; ++j) {
        if (used[static_cast<std::size_t>(j)]) {
          u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta;
          v[static_cast<std::size_t>(j)] -= delta;
        } else {
          minv[static_cast<std::size_t>(j)] -= delta;
        }
      }
      j0 = j1;
    } while (p[static_cast<std::size_t>(j0)] != 0);

    do {
      const int j1 = way[static_cast<std::size_t>(j0)];
      p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
      j0 = j1;
    } while (j0 != 0);
  }

  std::vector<int> assignment(static_cast<std::size_t>(rows), -1);
  for (int j = 1; j <= size; ++j) {
    const int row = p[static_cast<std::size_t>(j)] - 1;
    const int col = j - 1;
    if (row >= 0 && row < rows && col >= 0 && col < cols) {
      assignment[static_cast<std::size_t>(row)] = col;
    }
  }
  return assignment;
}

int state_priority(TrackState state)
{
  switch (state) {
    case TrackState::kTracking:
      return 2;
    case TrackState::kDetecting:
      return 1;
    case TrackState::kCoasting:
      return 0;
  }
  return -1;
}

}  // namespace

ArmorPlateTracker::ArmorPlateTracker(
  double x, double y, double width, double height, double proc_noise_pos, double proc_noise_vel,
  double proc_noise_acc, double meas_noise, double init_cov)
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
  double f[6][6] = {
    {1, 0, dt, 0, dt2, 0},
    {0, 1, 0, dt, 0, dt2},
    {0, 0, 1, 0, dt, 0},
    {0, 0, 0, 1, 0, dt},
    {0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 1}};

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
  return {
    state_[0] + (state_[2] * dt_safe) + (state_[4] * dt2),
    state_[1] + (state_[3] * dt_safe) + (state_[5] * dt2)};
}

const std::array<double, 6> & ArmorPlateTracker::state() const
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

TrackManager::TrackManager(TrackConfig config)
: config_(std::move(config))
{
}

void TrackManager::set_config(const TrackConfig & config)
{
  config_ = config;
}

const TrackConfig & TrackManager::config() const
{
  return config_;
}

void TrackManager::reset()
{
  tracks_.clear();
  locked_track_id_.reset();
  next_track_id_ = 1;
}

TrackingFrameResult TrackManager::update(
  const std::vector<TrackObservation> & detections, double dt)
{
  const double dt_safe = std::max(dt, kMinimumDt);
  std::vector<TrackObservation> enemy_detections;
  enemy_detections.reserve(detections.size());
  for (const auto & detection : detections) {
    if (detection.is_enemy) {
      enemy_detections.push_back(detection);
    }
  }

  for (auto & track : tracks_) {
    ++track.age;
    const auto predicted = track.tracker.predict(dt_safe);
    track.predicted_center = clamp_center(
      cv::Point2f(
        static_cast<float>(predicted.first),
        static_cast<float>(predicted.second)),
      track.observation.image_width,
      track.observation.image_height);
    if (track.has_3d_state) {
      track.predicted_position_camera = cv::Point3d(
        track.position_camera.x + (track.velocity_camera[0] * dt_safe),
        track.position_camera.y + (track.velocity_camera[1] * dt_safe),
        track.position_camera.z + (track.velocity_camera[2] * dt_safe));
    }
  }

  std::vector<std::vector<double>> cost_matrix(
    tracks_.size(), std::vector<double>(enemy_detections.size(), kInvalidAssignmentCost));
  for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
    const auto predicted_bbox = rect_from_center(
      tracks_[track_index].predicted_center,
      tracks_[track_index].tracker.width(),
      tracks_[track_index].tracker.height(),
      tracks_[track_index].observation.image_width,
      tracks_[track_index].observation.image_height);
    for (std::size_t detection_index = 0; detection_index < enemy_detections.size(); ++detection_index) {
      const auto & detection = enemy_detections[detection_index];
      if (tracks_[track_index].has_3d_state && detection.has_pose) {
        const double position_distance = distance_3d(
          tracks_[track_index].predicted_position_camera,
          detection.position_camera);
        if (position_distance > config_.association_max_3d_distance_m) {
          continue;
        }
        cost_matrix[track_index][detection_index] =
          position_distance / std::max(1e-6, config_.association_max_3d_distance_m);
      } else {
        const double center_ratio = normalized_center_distance(
          tracks_[track_index].predicted_center,
          detection.center,
          detection.image_width,
          detection.image_height);
        const double iou = rect_iou(predicted_bbox, detection.bbox);
        if (
          center_ratio > config_.association_max_center_distance_ratio ||
          iou < config_.association_min_iou)
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
    if (
      detection_index >= 0 &&
      static_cast<std::size_t>(detection_index) < enemy_detections.size() &&
      cost_matrix[track_index][static_cast<std::size_t>(detection_index)] < kInvalidAssignmentCost)
    {
      auto & track = tracks_[track_index];
      const auto & detection = enemy_detections[static_cast<std::size_t>(detection_index)];
      track.tracker.update(
        static_cast<double>(detection.center.x),
        static_cast<double>(detection.center.y),
        static_cast<double>(detection.bbox.width),
        static_cast<double>(detection.bbox.height));
      track.hit_streak += 1U;
      track.lost_frames = 0;
      if (detection.has_pose) {
        if (track.has_3d_state) {
          const cv::Vec3d measured_velocity(
            (detection.position_camera.x - track.position_camera.x) / dt_safe,
            (detection.position_camera.y - track.position_camera.y) / dt_safe,
            (detection.position_camera.z - track.position_camera.z) / dt_safe);
          track.velocity_camera = cv::Vec3d(
            (0.5 * track.velocity_camera[0]) + (0.5 * measured_velocity[0]),
            (0.5 * track.velocity_camera[1]) + (0.5 * measured_velocity[1]),
            (0.5 * track.velocity_camera[2]) + (0.5 * measured_velocity[2]));
          track.position_camera = cv::Point3d(
            (0.3 * track.predicted_position_camera.x) + (0.7 * detection.position_camera.x),
            (0.3 * track.predicted_position_camera.y) + (0.7 * detection.position_camera.y),
            (0.3 * track.predicted_position_camera.z) + (0.7 * detection.position_camera.z));
        } else {
          track.position_camera = detection.position_camera;
          track.velocity_camera = cv::Vec3d(0.0, 0.0, 0.0);
          track.has_3d_state = true;
        }
        track.predicted_position_camera = cv::Point3d(
          track.position_camera.x + (track.velocity_camera[0] * dt_safe),
          track.position_camera.y + (track.velocity_camera[1] * dt_safe),
          track.position_camera.z + (track.velocity_camera[2] * dt_safe));
      } else if (track.has_3d_state) {
        track.position_camera = track.predicted_position_camera;
      }
      track.observation = detection;
      track.state =
        track.hit_streak >= static_cast<std::uint32_t>(config_.track_min_hits) ?
        TrackState::kTracking :
        TrackState::kDetecting;
      track.predicted_center = clamp_center(
        project_center(track.tracker, dt_safe),
        detection.image_width,
        detection.image_height);
      detection_used[static_cast<std::size_t>(detection_index)] = true;
    } else {
      auto & track = tracks_[track_index];
      track.lost_frames += 1U;
      if (track.has_3d_state) {
        track.position_camera = track.predicted_position_camera;
      }
      if (track.state == TrackState::kTracking || track.state == TrackState::kCoasting) {
        track.state = TrackState::kCoasting;
      }
    }
  }

  for (std::size_t detection_index = 0; detection_index < enemy_detections.size(); ++detection_index) {
    if (detection_used[detection_index]) {
      continue;
    }
    const auto & detection = enemy_detections[detection_index];
    TrackRecord record;
    record.track_id = next_track_id_++;
    record.tracker = ArmorPlateTracker(
      static_cast<double>(detection.center.x),
      static_cast<double>(detection.center.y),
      static_cast<double>(detection.bbox.width),
      static_cast<double>(detection.bbox.height));
    record.state = TrackState::kDetecting;
    record.age = 1U;
    record.hit_streak = 1U;
    record.lost_frames = 0U;
    record.observation = detection;
    record.predicted_center = detection.center;
    if (detection.has_pose) {
      record.has_3d_state = true;
      record.position_camera = detection.position_camera;
      record.predicted_position_camera = detection.position_camera;
      record.velocity_camera = cv::Vec3d(0.0, 0.0, 0.0);
    }
    tracks_.push_back(std::move(record));
  }

  tracks_.erase(
    std::remove_if(
      tracks_.begin(), tracks_.end(),
      [this](const TrackRecord & track) {
        return track.lost_frames > static_cast<std::uint32_t>(config_.track_drop_frames);
      }),
    tracks_.end());

  TrackingFrameResult result;
  result.tracks.reserve(tracks_.size());
  for (const auto & track : tracks_) {
    result.tracks.push_back(make_snapshot(track));
  }
  result.selected_track_id = choose_selected_track_id(result.tracks);
  if (result.selected_track_id.has_value()) {
    const auto selected_id = *result.selected_track_id;
    auto it = std::find_if(
      result.tracks.begin(), result.tracks.end(),
      [selected_id](const TrackSnapshot & track) {return track.track_id == selected_id;});
    if (it != result.tracks.end()) {
      result.selected_target = *it;
    }
  }
  return result;
}

TrackSnapshot TrackManager::make_snapshot(const TrackRecord & track) const
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
    snapshot.observation.position_camera = track.position_camera;
    snapshot.predicted_position_camera = track.predicted_position_camera;
    snapshot.has_velocity = true;
    snapshot.velocity_camera = track.velocity_camera;
  }
  if (track.state == TrackState::kCoasting && track.lost_frames > 0U) {
    snapshot.observation.center = clamp_center(
      track.predicted_center,
      track.observation.image_width,
      track.observation.image_height);
    snapshot.observation.bbox = rect_from_center(
      snapshot.observation.center,
      track.tracker.width(),
      track.tracker.height(),
      track.observation.image_width,
      track.observation.image_height);
    if (track.has_3d_state) {
      snapshot.observation.position_camera = track.position_camera;
    }
  }
  snapshot.predicted_center = clamp_center(
    track.predicted_center,
    track.observation.image_width,
    track.observation.image_height);
  snapshot.selectable = is_selectable(track);
  return snapshot;
}

bool TrackManager::is_selectable(const TrackRecord & track) const
{
  switch (track.state) {
    case TrackState::kTracking:
      return track.lost_frames == 0U;
    case TrackState::kDetecting:
      return track.lost_frames == 0U;
    case TrackState::kCoasting:
      return track.lost_frames <= static_cast<std::uint32_t>(config_.track_max_coast_frames);
  }
  return false;
}

std::optional<int> TrackManager::choose_selected_track_id(const std::vector<TrackSnapshot> & tracks)
{
  std::vector<RankedTrack> ranked;
  ranked.reserve(tracks.size());
  std::unordered_map<int, const TrackSnapshot *> by_id;
  for (const auto & track : tracks) {
    by_id[track.track_id] = &track;
    if (!track.selectable) {
      continue;
    }
    const auto & center = track.state == TrackState::kCoasting ?
      track.predicted_center :
      track.observation.center;
    const double center_distance = normalized_center_distance(
      center,
      cv::Point2f(
        static_cast<float>(track.observation.image_width) / 2.0F,
        static_cast<float>(track.observation.image_height) / 2.0F),
      track.observation.image_width,
      track.observation.image_height);
    ranked.push_back(RankedTrack{&track, center_distance});
  }

  if (ranked.empty()) {
    locked_track_id_.reset();
    return std::nullopt;
  }

  std::sort(
    ranked.begin(), ranked.end(),
    [](const RankedTrack & lhs, const RankedTrack & rhs) {
      if (state_priority(lhs.snapshot->state) != state_priority(rhs.snapshot->state)) {
        return state_priority(lhs.snapshot->state) > state_priority(rhs.snapshot->state);
      }
      if (lhs.snapshot->observation.confidence != rhs.snapshot->observation.confidence) {
        return lhs.snapshot->observation.confidence > rhs.snapshot->observation.confidence;
      }
      if (lhs.center_distance != rhs.center_distance) {
        return lhs.center_distance < rhs.center_distance;
      }
      return lhs.snapshot->age > rhs.snapshot->age;
    });

  const TrackSnapshot * selected = ranked.front().snapshot;

  if (locked_track_id_.has_value()) {
    const auto locked_it = by_id.find(*locked_track_id_);
    if (locked_it != by_id.end() && locked_it->second->selectable) {
      const auto * locked = locked_it->second;
      if (locked->track_id != selected->track_id) {
        const double locked_distance = normalized_center_distance(
          locked->state == TrackState::kCoasting ? locked->predicted_center : locked->observation.center,
          cv::Point2f(
            static_cast<float>(locked->observation.image_width) / 2.0F,
            static_cast<float>(locked->observation.image_height) / 2.0F),
          locked->observation.image_width,
          locked->observation.image_height);
        const bool locked_exceeded_coast =
          locked->state == TrackState::kCoasting &&
          locked->lost_frames > static_cast<std::uint32_t>(config_.target_lock_hysteresis_frames);
        const bool challenger_is_better =
          (selected->observation.confidence >=
          locked->observation.confidence + static_cast<float>(config_.target_switch_min_score_delta)) &&
          (ranked.front().center_distance < locked_distance);
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
