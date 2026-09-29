#ifndef YOLO_CPP__TRACKER_HPP_
#define YOLO_CPP__TRACKER_HPP_

#include <array>
#include <cstdint>
#include <opencv2/core.hpp>
#include <optional>
#include <utility>
#include <vector>

namespace yolo_cpp
{

/// Constant-acceleration Kalman tracker for a single armor plate in image space.
class ArmorPlateTracker
{
public:
  /// Initialize a track from the first observed bounding box.
  ///
  /// Args:
  ///   x: Initial center x coordinate in pixels.
  ///   y: Initial center y coordinate in pixels.
  ///   width: Initial bounding-box width in pixels.
  ///   height: Initial bounding-box height in pixels.
  ///   proc_noise_pos: Process noise for position states.
  ///   proc_noise_vel: Process noise for velocity states.
  ///   proc_noise_acc: Process noise for acceleration states.
  ///   meas_noise: Measurement noise for observed centers.
  ///   init_cov: Initial covariance on all state dimensions.
  ArmorPlateTracker(double x, double y, double width, double height, double proc_noise_pos = 0.01,
    double proc_noise_vel = 0.05, double proc_noise_acc = 0.1, double meas_noise = 0.15,
    double init_cov = 0.1);

  /// Advance the filter by dt seconds and return the predicted center.
  std::pair<double, double> predict(double dt);

  /// Correct the filter with a measured center and size.
  std::pair<double, double> update(double x, double y, double width, double height);

  /// Project the current state forward without mutating the tracker.
  std::pair<double, double> project(double dt) const;

  /// Return the raw state vector [x, y, vx, vy, ax, ay].
  const std::array<double, 6>& state() const;

  /// Return the latest tracked bounding-box width in pixels.
  double width() const;

  /// Return the latest tracked bounding-box height in pixels.
  double height() const;

private:
  std::array<double, 6> state_;
  std::array<double, 6> q_diag_;
  double r_{0.15};
  double p_[6][6]{};
  double width_{0.0};
  double height_{0.0};
};

struct TrackerWrapper
{
  /// Wrap an initialized tracker with metadata used by the legacy pipeline.
  explicit TrackerWrapper(ArmorPlateTracker tracker_in) : tracker(std::move(tracker_in)) {}
  ArmorPlateTracker tracker;
  int lost_frames{0};
  std::optional<float> obb_angle;
};

enum class TrackState : std::uint8_t
{
  /// A new track that has not yet met the confirmation threshold.
  kDetecting = 0,
  /// A confirmed track updated by a detection this frame.
  kTracking = 1,
  /// A confirmed track predicted through a short detection gap.
  kCoasting = 2,
};

/// Normalized detector output used as input to the multi-target tracker.
struct TrackObservation
{
  cv::Rect bbox;
  cv::Point2f center;
  float confidence{0.0F};
  int class_id{0};
  bool is_enemy{true};
  std::optional<float> obb_angle;
  int image_width{0};
  int image_height{0};
  bool has_corners{false};
  std::array<cv::Point2f, 4> corners_source{};
  int armor_type{0};
  int armor_number{-1};
  float distance_to_image_center{0.0F};
  bool has_pose{false};
  cv::Point3d position_camera;
};

/// Public view of one track after prediction, association, and state update.
struct TrackSnapshot
{
  int track_id{-1};
  TrackState state{TrackState::kDetecting};
  std::uint32_t age{0};
  std::uint32_t hit_streak{0};
  std::uint32_t lost_frames{0};
  TrackObservation observation;
  cv::Point2f predicted_center;
  bool has_pose{false};
  cv::Point3d predicted_position_camera;
  bool has_velocity{false};
  cv::Vec3d velocity_camera;
  bool selectable{false};
};

/// Tunable association, confirmation, coasting, and target-lock parameters.
struct TrackConfig
{
  int track_min_hits{1};
  int track_max_coast_frames{1};
  int track_drop_frames{4};
  double association_max_center_distance_ratio{0.30};
  double association_min_iou{0.0};
  double association_max_3d_distance_m{0.35};
  double target_switch_min_score_delta{0.02};
  int target_lock_hysteresis_frames{0};
};

/// Tracker output for one processed frame.
struct TrackingFrameResult
{
  std::vector<TrackSnapshot> tracks;
  std::optional<TrackSnapshot> selected_target;
  std::optional<int> selected_track_id;
};

/// Maintains armor tracks across frames and chooses the active target.
class TrackManager
{
public:
  /// Construct a manager with the supplied tracking parameters.
  explicit TrackManager(TrackConfig config = {});

  /// Replace tracking parameters used on subsequent frames.
  void set_config(const TrackConfig& config);

  /// Return the active tracking configuration.
  const TrackConfig& config() const;

  /// Drop all tracks and target lock state.
  void reset();

  /// Process one frame of detections.
  ///
  /// Args:
  ///   detections: Armor observations for the current frame.
  ///   dt: Seconds elapsed since the previous processed frame.
  ///
  /// Returns:
  ///   Updated track snapshots plus the selected target, when one is available.
  TrackingFrameResult update(const std::vector<TrackObservation>& detections, double dt);

private:
  struct TrackRecord
  {
    int track_id{-1};
    ArmorPlateTracker tracker{0.0, 0.0, 0.0, 0.0};
    TrackState state{TrackState::kDetecting};
    std::uint32_t age{0};
    std::uint32_t hit_streak{0};
    std::uint32_t lost_frames{0};
    TrackObservation observation;
    cv::Point2f predicted_center;
    bool has_3d_state{false};
    cv::Point3d position_camera;
    cv::Point3d predicted_position_camera;
    cv::Vec3d velocity_camera;
  };

  struct RankedTrack
  {
    const TrackSnapshot* snapshot{nullptr};
    double center_distance{0.0};
    double selection_score{0.0};
  };

  TrackSnapshot make_snapshot(const TrackRecord& track) const;
  std::optional<int> choose_selected_track_id(const std::vector<TrackSnapshot>& tracks);
  bool is_selectable(const TrackRecord& track) const;

  TrackConfig config_;
  int next_track_id_{1};
  std::optional<int> locked_track_id_;
  std::vector<TrackRecord> tracks_;
};

}  // namespace yolo_cpp

#endif
