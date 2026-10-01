# Tracker-level view of one armor candidate across frames.
uint8 DETECTING=0
uint8 TRACKING=1
uint8 COASTING=2

# Latest detector observation associated with this track.
cv_auto_aim_interfaces/Armor armor
# Stable track id assigned by the tracker.
int32 track_id
# Current track state: DETECTING, TRACKING, or COASTING.
uint8 state
# Frames since the track was created.
uint32 age
# Consecutive detector hits for this track.
uint32 hit_streak
# Consecutive missed frames since the last detection hit.
uint32 lost_frames
# Predicted image-space center after Kalman projection.
float32 predicted_center_x
float32 predicted_center_y
# True when pose_camera and predicted_position_camera are valid.
bool has_pose
# Latest measured camera-frame pose for the armor.
geometry_msgs/Pose pose_camera
# Predicted camera-frame position after tracking projection.
geometry_msgs/Point predicted_position_camera
# True when velocity_camera is valid.
bool has_velocity
# Estimated camera-frame target velocity in meters per second.
geometry_msgs/Vector3 velocity_camera

# Filtered pose, prediction and velocity in TrackedArmors.pose_frame_id.
geometry_msgs/Pose pose_tracking
geometry_msgs/Point predicted_position_tracking
geometry_msgs/Vector3 velocity_tracking
# Time of the most recent real detector observation; unchanged during coasting.
int64 measurement_stamp_ns
