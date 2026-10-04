# Tracker output for one detector frame.
std_msgs/Header header
# Camera/backend source id inherited from the detector message.
string source_id
# Active tracks, including confirmed and short-coasting tracks.
cv_auto_aim_interfaces/TrackedArmor[] tracks
# Selected target track id, or -1 when no target is selected.
int32 selected_track_id

# Frame for pose_tracking and camera_pose. Axes: X right, Y down, Z forward
# at controller zero attitude; independent of current camera rotation.
string pose_frame_id
bool camera_pose_valid
geometry_msgs/Pose camera_pose

# Cheap frame diagnostics before enemy/pose filtering; matched excludes new tracks.
uint32 raw_count
uint32 enemy_count
uint32 posed_count
uint32 matched_count
uint32 quality_rejected_count
uint32 innovation_rejected_count
