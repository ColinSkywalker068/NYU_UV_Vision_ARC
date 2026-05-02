# Tracker output for one detector frame.
std_msgs/Header header
# Camera/backend source id inherited from the detector message.
string source_id
# Active tracks, including confirmed and short-coasting tracks.
cv_auto_aim_interfaces/TrackedArmor[] tracks
# Selected target track id, or -1 when no target is selected.
int32 selected_track_id
