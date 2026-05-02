# Debug output from the armor processor spin observer.
std_msgs/Header header
# True when recent armor jumps indicate a spinning target.
bool target_spinning
# Fire recommendation after spin gating.
bool suggest_fire
# Count of accepted armor jumps for the current target.
int64 jump_count
# Latest yaw jump difference in radians.
float64 yaw_diff
# Time between recent jumps in seconds.
float64 jump_period
# Seconds elapsed after the latest detected jump.
float64 time_after_jumping
