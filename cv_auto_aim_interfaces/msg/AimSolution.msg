# Final solver output consumed by the STM bridge and debug tools.
std_msgs/Header header

# True when yaw/pitch/range fields describe a valid target.
bool has_solution
# True when the solver used a 3D pose instead of 2D image offsets.
bool used_3d
# Fire recommendation after tracking, spin, and solver checks.
bool suggest_fire
# Desired yaw in radians.
float32 yaw
# Desired pitch in radians.
float32 pitch
# Desired yaw in degrees for legacy consumers.
float32 yaw_degrees
# Desired pitch in degrees for legacy consumers.
float32 pitch_degrees
# Estimated projectile flight time in seconds.
float32 flight_time
# Target range in meters.
float32 range
# Bullet speed used for this solution in meters per second.
float32 bullet_speed

# Diagnostics and atomic command contract.
string output_mode
string status
float32 frame_age_s
float32 measurement_age_s
float32 prediction_horizon_s
float32 minimum_slew_s
bool reachable
int32 selected_slot

# Atomic rates of the absolute ballistic aim point in gimbal_inertial (rad/s).
# False/zero for acquisition, identity/plate changes, coasting, stale/invalid
# input, 2D/offset preview, or rate/innovation guards. Zero also means stationary.
bool has_angular_velocity
float32 yaw_velocity
float32 pitch_velocity
