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
