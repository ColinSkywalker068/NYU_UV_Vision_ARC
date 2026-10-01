# Estimated measurement time in the Jetson ROS clock. The current SP protocol
# has no MCU timestamp: the bridge subtracts its calibrated receive delay.
std_msgs/Header header
int64 receive_stamp_ns
bool source_timestamped
bool valid
# Controller convention: radians, +yaw right, +pitch up, +roll clockwise.
float64 yaw
float64 pitch
float64 roll
float64 yaw_rate
float64 pitch_rate
