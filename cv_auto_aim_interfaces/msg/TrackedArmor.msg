uint8 DETECTING=0
uint8 TRACKING=1
uint8 COASTING=2

cv_auto_aim_interfaces/Armor armor
int32 track_id
uint8 state
uint32 age
uint32 hit_streak
uint32 lost_frames
float32 predicted_center_x
float32 predicted_center_y
bool has_pose
geometry_msgs/Pose pose_camera
geometry_msgs/Point predicted_position_camera
bool has_velocity
geometry_msgs/Vector3 velocity_camera
