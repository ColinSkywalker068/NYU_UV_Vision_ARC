# Selected target message consumed by the processor and solver.
uint8 DETECTING=0
uint8 TRACKING=1
uint8 COASTING=2

std_msgs/Header header
# Camera/backend source id inherited from upstream messages.
string source_id
# True when the remaining fields describe a selected target.
bool has_target
# True when the selected target is currently predicted rather than measured.
bool is_coasting
# Stable tracker id for this target, or -1 when unavailable.
int32 track_id
# Bounding box in source-image pixel coordinates: xmin, ymin, xmax, ymax.
int32[4] bbox_xyxy
# Measured target center in source-image pixels.
float32 center_x
float32 center_y
# Predicted target center in source-image pixels.
float32 predicted_center_x
float32 predicted_center_y
# Confidence copied from the selected detector observation.
float32 confidence
# Model class id for the selected armor.
int32 class_id
# True when the target matches the current enemy color.
bool is_enemy
# True when obb_angle contains a valid oriented-box angle.
bool has_obb_angle
# Oriented bounding-box angle in radians.
float32 obb_angle
# Track state: DETECTING, TRACKING, or COASTING.
uint8 state
# Source image dimensions used by 2D solvers.
int32 image_width
int32 image_height
# Legacy compatibility flag equivalent to a confirmed tracking state.
bool tracking
# Fire recommendation inherited or refined by downstream processing.
bool suggest_fire
# True when pose and range are valid.
bool has_pose
# Target pose in the configured processor frame.
geometry_msgs/Pose pose
# True when velocity contains a valid estimate.
bool has_velocity
# Target velocity in the configured processor frame.
geometry_msgs/Vector3 velocity
# Target range in meters.
float32 range
