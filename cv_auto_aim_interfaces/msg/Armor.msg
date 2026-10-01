# Detector-level armor candidate before temporal tracking.
uint8 ARMOR_SMALL=0
uint8 ARMOR_LARGE=1

# Bounding box in source-image pixel coordinates: xmin, ymin, xmax, ymax.
int32[4] bbox_xyxy
# Candidate center in source-image pixels.
float32 center_x
float32 center_y
# Detector confidence after model score decoding.
float32 confidence
# Model class id, usually encoding target color/team.
int32 class_id
# True when the candidate matches the currently selected enemy color.
bool is_enemy
# True when obb_angle contains a valid oriented-box angle.
bool has_obb_angle
# Oriented bounding-box angle in radians.
float32 obb_angle
# Source image dimensions for consumers that need normalized offsets.
int32 image_width
int32 image_height
# True when corners contains valid oriented armor corners.
bool has_corners
# Armor corners in source-image pixel coordinates.
geometry_msgs/Point32[4] corners
# Armor size classification using ARMOR_SMALL or ARMOR_LARGE.
uint8 armor_type
# Recognized armor number, or -1 when unavailable.
int32 armor_number
# Pixel distance from the image center, used for target ranking.
float32 distance_to_image_center
# True when pose_camera contains a solved camera-frame pose.
bool has_pose
# Camera-frame armor pose from detector-side PnP.
geometry_msgs/Pose pose_camera

# Real model keypoints and pose-quality provenance; synthetic box corners are not pose measurements.
bool corners_are_keypoints
bool pose_normal_reliable
float32 reprojection_rms_px
