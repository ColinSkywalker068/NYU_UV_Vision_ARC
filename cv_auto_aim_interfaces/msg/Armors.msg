# Detector output for one image frame.
std_msgs/Header header
# Camera/backend source id that produced this frame.
string source_id
# Armor candidates that passed detector and color filters.
cv_auto_aim_interfaces/Armor[] armors
