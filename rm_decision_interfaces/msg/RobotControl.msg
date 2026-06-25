# Robot control (gimbal, chassis)
# legacy compatibility fields
bool stop_gimbal_scan # Legacy: 0 keep scanning, 1 stop scan and hand off to auto-aim
float32 chassis_spin_vel # Chassis "spin" yaw rate (rad/s)

# Sentry control fields
bool scan_enabled # Enable gimbal scan pattern
bool allow_vision_control # Allow vision auto-aim to take gimbal
bool search_when_target_lost # When auto-aim has no target, return to scan search
float32 scan_yaw_rate_deg_s # Gimbal yaw rate while scanning (deg/s)
float32 search_pitch_deg # Gimbal pitch setpoint in scan/search (deg)
