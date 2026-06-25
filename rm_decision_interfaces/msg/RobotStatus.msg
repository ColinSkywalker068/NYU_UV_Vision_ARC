# v1.6.1 0x0201 Robot performance data
uint8 robot_id # Robot ID: 1~7 red, 101~107 blue
uint16 current_hp # Hit points

# v1.6.1 0x0202 Real-time power / heat
uint16 shooter_heat # Barrel heat

# Processed on the embedded side
bool team_color # Team color: 0 red, 1 blue
bool is_attacked # Under attack: 0 no, 1 yes
