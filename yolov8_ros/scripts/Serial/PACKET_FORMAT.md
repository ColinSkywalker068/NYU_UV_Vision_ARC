# STM32 Seasky Packet Format

The active controls firmware uses the Seasky framing implemented in
`yolov8_ros/yolov8_ros/stm32_serial_bridge.py`. The older `0xE7` micro-radian
packet is not the current STM32 contract.

## CV to STM32: Vision Command

CV sends Seasky command `0x0001` as an 18-byte frame:

| Bytes | Field | Encoding |
|-------|-------|----------|
| 0 | SOF | `0xA5` |
| 1-2 | Data length | Little-endian `uint16`, value `12` |
| 3 | Header CRC8 | CRC over bytes 0-2 |
| 4-5 | Command id | Little-endian `uint16`, value `0x0001` |
| 6-7 | Flags | Little-endian `uint16` |
| 8-11 | Pitch error | Little-endian `float32`, radians |
| 12-15 | Yaw error | Little-endian `float32`, radians |
| 16-17 | Frame CRC16 | CRC over bytes 0-15 |

Flags for command `0x0001`:

| Bits | Field |
|------|-------|
| 0-1 | Fire mode: `0=NO_FIRE`, `1=AUTO_FIRE`, `2=AUTO_AIM` |
| 2-3 | Target state: `0=NO_TARGET`, `1=TARGET_CONVERGING`, `2=READY_TO_FIRE` |
| 4-7 | Target type |

Current controls uses `target_state != 0` as the gimbal validity gate. `fire_mode`
and `target_type` are logged/HUD metadata in the current firmware and do not fire
the shooter by themselves.

Example yaw-only packet (`pitch=0`, `yaw=0.05 rad`, `AUTO_AIM`,
`TARGET_CONVERGING`, target type `0`):

```text
a5 0c 00 3c 01 00 06 00 00 00 00 00 cd cc 4c 3d 64 83
```

## STM32 to CV: Attitude Frame

STM32 sends Seasky command `0x0002` with three little-endian `float32` attitude
values:

| Payload order | Unit sent by STM32 | Unit published by CV bridge |
|---------------|--------------------|------------------------------|
| Yaw | degrees | radians |
| Pitch | degrees | radians |
| Roll | degrees | radians |

Flags for command `0x0002` are packed by controls as:

| Bits | Field |
|------|-------|
| 0 | Enemy color |
| 1-2 | Work mode |
| 8-15 | Bullet speed |

The current controls code masks enemy color to one bit, so CV should not rely on
that field to distinguish all enum values.
