# Basler SP/ST Bringup

Related pages: [Overview](./overview.md) | [Pipeline](./pipeline.md) | [Launch and Config](./launch-and-config.md) | [Development Notes](./development-notes.md)

Use this path for the current Basler robot stack:

```text
Basler camera
-> armor_detector_node
-> armor_tracker_node
-> armor_processor_node
-> aim_solver_node
-> stm32_serial_bridge_node
-> SP/ST Type-C protocol
```

Do not add the sentry adapter, old `rm_serial_driver`, or `nyush-rm-vision` to this path. Those are different integration routes and make serial ownership/protocol debugging much harder.

## Camera and Inference Only

Use `transmit_enable:=false` when the Type-C board is absent or the robot is not in a safe bench state:

```bash
ros2 launch yolo_cpp basler.launch.py \
  target_color:=red \
  show:=web \
  transmit_enable:=false
```

This proves Basler capture, the model, detector output, tracker selection, processor output, and aim solution topics. It does not open the serial device.

## Direct Type-C Smoke Test

For a safe protocol smoke test with only the Type-C board attached:

```bash
ros2 launch yolo_cpp basler.launch.py \
  target_color:=red \
  serial_port:=/dev/ttyACM0 \
  serial_verbose:=true \
  transmit_enable:=true \
  show:=web
```

The active bridge sends raw `SP` frames:

```text
S P mode yaw yaw_vel yaw_acc pitch pitch_vel pitch_acc crc16
```

Yaw and pitch are radians by default. Keep `transmit_angles_in_degrees:=false` unless the firmware contract changes.

## Manual SP/ST Serial Probe

Use the shell helper when you want to test the serial device without camera inference:

```bash
./yolo_cpp/tools/hardware_serial_smoke.sh --port /dev/ttyACM0
```

The default command sends `mode=0` with zero yaw/pitch. To send a small aim command:

```bash
./yolo_cpp/tools/hardware_serial_smoke.sh \
  --port /dev/ttyACM0 \
  --mode aim \
  --yaw-rad 0.05
```

The helper prints the exact 29-byte `SP` frame it sent and decodes basic incoming `SP` / `ST` frames if the board replies.
