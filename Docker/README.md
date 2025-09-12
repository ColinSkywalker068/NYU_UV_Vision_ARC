# How to Run the Object Detection Pipeline
https://limewire.com/d/Ha8tz#M7d4qSlJjH

Shortcut:
Instead of two windows, you can run with just one:
Open a window, run `uv update`
Then run:
```
uv docker
uv build yolov8_ros launch.py
uv run yolov8_ros launch.py
```
This runs both nodes using the launch file in /yolov8_ros/launch/launch.py
Unlike the manual run, the arguments have to be configured directly in launch, 
so make sure to check if it is the right configuration.
I would recommend running the two terminal window method first to understand the parameters, before using the launch.

You will need two terminal windows:
In either of the windows, run `uv update` to make sure the Jetson software is up to date. 
In the first run the yolov8_node:
```
uv docker
uv build yolov8_ros yolov8_node --ros-args -r video_frames:=/camera/camera/color/image_raw -p model_path:='/home/ultraviolet/CV_Detection/yolov8_ros/models/best.pt' -p task:='obb' -p show:=False 
```
Important Stuff:
1. model_path must be the exact path to the model you want to test. Example Path is '/home/ultraviolet/CV_Detection/yolov8_ros/yolov8_ros/best.pt'
2. show:=True is to be set if you want a window with the realsense_camera feed and bounding boxes if they are detected. 
You will not see anything here if it is set to False, but there will be a text feed of frames, with bounding box info. 

In the second window run the realsense_node:
```
uv docker
ros2 launch realsense2_camera rs_launch.py rgb_camera.profile:=640x480x60
```
Important Stuff: 
Make sure the camera is connected properly when the command is run. 
Ideally you want to run this before running CV_Detection but it should still work in both orders. 


Camera Options for RealSense:
```
/camera/camera:
  .camera.color.image_raw.format
  .camera.color.image_raw.jpeg_quality
  .camera.color.image_raw.png_level
  .camera.color.image_raw.tiff.res_unit
  .camera.color.image_raw.tiff.xdpi
  .camera.color.image_raw.tiff.ydpi
  align_depth.enable
  align_depth.frames_queue_size
  angular_velocity_cov
  base_frame_id
  camera_name
  clip_distance
  color_info_qos
  color_qos
  colorizer.color_scheme
  colorizer.enable
  colorizer.frames_queue_size
  colorizer.histogram_equalization_enabled
  colorizer.max_distance
  colorizer.min_distance
  colorizer.stream_filter
  colorizer.stream_format_filter
  colorizer.stream_index_filter
  colorizer.visual_preset
  decimation_filter.enable
  decimation_filter.filter_magnitude
  decimation_filter.frames_queue_size
  decimation_filter.stream_filter
  decimation_filter.stream_format_filter
  decimation_filter.stream_index_filter
  depth_info_qos
  depth_module.auto_exposure_limit
  depth_module.auto_exposure_limit_toggle
  depth_module.auto_exposure_roi.bottom
  depth_module.auto_exposure_roi.left
  depth_module.auto_exposure_roi.right
  depth_module.auto_exposure_roi.top
  depth_module.auto_gain_limit
  depth_module.auto_gain_limit_toggle
  depth_module.depth_format
  depth_module.emitter_always_on
  depth_module.emitter_enabled
  depth_module.emitter_on_off
  depth_module.enable_auto_exposure
  depth_module.error_polling_enabled
  depth_module.exposure
  depth_module.frames_queue_size
  depth_module.gain
  depth_module.global_time_enabled
  depth_module.hdr_enabled
  depth_module.infra1_format
  depth_module.infra2_format
  depth_module.inter_cam_sync_mode
  depth_module.laser_power
  depth_module.output_trigger_enabled
  depth_module.profile
  depth_module.sequence_id
  depth_module.sequence_name
  depth_module.sequence_size
  depth_qos
  device_type
  diagnostics_period
  disparity_filter.enable
  disparity_to_depth.enable
  enable_color
  enable_depth
  enable_infra1
  enable_infra2
  enable_rgbd
  enable_sync
  filter_by_sequence_id.enable
  filter_by_sequence_id.frames_queue_size
  filter_by_sequence_id.sequence_id
  hdr_merge.enable
  hdr_merge.frames_queue_size
  hold_back_imu_for_frames
  hole_filling_filter.enable
  hole_filling_filter.frames_queue_size
  hole_filling_filter.holes_fill
  hole_filling_filter.stream_filter
  hole_filling_filter.stream_format_filter
  hole_filling_filter.stream_index_filter
  infra1_info_qos
  infra1_qos
  infra2_info_qos
  infra2_qos
  initial_reset
  json_file_path
  linear_accel_cov
  pointcloud.allow_no_texture_points
  pointcloud.enable
  pointcloud.filter_magnitude
  pointcloud.frames_queue_size
  pointcloud.ordered_pc
  pointcloud.pointcloud_qos
  pointcloud.stream_filter
  pointcloud.stream_format_filter
  pointcloud.stream_index_filter
  publish_tf
  qos_overrides./parameter_events.publisher.depth
  qos_overrides./parameter_events.publisher.durability
  qos_overrides./parameter_events.publisher.history
  qos_overrides./parameter_events.publisher.reliability
  reconnect_timeout
  rgb_camera.auto_exposure_priority
  rgb_camera.auto_exposure_roi.bottom
  rgb_camera.auto_exposure_roi.left
  rgb_camera.auto_exposure_roi.right
  rgb_camera.auto_exposure_roi.top
  rgb_camera.backlight_compensation
  rgb_camera.brightness
  rgb_camera.color_format
  rgb_camera.contrast
  rgb_camera.enable_auto_exposure
  rgb_camera.enable_auto_white_balance
  rgb_camera.exposure
  rgb_camera.frames_queue_size
  rgb_camera.gain
  rgb_camera.gamma
  rgb_camera.global_time_enabled
  rgb_camera.hue
  rgb_camera.power_line_frequency
  rgb_camera.profile
  rgb_camera.saturation
  rgb_camera.sharpness
  rgb_camera.white_balance
  rosbag_filename
  serial_no
  spatial_filter.enable
  spatial_filter.filter_magnitude
  spatial_filter.filter_smooth_alpha
  spatial_filter.filter_smooth_delta
  spatial_filter.frames_queue_size
  spatial_filter.holes_fill
  spatial_filter.stream_filter
  spatial_filter.stream_format_filter
  spatial_filter.stream_index_filter
  temporal_filter.enable
  temporal_filter.filter_smooth_alpha
  temporal_filter.filter_smooth_delta
  temporal_filter.frames_queue_size
  temporal_filter.holes_fill
  temporal_filter.stream_filter
  temporal_filter.stream_format_filter
  temporal_filter.stream_index_filter
  tf_publish_rate
  unite_imu_method
  usb_port_id
  use_sim_time
  wait_for_device_timeout
```