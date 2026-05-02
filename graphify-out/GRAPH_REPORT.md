# Graph Report - .  (2026-05-01)

## Corpus Check
- 124 files · ~145,174 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 749 nodes · 1160 edges · 70 communities detected
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 256 edges (avg confidence: 0.78)
- Token cost: 3,250 input · 1,500 output

## Community Hubs (Navigation)
- [[_COMMUNITY_.send Main()|.send Main()]]
- [[_COMMUNITY_Init Resolve|Init Resolve]]
- [[_COMMUNITY_.get The|.get The]]
- [[_COMMUNITY_The Bounding|The Bounding]]
- [[_COMMUNITY_Target() Callback()|Target() Callback()]]
- [[_COMMUNITY_Run Center()|Run Center()]]
- [[_COMMUNITY_Log Runtime|Log Runtime]]
- [[_COMMUNITY_Value() Engine|Value() Engine]]
- [[_COMMUNITY_Log Runtime|Log Runtime]]
- [[_COMMUNITY_Log Handle|Log Handle]]
- [[_COMMUNITY_Main Detector|Main Detector]]
- [[_COMMUNITY_Bits The|Bits The]]
- [[_COMMUNITY_Plate Load|Plate Load]]
- [[_COMMUNITY_Camera The|Camera The]]
- [[_COMMUNITY_Armor Make|Armor Make]]
- [[_COMMUNITY_Merge Xywh|Merge Xywh]]
- [[_COMMUNITY_The Checks|The Checks]]
- [[_COMMUNITY_.get Todo|.get Todo:]]
- [[_COMMUNITY_Resolve Data|Resolve Data]]
- [[_COMMUNITY_Resolve Run|Resolve Run]]
- [[_COMMUNITY_Resolve Run|Resolve Run]]
- [[_COMMUNITY_Jetson Developer|Jetson Developer]]
- [[_COMMUNITY_Camera Camera|Camera Camera]]
- [[_COMMUNITY_Generate Launch|Generate Launch]]
- [[_COMMUNITY_Deepstreambackend Detectorbackend|Deepstreambackend Detectorbackend]]
- [[_COMMUNITY_Export Mac|Export Mac]]
- [[_COMMUNITY_Armordetectornode Armor|Armordetectornode Armor]]
- [[_COMMUNITY_Camerabackend .camerabackend()|Camerabackend .camerabackend()]]
- [[_COMMUNITY_Kalman) Baslertrack|Kalman) Baslertrack]]
- [[_COMMUNITY_For Fast-lio2|For Fast-lio2]]
- [[_COMMUNITY_Target Extended|Target Extended]]
- [[_COMMUNITY_Deepstreamdetector Logger|Deepstreamdetector Logger]]
- [[_COMMUNITY_Stm32 Serial|Stm32 Serial]]
- [[_COMMUNITY_Detectfiducial Findmarker|Detectfiducial Findmarker]]
- [[_COMMUNITY_Old Utils|Old Utils]]
- [[_COMMUNITY_Libnvvpi3 Dockerfile.realsense|Libnvvpi3 Dockerfile.realsense]]
- [[_COMMUNITY_Intra-process Zero-copy|Intra-process Zero-copy]]
- [[_COMMUNITY_For Singularity|For Singularity]]
- [[_COMMUNITY_Image Stream|Image Stream]]
- [[_COMMUNITY_Deepstreamruntime Deepstream|Deepstreamruntime Deepstream]]
- [[_COMMUNITY_Train Ultralytics.py|Train Ultralytics.py]]
- [[_COMMUNITY_The Fid|The Fid]]
- [[_COMMUNITY_Legacy Config|Legacy Config]]
- [[_COMMUNITY_Trackedrobot Tracker|Trackedrobot Tracker]]
- [[_COMMUNITY_Run Bytetrack|Run Bytetrack]]
- [[_COMMUNITY_Python Bottlenecks|Python Bottlenecks:]]
- [[_COMMUNITY_Jetson Serial|Jetson Serial]]
- [[_COMMUNITY_Logged Hyperparameters|Logged Hyperparameters]]
- [[_COMMUNITY_Node Aim|Node Aim]]
- [[_COMMUNITY_Tracker Armor|Tracker Armor]]
- [[_COMMUNITY_Node Camera|Node Camera]]
- [[_COMMUNITY_Deepstream Detector|Deepstream Detector]]
- [[_COMMUNITY_System Architecture|System Architecture]]
- [[_COMMUNITY_Engine Video|Engine Video]]
- [[_COMMUNITY_Merge Datasets|Merge Datasets]]
- [[_COMMUNITY_Ultralytics Train|Ultralytics Train]]
- [[_COMMUNITY_Distance Mapper|Distance Mapper]]
- [[_COMMUNITY_Imu Sensor|Imu Sensor]]
- [[_COMMUNITY_Iou|Iou]]
- [[_COMMUNITY_Hsv Trackbar|Hsv Trackbar]]
- [[_COMMUNITY_Uart Server|Uart Server]]
- [[_COMMUNITY_Reading Uart|Reading Uart]]
- [[_COMMUNITY_Arm Nav|Arm Nav]]
- [[_COMMUNITY_Legacy Viam|Legacy Viam]]
- [[_COMMUNITY_Pre-commit For|Pre-commit For]]
- [[_COMMUNITY_Pyrealsense Build|Pyrealsense Build:]]
- [[_COMMUNITY_Macrequirements Coreml,|Mac/requirements: Coreml,]]
- [[_COMMUNITY_Ros2 Node|Ros2 Node]]
- [[_COMMUNITY_Pipeline Assets|Pipeline Assets]]
- [[_COMMUNITY_Pipeline Diagram|Pipeline Diagram]]

## God Nodes (most connected - your core abstractions)
1. `Sort` - 23 edges
2. `Uart` - 20 edges
3. `Publisher_node` - 20 edges
4. `SentryDriver` - 18 edges
5. `run_tracking()` - 18 edges
6. `process_detection_cycle()` - 16 edges
7. `run_tracking()` - 16 edges
8. `main()` - 14 edges
9. `update()` - 13 edges
10. `infer()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `mac quantize (YOLO export engine/coreml)` --semantically_similar_to--> `export_int8_engine()`  [INFERRED] [semantically similar]
  mac/quantize.py → model_test/quantize.py
- `choose_selected_track_id()` --calls--> `Sort`  [INFERRED]
  yolo_cpp/src/tracker.cpp → legacy/utils/tracking/sort.py
- `rotated_nms()` --calls--> `Sort`  [INFERRED]
  yolo_cpp/src/detection_postprocess.cpp → legacy/utils/tracking/sort.py
- `axis_aligned_nms()` --calls--> `Sort`  [INFERRED]
  yolo_cpp/src/detection_postprocess.cpp → legacy/utils/tracking/sort.py
- `extract_candidates()` --calls--> `Sort`  [INFERRED]
  yolo_cpp/src/perception_node.cpp → legacy/utils/tracking/sort.py

## Hyperedges (group relationships)
- **Fiducial detection and field localization** — legacy_utils_fiducial_detection_fiducial_detector, legacy_utils_fiducial_detection_findmarker, legacy_utils_fid_locs_locs, concept_robomaster_fiducial_field [INFERRED 0.86]
- **Legacy Jetson auto-aim UART stack** — legacy_utils_plate_detector_plate_detector, legacy_utils_uart_comms_publisher_node, legacy_utils_uart_server_uart, legacy_utils_uart_comms_send_auto_aim, concept_serial_auto_aim_protocol [INFERRED 0.83]
- **macOS/YOLO inference and export tooling** — mac_baslertrack_main, mac_bytetrackoptim2_pipeline, mac_quantize_export, mac_bare_main, model_test_quantize_export_int8_engine [INFERRED 0.72]
- **FAST-LIO2 field localization constraints** — nav_26navplan_fast_lio2_mid360, nav_26navplan_pcd_localization_not_slam, nav_26navplan_lid_height_dynamic_costmap [EXTRACTED 0.90]
- **JetPack SD image acquisition and media prep** — downloads_installing_jetpack_page, downloads_xavier_nx_sd_image, downloads_nano_sd_image, downloads_nano_2gb_sd_image, sd_format_windows_formatter_app [INFERRED 0.78]
- **RoboMaster Perception Pipeline** — camera_node, armor_detector_node, armor_tracker_node [INFERRED 0.85]
- **Primary Perception Pipeline** — cameranode_camera, yolonode_yolo, armortrackernode_armor_tracker, aimsolvernode_aim_solver [INFERRED 0.88]
- **DeepStream Backend Orchestration** — armor_detector_node_class, detector_backend_class, deepstream_runtime_class [INFERRED 0.85]
- **Pipeline Execution Flow** — target_classification_3d_estimation, extended_kalman_filter, target_selector_trajectory_predictor [EXTRACTED 1.00]
- **RealSense Data Outputs** — intel_realsense_camera, realsense_rgb_stream, realsense_depth_stream [INFERRED 0.80]

## Communities

### Community 0 - ".send Main()"
Cohesion: 0.04
Nodes (15): Armor Detector Node, Base, Perception Node, main(), Get the base width and turning radius          Returns:             Propertie, SentryDriver, main(), main() (+7 more)

### Community 1 - "Init Resolve"
Cohesion: 0.06
Nodes (42): BaslerInferenceThread, _disable_frame_rate_limit(), _load_model_threaded(), Grab BGR frames from Basler, run YOLO per frame, queue latest Results (size 1)., Load YOLO on a worker thread; main thread must pump cv2 (macOS WindowServer watc, _run_coreml_warmup(), run_tracking(), _warmup_threaded_with_gui_pump() (+34 more)

### Community 2 - ".get The"
Cohesion: 0.06
Nodes (16): det_move_(), main(), red_or_blue(), send_cords(), DepthCamera, process_frame(), show_frame(), uart_server (+8 more)

### Community 3 - "The Bounding"
Cohesion: 0.05
Nodes (31): KalmanBoxTracker, This class represents the internel state of individual tracked objects observed, Initialises a tracker using initial bounding box., Updates the state vector with observed bbox., Returns the current bounding box estimate., SORT: A Simple, Online and Realtime Tracker     Copyright (C) 2016 Alex Bewley a, Sets key parameters for SORT, Params:           dets - a numpy array of detections in the format [[x1,y1,x2,y2 (+23 more)

### Community 4 - "Target() Callback()"
Cohesion: 0.09
Nodes (30): Deepstream Detector, handle_target(), publish_no_target(), rect_from_xyxy(), id_listener_callback(), id_listener_callback(), DeepStreamDetector(), id_listener_callback() (+22 more)

### Community 5 - "Run Center()"
Cohesion: 0.11
Nodes (29): run_coreml_probe_worker(), clamp_bbox(), run_kalman_pipeline(), ArmorPlateTracker(), choose_selected_track_id(), clamp_center(), height(), is_selectable() (+21 more)

### Community 6 - "Log Runtime"
Cohesion: 0.09
Nodes (30): ArmorDetectorNode(), extract_candidates(), frame_to_image_message(), handle_attitude_flags(), handle_deepstream_result(), initialize_deepstream_runtime(), initialize_native_pipeline(), log_runtime_fps_once_per_second() (+22 more)

### Community 7 - "Value() Engine"
Cohesion: 0.1
Nodes (28): axis_aligned_iou(), axis_aligned_nms(), clamp_value(), engine_plan_offset(), extract_engine_metadata(), extract_json_bool_value(), extract_json_string_value(), extract_names() (+20 more)

### Community 8 - "Log Runtime"
Cohesion: 0.16
Nodes (25): extract_candidates(), frame_to_image_message(), handle_attitude_flags(), handle_deepstream_result(), initialize_deepstream_runtime(), log_runtime_fps_once_per_second(), log_runtime_note(), log_runtime_opponent_event() (+17 more)

### Community 9 - "Log Handle"
Cohesion: 0.13
Nodes (22): Seasky Protocol, append_u16(), build_seasky_frame(), build_vision_flags(), compute_crc16(), compute_crc8(), parse_attitude_flags(), parse_attitude_frames() (+14 more)

### Community 10 - "Main Detector"
Cohesion: 0.09
Nodes (26): RoboMaster fiducial localization, Serial auto-aim / chassis protocol (0xE7 header), rpc server main (asyncio), SentryDriver (Viam Base + UART), camera (cv2 VideoCapture), depth_camera (RealSense), classic_plate_detector (HSV blue contours), fid_locs.locs (field marker poses) (+18 more)

### Community 11 - "Bits The"
Cohesion: 0.15
Nodes (14): Fiducial_detector, get_contour_bits(), getMarkerMatch(), getPixelLen_Corners(), getRes(), load_marker_dictionary(), loadMarkers(), matchMarker() (+6 more)

### Community 12 - "Plate Load"
Cohesion: 0.17
Nodes (19): apply_plate_hue_bias_bgr(), create_frame_converter(), load_model(), load_runtime_model(), main(), open_default_camera(), parse_plate_target(), In-place BGR channel nudge — minimal cost vs full color space conversion. (+11 more)

### Community 13 - "Camera The"
Cohesion: 0.11
Nodes (10): camera, depth_camera, imu_sensor, Extracts depth and color frame from camera feed.         Returns:         ret =, Stops the capture pipeline., # TODO: potentially return time data along with accel/gyro value, Constructor.         Attributes:             cap = A VideoCapture object of cv2, Obtains a frame from the camera and returns it.         Returns:         ret = T (+2 more)

### Community 14 - "Armor Make"
Cohesion: 0.22
Nodes (11): Armor Tracker Node, armors_callback(), ArmorTrackerNode(), log_stage_timing(), main(), make_armor_message(), make_track_observation(), publish_target() (+3 more)

### Community 15 - "Merge Xywh"
Cohesion: 0.27
Nodes (11): convert_xywh_to_obb(), ensure_dir(), load_yaml(), merge_datasets(), process_dataset(), Main function to control the merge process., Creates a directory if it doesn't exist., Converts a mixed list of class names and IDs into a set of class IDs     valid f (+3 more)

### Community 16 - "The Checks"
Cohesion: 0.21
Nodes (7): Flagger, Checks what is the color of the robot in the image using cuda.     Parameters:, Checks what is the color of the robot in the image.     Parameters:     frame =, Constructor.         opponent_color: 'RED' or "BLUE' indicating the color of the, Checks if the frame contains the robot of enemy team or ally.         Paramter:, red_or_blue(), red_or_blue_cuda()

### Community 17 - ".get Todo:"
Cohesion: 0.21
Nodes (4): # TODO: add more methods for getting position in a useful format,ex: getting pos, # TODO: need to think of logic needed of when we want to track, forget, etc. rob, TrackedRobot, Tracker

### Community 18 - "Resolve Data"
Cohesion: 0.36
Nodes (10): build_local_data_yaml(), count_images(), export_model(), main(), parse_args(), remove_calibration_cache(), resolve_data_path(), resolve_export_target() (+2 more)

### Community 19 - "Resolve Run"
Cohesion: 0.46
Nodes (7): main(), parse_args(), resolve_device(), resolve_model_path(), resolve_source(), run_tracking(), stream_tracks()

### Community 20 - "Resolve Run"
Cohesion: 0.46
Nodes (7): main(), parse_args(), resolve_device(), resolve_model_path(), resolve_source(), run_tracking(), stream_tracks()

### Community 21 - "Jetson Developer"
Cohesion: 0.25
Nodes (8): Installing JetPack — SD card image method (NVIDIA docs), Jetson Nano 2GB Developer Kit — download SD card image, Jetson Nano Developer Kit — download SD card image, Jetson Xavier NX Developer Kit — download SD card image, Jetson Nano Developer Kit — DC jack circled, heatsink, USB, Ethernet, CSI, SD Card Formatter (Windows) — select drive, quick/overwrite format, Jetson module developer kit photo (USB-C and I/O row highlighted), Jetson Xavier NX Developer Kit — heatsink, fan, ports, GPIO, CSI

### Community 22 - "Camera Camera"
Cohesion: 0.38
Nodes (5): Camera Node, main(), publish_no_frame_warning(), timer_callback(), to_image_message()

### Community 23 - "Generate Launch"
Cohesion: 0.33
Nodes (3): generate_launch_description(), generate_launch_description(), Node()

### Community 24 - "Deepstreambackend Detectorbackend"
Cohesion: 0.4
Nodes (3): DeepStreamBackend, DetectorBackend, NativeTensorRTBackend

### Community 25 - "Export Mac"
Cohesion: 0.6
Nodes (4): mac quantize (YOLO export engine/coreml), clean_export_artifacts(), export_int8_engine(), main()

### Community 26 - "Armordetectornode Armor"
Cohesion: 0.4
Nodes (4): ArmorDetectorNode, DeepStream Configuration, DeepstreamRuntime, DetectorBackend

### Community 27 - "Camerabackend .camerabackend()"
Cohesion: 0.5
Nodes (2): CameraBackend, Node

### Community 28 - "Kalman) Baslertrack"
Cohesion: 0.5
Nodes (4): baslertrack (Basler + YOLO + Kalman), ArmorPlateTracker import (yolov8_ros), bytetrackOptim2 (YOLO + custom Kalman), bytetrackOptim (BoT-SORT style)

### Community 29 - "For Fast-lio2"
Cohesion: 0.5
Nodes (4): FAST-LIO2 + Livox Mid-360 on CV Docker base, Z-slice lidar height; local costmap for dynamic obstacles, Orin: MAXN/jetson_clocks; eth MTU; swap for point cloud memory, Prebuilt field PCD for localization vs runtime SLAM

### Community 30 - "Target Extended"
Cohesion: 0.83
Nodes (4): Extended Kalman Filter, Pipeline Steps 3, 4, 5 Diagram, Target Classification & 3D Estimation, Target Selector & Trajectory Predictor

### Community 31 - "Deepstreamdetector Logger"
Cohesion: 0.67
Nodes (2): DeepStreamDetector, Logger

### Community 32 - "Stm32 Serial"
Cohesion: 0.67
Nodes (1): SerialTransport

### Community 33 - "Detectfiducial Findmarker"
Cohesion: 0.67
Nodes (3): detectFiducial, findMarker, matchMarker

### Community 34 - "Old Utils"
Cohesion: 0.67
Nodes (3): old_utils main (RealSense + UART), DepthCamera (pyrealsense2), process_frame (median depth in bbox)

### Community 35 - "Libnvvpi3 Dockerfile.realsense"
Cohesion: 0.67
Nodes (3): Dockerfile.realsense installs libnvvpi3 from Jetson apt (baked in image), JetPack r36 vs r35 and SOC t234/t194/t186 path selection, libnvvpi3 required for DeepStream 6.4

### Community 36 - "Intra-process Zero-copy"
Cohesion: 0.67
Nodes (3): ros2 topic info intra-process subscription count validates zero-copy, ComposableNodeContainer for intra-process zero-copy between nodes, Decouple camera into standalone camera_node publishing image_raw

### Community 37 - "For Singularity"
Cohesion: 0.67
Nodes (3): Singularity overlay rw for installs, ro for run to avoid wrong persistence, Avoid pip/cache under home; redirect wandb cache to scratch, Weights & Biases for multi-experiment comparison on cluster

### Community 38 - "Image Stream"
Cohesion: 0.67
Nodes (3): Intel RealSense Camera, Depth Image Stream, RGB Image Stream

### Community 41 - "Deepstreamruntime Deepstream"
Cohesion: 1.0
Nodes (1): DeepStreamRuntime

### Community 45 - "Train Ultralytics.py"
Cohesion: 1.0
Nodes (1): make sure data is in right format, i am training yolov8 so i downloaded in yolov

### Community 46 - "The Fid"
Cohesion: 1.0
Nodes (1): Marker locations in the real orld coordinate system Tx, Ty, Tz is the location o

### Community 48 - "Legacy Config"
Cohesion: 1.0
Nodes (2): legacy config dicts camera/yolo/gameplay, VideoRecord threaded mp4 batch save

### Community 49 - "Trackedrobot Tracker"
Cohesion: 1.0
Nodes (2): TrackedRobot, Tracker (deprecated robot tracking)

### Community 50 - "Run Bytetrack"
Cohesion: 1.0
Nodes (2): run_bytetrack copy (duplicate runner), run_bytetrack (ultralytics ByteTrack)

### Community 51 - "Python Bottlenecks:"
Cohesion: 1.0
Nodes (2): Python bottlenecks: copies, GIL, rclpy vs rclcpp, Proposed yolov8_ros_cpp: TensorRT + Pylon C++ + Kalman

### Community 52 - "Jetson Serial"
Cohesion: 1.0
Nodes (2): Serial USB vs WiFi SSH trade-offs for Jetson, Flash Jetson from SD image; do not use SDK Manager per doc

### Community 53 - "Logged Hyperparameters"
Cohesion: 1.0
Nodes (2): Logged hyperparameters and config columns on runs, W&B Runs table (13 runs): states, batch, copy_p, cos_lr, data path, epochs

### Community 54 - "Node Aim"
Cohesion: 1.0
Nodes (2): Aim Solver Node, STM32 Serial Bridge Node

### Community 55 - "Tracker Armor"
Cohesion: 1.0
Nodes (2): Armor Tracker Node, Tracker

### Community 56 - "Node Camera"
Cohesion: 1.0
Nodes (2): Camera Node, Perception Node

### Community 57 - "Deepstream Detector"
Cohesion: 1.0
Nodes (2): Deepstream Detector, YOLO Node

### Community 58 - "System Architecture"
Cohesion: 1.0
Nodes (2): CV System Architecture, Computer Vision Pipeline Flow Diagram

### Community 67 - "Engine Video"
Cohesion: 1.0
Nodes (1): Engine video predict smoke test

### Community 68 - "Merge Datasets"
Cohesion: 1.0
Nodes (1): merge_datasets OBB remap utility

### Community 69 - "Ultralytics Train"
Cohesion: 1.0
Nodes (1): Ultralytics train + W&B callback

### Community 70 - "Distance Mapper"
Cohesion: 1.0
Nodes (1): distance_mapper ultrasonic quadrants

### Community 71 - "Imu Sensor"
Cohesion: 1.0
Nodes (1): imu_sensor (RealSense IMU)

### Community 72 - "Iou"
Cohesion: 1.0
Nodes (1): iou

### Community 73 - "Hsv Trackbar"
Cohesion: 1.0
Nodes (1): HSV trackbar tuning tool

### Community 74 - "Uart Server"
Cohesion: 1.0
Nodes (1): uart_server (Jetson THS1)

### Community 75 - "Reading Uart"
Cohesion: 1.0
Nodes (1): reading_uart serial echo loop

### Community 76 - "Arm Nav"
Cohesion: 1.0
Nodes (1): ARM nav container + TigerVNC/noVNC for RViz2 in browser

### Community 77 - "Legacy Viam"
Cohesion: 1.0
Nodes (1): Legacy Viam nanoviam Dockerfile build/run

### Community 78 - "Pre-commit For"
Cohesion: 1.0
Nodes (1): pre-commit for lint/format before review; commitlint on commit-msg

### Community 79 - "Pyrealsense Build:"
Cohesion: 1.0
Nodes (1): PyRealsense build: CUDA env, FORCE_RSUSB, Python bindings

### Community 80 - "Mac/requirements: Coreml,"
Cohesion: 1.0
Nodes (1): mac/requirements: CoreML, numpy pin, opencv, pypylon, ultralytics

### Community 81 - "Ros2 Node"
Cohesion: 1.0
Nodes (1): ROS2 Node Concept

### Community 82 - "Pipeline Assets"
Cohesion: 1.0
Nodes (1): Pipeline Assets 6-7

### Community 83 - "Pipeline Diagram"
Cohesion: 1.0
Nodes (1): Pipeline Diagram

## Ambiguous Edges - Review These
- `Jetson module developer kit photo (USB-C and I/O row highlighted)` → `Jetson Nano 2GB Developer Kit — download SD card image`  [AMBIGUOUS]
  docs/Setup/Assets/Downloads.png · relation: references

## Knowledge Gaps
- **129 isolated node(s):** `DeepStreamRuntime`, `NativeTensorRTBackend`, `DeepStreamBackend`, `Logger`, `DeepStreamDetector` (+124 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Camerabackend .camerabackend()`** (4 nodes): `CameraBackend`, `.CameraBackend()`, `Node`, `camera_backend.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deepstreamdetector Logger`** (3 nodes): `DeepStreamDetector`, `Logger`, `deepstream_detector.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Stm32 Serial`** (3 nodes): `stm32_serial_bridge_node.hpp`, `Node()`, `SerialTransport`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deepstreamruntime Deepstream`** (2 nodes): `DeepStreamRuntime`, `deepstream_runtime.hpp`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Train Ultralytics.py`** (2 nodes): `train_ultralytics.py`, `make sure data is in right format, i am training yolov8 so i downloaded in yolov`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `The Fid`** (2 nodes): `fid_locs.py`, `Marker locations in the real orld coordinate system Tx, Ty, Tz is the location o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Legacy Config`** (2 nodes): `legacy config dicts camera/yolo/gameplay`, `VideoRecord threaded mp4 batch save`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Trackedrobot Tracker`** (2 nodes): `TrackedRobot`, `Tracker (deprecated robot tracking)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Run Bytetrack`** (2 nodes): `run_bytetrack copy (duplicate runner)`, `run_bytetrack (ultralytics ByteTrack)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Python Bottlenecks:`** (2 nodes): `Python bottlenecks: copies, GIL, rclpy vs rclcpp`, `Proposed yolov8_ros_cpp: TensorRT + Pylon C++ + Kalman`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Jetson Serial`** (2 nodes): `Serial USB vs WiFi SSH trade-offs for Jetson`, `Flash Jetson from SD image; do not use SDK Manager per doc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Logged Hyperparameters`** (2 nodes): `Logged hyperparameters and config columns on runs`, `W&B Runs table (13 runs): states, batch, copy_p, cos_lr, data path, epochs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Node Aim`** (2 nodes): `Aim Solver Node`, `STM32 Serial Bridge Node`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tracker Armor`** (2 nodes): `Armor Tracker Node`, `Tracker`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Node Camera`** (2 nodes): `Camera Node`, `Perception Node`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Deepstream Detector`** (2 nodes): `Deepstream Detector`, `YOLO Node`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `System Architecture`** (2 nodes): `CV System Architecture`, `Computer Vision Pipeline Flow Diagram`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Engine Video`** (1 nodes): `Engine video predict smoke test`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Merge Datasets`** (1 nodes): `merge_datasets OBB remap utility`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ultralytics Train`** (1 nodes): `Ultralytics train + W&B callback`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Distance Mapper`** (1 nodes): `distance_mapper ultrasonic quadrants`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Imu Sensor`** (1 nodes): `imu_sensor (RealSense IMU)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Iou`** (1 nodes): `iou`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Hsv Trackbar`** (1 nodes): `HSV trackbar tuning tool`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Uart Server`** (1 nodes): `uart_server (Jetson THS1)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Reading Uart`** (1 nodes): `reading_uart serial echo loop`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Arm Nav`** (1 nodes): `ARM nav container + TigerVNC/noVNC for RViz2 in browser`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Legacy Viam`** (1 nodes): `Legacy Viam nanoviam Dockerfile build/run`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pre-commit For`** (1 nodes): `pre-commit for lint/format before review; commitlint on commit-msg`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pyrealsense Build:`** (1 nodes): `PyRealsense build: CUDA env, FORCE_RSUSB, Python bindings`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Mac/requirements: Coreml,`** (1 nodes): `mac/requirements: CoreML, numpy pin, opencv, pypylon, ultralytics`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Ros2 Node`** (1 nodes): `ROS2 Node Concept`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pipeline Assets`** (1 nodes): `Pipeline Assets 6-7`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pipeline Diagram`** (1 nodes): `Pipeline Diagram`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Jetson module developer kit photo (USB-C and I/O row highlighted)` and `Jetson Nano 2GB Developer Kit — download SD card image`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `Sort` connect `The Bounding` to `Target() Callback()`, `Run Center()`, `Log Runtime`, `Value() Engine`, `Log Runtime`, `Bits The`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `main()` connect `.get The` to `The Checks`, `The Bounding`, `Camera The`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Why does `Publisher_node` connect `.send Main()` to `The Bounding`, `.get The`, `Bits The`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Are the 20 inferred relationships involving `Sort` (e.g. with `plate_detector` and `Constructor. Initializes attributes. Uses torch.hub.load() of PyTorch module to`) actually correct?**
  _`Sort` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `Uart` (e.g. with `.__init__()` and `._start()`) actually correct?**
  _`Uart` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `Publisher_node` (e.g. with `UartDistance` and `Fiducial_detector`) actually correct?**
  _`Publisher_node` has 4 INFERRED edges - model-reasoned connections that need verification._