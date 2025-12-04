import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32, Int32MultiArray, Bool
from cv_bridge import CvBridge
from ultralytics import YOLO, RTDETR, NAS  # Ensure ultralytics is installed
import cv2
import numpy as np
import os
import time
import random  # For Kalman
from pypylon import pylon  # Import pypylon for Basler cameras
import traceback
import struct
import math
import sys
from pathlib import Path

# Add scripts directory to path for local imports
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from stm32_serial import STM32Serial

# Prevent memory corruption on Jetson with TensorRT
os.environ["MALLOC_CHECK_"] = "0"  # Disable glibc malloc checks
os.environ["PYTHONMALLOC"] = "malloc"  # Use system malloc
os.environ["GDK_SYNCHRONIZE"] = "1"  # If using GTK-based cv2.imshow concurrently

# Team and Class Definitions
RED_TEAM = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
BLUE_TEAM = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111)

RED_CLASS = 1
BLUE_CLASS = 0

# Color Ranges for HSV
BLUE_RANGES = [
    [np.array((100, 150, 150), np.uint8), np.array((128, 255, 255), np.uint8)]
]
RED_RANGES = [
    [np.array((0, 150, 150), np.uint8), np.array((10, 255, 255), np.uint8)],
    [np.array((170, 150, 150), np.uint8), np.array((180, 255, 255), np.uint8)],
]

# Bounding Box Drawing Parameters
BBOX_RAW_COLOR = (0, 255, 0)  # Green for raw YOLO detection
BBOX_KF_COLOR = (255, 0, 0)  # Blue for Kalman Filter prediction
BBOX_LINE_SIZE = 2

# This was in your original code but appeared unused.
# COLOR_THRESHOLD = 200

# ──────────────────────────────────────────────────────────────
# Kalman Filter Tracker Classes
# ──────────────────────────────────────────────────────────────
class ArmorPlateTracker:
    def __init__(
        self, initial_x, initial_y, width, height
    ):  # dt removed from constructor
        self.kf = cv2.KalmanFilter(
            6, 2
        )  # 6 states (x,y,vx,vy,ax,ay), 2 measurements (x,y)

        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32
        )

        # --- TUNABLE KALMAN PARAMETERS ---
        # processNoiseCov (Q): How much uncertainty in the motion model.
        # Higher values make the filter adapt faster to unmodeled changes (e.g., sudden acceleration)
        # but can make the prediction jittery if too high.
        # Order of states: x, y, vx, vy, ax, ay
        proc_noise_pos = 0.01  # For position (usually kept relatively small)
        proc_noise_vel = 0.05  # For velocity
        proc_noise_acc = (
            0.1  # For acceleration (increase if lagging during accelerations)
        )

        self.kf.processNoiseCov = (
            np.diag(
                [
                    proc_noise_pos,
                    proc_noise_pos,
                    proc_noise_vel,
                    proc_noise_vel,
                    proc_noise_acc,
                    proc_noise_acc,
                ]
            ).astype(np.float32)
            * 1.0
        )  # Overall multiplier for process noise

        # measurementNoiseCov (R): How much uncertainty in the YOLO detections.
        # Lower values mean more trust in the measurement, making the filter follow detections closely.
        # Higher values mean less trust, relying more on the model's prediction (smoother but can lag).
        meas_noise = 0.15  # Tunable: adjust based on YOLO detection stability
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        # --- END TUNABLE KALMAN PARAMETERS ---

        self.kf.errorCovPost = (
            np.eye(6, dtype=np.float32) * 0.1
        )  # Initial estimate error covariance

        # Initial state: [x, y, vx, vy, ax, ay]'
        self.kf.statePost = np.array(
            [np.float32(initial_x), np.float32(initial_y), 0, 0, 0, 0], np.float32
        ).reshape(6, 1)

        self.color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )  # For viz
        self.width = float(width)
        self.height = float(height)
        self._last_dt_used = 0.033  # For reference/debugging

    def _update_transition_matrix(self, dt):
        self._last_dt_used = dt
        # State transition matrix for constant acceleration model
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, dt, 0, 0.5 * dt * dt, 0],  # x = x_prev + vx*dt + 0.5*ax*dt^2
                [0, 1, 0, dt, 0, 0.5 * dt * dt],  # y = y_prev + vy*dt + 0.5*ay*dt^2
                [0, 0, 1, 0, dt, 0],  # vx = vx_prev + ax*dt
                [0, 0, 0, 1, 0, dt],  # vy = vy_prev + ay*dt
                [0, 0, 0, 0, 1, 0],  # ax = ax_prev
                [0, 0, 0, 0, 0, 1],  # ay = ay_prev
            ],
            np.float32,
        )

    def predict(
        self, dt
    ):  # dt is now an argument to update transition matrix dynamically
        self._update_transition_matrix(dt)
        prediction_state = (
            self.kf.predict()
        )  # This updates kf.statePre with the predicted state
        return int(prediction_state[0, 0]), int(prediction_state[1, 0])  # x, y

    def update(self, x, y, width, height):  # Called when a new measurement is available
        measurement = np.array([[np.float32(x)], [np.float32(y)]], np.float32)
        self.kf.correct(
            measurement
        )  # Corrects the state based on the new measurement, updates kf.statePost
        self.width = float(width)
        self.height = float(height)


class TrackerWrapper:
    def __init__(self, tracker: ArmorPlateTracker):
        self.tracker = tracker
        self.lost_frames = 0


# ──────────────────────────────────────────────────────────────


class YOLOv8Node(Node):
    def __init__(self):
        super().__init__("yolov8_node")
        self.get_logger().info(
            "Initializing YOLOv8Node with Kalman Filter & Lag Compensation (Basler Ace2)..."
        )
        fps = 60
        # --- ROS Parameters ---
        self.declare_parameter(
            "model_path",
            "/home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine",
        )
        self.declare_parameter("task", "detect")  # e.g., "detect", "obb"
        self.declare_parameter("type", "yolo")  # "yolo", "rtdetr", "nas"
        self.declare_parameter("half", True)  # Use FP16 precision
        # Set imgsz to match camera resolution
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter(
            "show", True
        )  # Enable/disable OpenCV visualization windows
        self.declare_parameter(
            "conf", 0.05
        )  # YOLO confidence threshold (LOWERED for better initial detection)
        self.declare_parameter(
            "tracker_timeout", 15
        )  # Frames to keep a lost track before deleting
        self.declare_parameter(
            "kalman_initial_dt", 1.0 / fps
        )  # Initial/fallback dt for Kalman filter (e.g., for 30fps)
        self.declare_parameter(
            "prediction_lookahead_steps", 0.5
        )  # How many extra 'dt' steps to look ahead (0=std KF, 0.5=half dt ahead, 1.0=full dt ahead)

        # Basler camera specific parameters
        self.declare_parameter("basler_width", 640)
        self.declare_parameter("basler_height", 480)
        self.declare_parameter(
            "basler_use_full_sensor", False
        )  # If True, use maximum sensor resolution instead of basler_width/height
        self.declare_parameter(
            "basler_fps", fps
        )  # Will attempt to get max FPS supported by the camera
        # Diagnostics/toggles for acquisition and inference
        self.declare_parameter("basler_timeout_ms", 1000)
        self.declare_parameter("enable_inference", True)
        self.declare_parameter("use_poll_for_frames", False)
        self.declare_parameter("enable_tracking", True)
        self.declare_parameter("persist_tracking", True)
        # Preallocate letterbox to minimize per-frame allocations
        self.declare_parameter("preallocate_letterbox", True)
        # Acquisition tuning
        self.declare_parameter("poll_idle_sleep_ms", 2)
        # Camera controls
        self.declare_parameter("camera_auto_exposure", True)
        self.declare_parameter("camera_exposure", 12000.0)  # in microseconds
        self.declare_parameter("camera_gain", 5.0)
        self.declare_parameter("camera_auto_gain", False)
        self.declare_parameter("use_kalman", False)  # Gate Kalman usage
        # Serial communication parameters
        self.declare_parameter(
            "serial_enable", True
        )  # Enable STM32 serial communication
        self.declare_parameter(
            "serial_port", "/dev/serial/by-path/platform-3610000.usb-usb-0:2.1:1.0"
        )  # Serial port path
        self.declare_parameter("serial_baudrate", 115200)  # Serial baud rate
        self.declare_parameter("serial_verbose", True)  # Verbose serial logging
        self.declare_parameter(
            "hfov_degrees", 60.0
        )  # Horizontal field of view in degrees
        self.declare_parameter(
            "vfov_degrees", 45.0
        )  # Vertical field of view in degrees

        # Get parameter values
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.task = self.get_parameter("task").get_parameter_value().string_value
        model_type = self.get_parameter("type").get_parameter_value().string_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.imgsz_width = (
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )
        self.imgsz_height = (
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )
        self.show = self.get_parameter("show").get_parameter_value().bool_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.tracker_timeout_val = (
            self.get_parameter("tracker_timeout").get_parameter_value().integer_value
        )
        self.kalman_initial_dt_val = (
            self.get_parameter("kalman_initial_dt").get_parameter_value().double_value
        )
        self.prediction_lookahead_steps_val = (
            self.get_parameter("prediction_lookahead_steps")
            .get_parameter_value()
            .double_value
        )

        self.basler_width = (
            self.get_parameter("basler_width").get_parameter_value().integer_value
        )
        self.basler_height = (
            self.get_parameter("basler_height").get_parameter_value().integer_value
        )
        self.basler_use_full_sensor = (
            self.get_parameter("basler_use_full_sensor").get_parameter_value().bool_value
        )
        self.basler_fps = (
            self.get_parameter("basler_fps").get_parameter_value().integer_value
        )
        self.basler_timeout_ms = (
            self.get_parameter("basler_timeout_ms").get_parameter_value().integer_value
        )
        self.enable_inference = (
            self.get_parameter("enable_inference").get_parameter_value().bool_value
        )
        self.use_poll_for_frames = (
            self.get_parameter("use_poll_for_frames").get_parameter_value().bool_value
        )
        self.enable_tracking = (
            self.get_parameter("enable_tracking").get_parameter_value().bool_value
        )
        self.persist_tracking = (
            self.get_parameter("persist_tracking").get_parameter_value().bool_value
        )
        self.preallocate_letterbox = (
            self.get_parameter("preallocate_letterbox").get_parameter_value().bool_value
        )
        self.poll_idle_sleep_ms = (
            self.get_parameter("poll_idle_sleep_ms").get_parameter_value().integer_value
        )
        # Camera control values
        self.camera_auto_exposure = (
            self.get_parameter("camera_auto_exposure").get_parameter_value().bool_value
        )
        self.camera_exposure = (
            self.get_parameter("camera_exposure").get_parameter_value().double_value
        )
        self.camera_gain = (
            self.get_parameter("camera_gain").get_parameter_value().double_value
        )
        self.camera_auto_gain = (
            self.get_parameter("camera_auto_gain").get_parameter_value().bool_value
        )
        self.use_kalman = (
            self.get_parameter("use_kalman").get_parameter_value().bool_value
        )
        # Serial communication parameters
        self.serial_enable = (
            self.get_parameter("serial_enable").get_parameter_value().bool_value
        )
        self.serial_port = (
            self.get_parameter("serial_port").get_parameter_value().string_value
        )
        self.serial_baudrate = (
            self.get_parameter("serial_baudrate").get_parameter_value().integer_value
        )
        self.serial_verbose = (
            self.get_parameter("serial_verbose").get_parameter_value().bool_value
        )
        self.hfov_degrees = (
            self.get_parameter("hfov_degrees").get_parameter_value().double_value
        )
        self.vfov_degrees = (
            self.get_parameter("vfov_degrees").get_parameter_value().double_value
        )

        # Convert FOV to radians for calculations
        self.hfov_rad = math.radians(self.hfov_degrees)
        self.vfov_rad = math.radians(self.vfov_degrees)

        # Initialize STM32 serial if enabled
        self.stm32_serial = None
        if self.serial_enable:
            try:
                self.stm32_serial = STM32Serial(
                    port=self.serial_port,
                    baudrate=self.serial_baudrate,
                    verbose=self.serial_verbose,
                )
                if self.stm32_serial.connected:
                    self.get_logger().info(
                        f"STM32 serial enabled: {self.serial_port} @ {self.serial_baudrate} baud"
                    )
                    if self.serial_verbose:
                        self.get_logger().info("Serial verbose logging enabled")
                        # Test connection
                        if self.stm32_serial.test_connection():
                            self.get_logger().info("✓ Serial connection test passed")
                        else:
                            self.get_logger().warn("⚠ Serial connection test failed")
                else:
                    self.get_logger().warn(
                        f"STM32 serial initialization failed, continuing without serial"
                    )
            except ImportError as e:
                self.get_logger().error(f"Failed to import serial module: {e}")
                self.get_logger().error(
                    "Please install pyserial: pip3 install pyserial"
                )
                self.stm32_serial = None
            except Exception as e:
                self.get_logger().error(
                    f"Failed to initialize STM32 serial: {type(e).__name__}: {e}"
                )
                self.get_logger().error(f"Full traceback: {traceback.format_exc()}")
                self.stm32_serial = None

        self.get_logger().info(f"Model Path: {model_path}")
        self.get_logger().info(
            f"Model Type: {model_type}, Task: {self.task}, ImgSz (W,H): ({self.imgsz_width},{self.imgsz_height})"
        )
        self.get_logger().info(
            f"Confidence: {self.conf}, Show Viz: {self.show}, Half Prec: {self.half}"
        )
        self.get_logger().info(f"Tracker Timeout: {self.tracker_timeout_val} frames")
        self.get_logger().info(f"Kalman Initial dt: {self.kalman_initial_dt_val:.4f}s")
        self.get_logger().info(
            f"Prediction Lookahead Steps: {self.prediction_lookahead_steps_val}"
        )
        self.get_logger().info(
            f"Basler Config - Width: {self.basler_width}, Height: {self.basler_height}, FPS: {self.basler_fps}"
        )
        self.get_logger().info(f"Kalman enabled: {self.use_kalman}")

        self.target_class = BLUE_CLASS  # Default target class

        # --- Model Initialization ---
        if model_type == "yolo":
            self.model = YOLO(model_path, task=self.task)
        elif model_type == "rtdetr":
            self.model = RTDETR(model_path)  # Task might be implicit
        elif model_type == "nas":
            self.model = NAS(model_path)  # Task might be implicit
        else:
            self.get_logger().error(f"Unsupported model type specified: {model_type}")
            raise NotImplementedError("Unsupported model type specified.")
        self.get_logger().info(f"YOLO Model '{model_type}' initialized successfully.")

        # --- Basler Camera Setup ---
        try:
            # Get the transport layer factory
            tl_factory = pylon.TlFactory.GetInstance()

            # Get all available devices
            devices = tl_factory.EnumerateDevices()
            if len(devices) == 0:
                self.get_logger().error("No Basler camera detected!")
                raise RuntimeError("No Basler camera found")

            self.get_logger().info(f"Found {len(devices)} Basler device(s)")

            # Create camera object (use first available camera)
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            self.camera.Open()

            # Get maximum sensor resolution
            try:
                max_width = self.camera.WidthMax.GetValue()
                max_height = self.camera.HeightMax.GetValue()
                self.get_logger().info(
                    f"Camera sensor maximum resolution: {max_width}x{max_height}"
                )
            except Exception as e:
                self.get_logger().warn(f"Could not get max sensor resolution: {e}")
                max_width = None
                max_height = None

            # Set camera parameters
            # Use full sensor if requested, otherwise use specified resolution
            if self.basler_use_full_sensor and max_width and max_height:
                actual_width = max_width
                actual_height = max_height
                self.get_logger().info(
                    f"Using full sensor resolution: {actual_width}x{actual_height}"
                )
            else:
                # Set resolution (if requested size is larger than max, use max)
                actual_width = min(self.basler_width, max_width) if max_width else self.basler_width
                actual_height = min(self.basler_height, max_height) if max_height else self.basler_height
            
            # Reset AOI to full sensor first (centers the AOI)
            try:
                self.camera.OffsetX.SetValue(0)
                self.camera.OffsetY.SetValue(0)
            except Exception as e:
                self.get_logger().warn(f"Could not reset AOI offset: {e}")
            
            # Set resolution
            self.camera.Width.SetValue(actual_width)
            self.camera.Height.SetValue(actual_height)
            
            if actual_width != self.basler_width or actual_height != self.basler_height:
                self.get_logger().warn(
                    f"Requested resolution {self.basler_width}x{self.basler_height} "
                    f"adjusted to {actual_width}x{actual_height} (sensor max)"
                )

            # Set frame rate (if supported)
            try:
                if self.camera.AcquisitionFrameRateEnable.GetValue():
                    self.camera.AcquisitionFrameRate.SetValue(float(self.basler_fps))
                else:
                    self.get_logger().warn(
                        "Frame rate control not available on this camera"
                    )
            except Exception as e:
                self.get_logger().warn(f"Could not set frame rate: {e}")

            # Set exposure
            if self.camera_auto_exposure:
                try:
                    self.camera.ExposureAuto.SetValue("Continuous")
                    self.get_logger().info("Auto exposure enabled")
                except Exception as e:
                    self.get_logger().warn(f"Could not enable auto exposure: {e}")
            else:
                try:
                    self.camera.ExposureAuto.SetValue("Off")
                    self.camera.ExposureTime.SetValue(float(self.camera_exposure))
                    self.get_logger().info(
                        f"Manual exposure set to: {self.camera_exposure} us"
                    )
                except Exception as e:
                    self.get_logger().warn(f"Could not set exposure: {e}")

            # Set gain
            if self.camera_auto_gain:
                try:
                    self.camera.GainAuto.SetValue("Continuous")
                    self.get_logger().info("Auto gain enabled")
                except Exception as e:
                    self.get_logger().warn(f"Could not enable auto gain: {e}")
            else:
                try:
                    self.camera.GainAuto.SetValue("Off")
                    self.camera.Gain.SetValue(float(self.camera_gain))
                    self.get_logger().info(f"Manual gain set to: {self.camera_gain}")
                except Exception as e:
                    self.get_logger().warn(f"Could not set gain: {e}")

            # Start grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            # Get actual resolution and frame rate
            actual_width = self.camera.Width.GetValue()
            actual_height = self.camera.Height.GetValue()
            try:
                actual_fps = self.camera.ResultingFrameRate.GetValue()
            except Exception:
                actual_fps = self.basler_fps

            self.get_logger().info(
                f"Basler camera started: {actual_width}x{actual_height} @ {actual_fps:.2f} FPS"
            )

            # Update kalman dt based on actual FPS
            self.kalman_initial_dt_val = (
                1.0 / actual_fps if actual_fps > 0 else self.kalman_initial_dt_val
            )
            self.get_logger().info(
                f"Updated Kalman initial dt based on actual FPS: {self.kalman_initial_dt_val:.4f}s"
            )

            # Precompute letterbox geometry and preallocate buffer if enabled
            self.lb_ready = False
            if self.preallocate_letterbox:
                try:
                    target_w = self.imgsz_width
                    target_h = self.imgsz_height
                    if (
                        actual_width > 0
                        and actual_height > 0
                        and target_w > 0
                        and target_h > 0
                    ):
                        scale = min(target_w / actual_width, target_h / actual_height)
                        new_w = int(actual_width * scale)
                        new_h = int(actual_height * scale)
                        dw = (target_w - new_w) // 2
                        dh = (target_h - new_h) // 2
                        self.prealloc_padded_image = np.full(
                            (target_h, target_w, 3), (114, 114, 114), dtype=np.uint8
                        )
                        self.lb_slice_rows = (dh, dh + new_h)
                        self.lb_slice_cols = (dw, dw + new_w)
                        self.lb_target_size = (new_w, new_h)
                        self.lb_ready = True
                        self.get_logger().info(
                            f"Preallocated letterbox: in {actual_width}x{actual_height} -> out {target_w}x{target_h} scale={scale:.4f}"
                        )
                except Exception as e:
                    self.get_logger().warn(
                        f"Preallocate letterbox failed; falling back to dynamic: {e}"
                    )

        except Exception as e:
            self.get_logger().error(f"Failed to initialize Basler camera: {e}")
            self.get_logger().error(
                "Try unplugging and replugging the camera, or check camera permissions"
            )
            raise RuntimeError(f"Camera initialization failed: {e}")

        # --- ROS Publishers ---
        self.int_subscription = self.create_subscription(
            Int32, "robot_id", self.id_listener_callback, 10
        )
        self.bbox_publisher = self.create_publisher(Int32MultiArray, "bounding_box", 10)
        self.override_publisher = self.create_publisher(Bool, "override", 10)
        self.get_logger().info("ROS Publishers/Subscribers initialized.")

        # --- Node State Variables ---
        self.override = Bool()
        self.bbox_msg = Int32MultiArray()
        default_center = self.imgsz_width // 2  # Use imgsz_width for default center
        self.bbox_msg.data = [
            default_center,
            default_center,
            default_center,
            default_center,
        ]  # Default bbox
        self.active_trackers = {}  # Stores {track_id: TrackerWrapper}
        self.last_frame_time = None  # For dynamic dt calculation
        self.fps_frame_count = 0  # For aggregated FPS logging
        self.fps_time_accum_s = 0.0  # Sum of dt across frames for FPS

        # Create a timer to capture frames and process them
        self.timer = self.create_timer(
            1.0 / self.basler_fps, self.timer_callback
        )  # Use configured FPS for timer

    # --- Helper for Angle Calculation ---
    def calculate_angles(self, x1: int, y1: int, x2: int, y2: int):
        """
        Calculate pitch and yaw angles from bounding box coordinates.

        Args:
            x1, y1: Top-left corner of bounding box
            x2, y2: Bottom-right corner of bounding box

        Returns:
            Tuple of (yaw_angle, pitch_angle) in radians
        """
        center_x = x1 + (x2 - x1) / 2.0
        center_y = y1 + (y2 - y1) / 2.0

        # Normalize pixel offset to [-1, 1] based on image dimensions
        pixel_offset_x = (center_x - self.imgsz_width / 2.0) / (self.imgsz_width / 2.0)
        pixel_offset_y = (center_y - self.imgsz_height / 2.0) / (
            self.imgsz_height / 2.0
        )

        # Convert pixel offset to angles using FOV
        yaw_angle = math.atan(pixel_offset_x * math.tan(self.hfov_rad / 2.0))
        pitch_angle = math.atan(pixel_offset_y * math.tan(self.vfov_rad / 2.0))

        return yaw_angle, pitch_angle

    # --- Timer Callback for Frame Acquisition and Processing ---
    def timer_callback(self):
        try:
            # Grab frame from Basler camera
            if self.use_poll_for_frames:
                grab_result = self.camera.RetrieveResult(
                    self.basler_timeout_ms, pylon.TimeoutHandling_Return
                )
                if not grab_result.GrabSucceeded():
                    # Avoid busy spin; sleep briefly if configured
                    if self.poll_idle_sleep_ms and self.poll_idle_sleep_ms > 0:
                        time.sleep(self.poll_idle_sleep_ms / 1000.0)
                    return
            else:
                grab_result = self.camera.RetrieveResult(
                    self.basler_timeout_ms, pylon.TimeoutHandling_ThrowException
                )

            if not grab_result.GrabSucceeded():
                self.get_logger().warn(
                    "Failed to grab frame from Basler camera. Skipping this frame."
                )
                grab_result.Release()
                return

            # Convert Basler frame to OpenCV image
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

            image = converter.Convert(grab_result)
            frame = image.GetArray()
            grab_result.Release()

            # frame is now a BGR8 OpenCV image

            if not self.enable_inference:
                return

            self.listener_callback(frame)  # Call the existing processing logic

        except pylon.TimeoutException as e:
            self.get_logger().warn(f"Frame timeout: {e}. Skipping frame.")
        except Exception as e:
            self.get_logger().error(
                f"Error in Basler frame acquisition or processing: {e}"
            )
            try:
                self.get_logger().error(
                    f"Exception type: {type(e).__name__} repr: {repr(e)}"
                )
                self.get_logger().error(traceback.format_exc())
            except Exception:
                pass

    # --- Main Image Processing Logic (now takes a cv2 image directly) ---
    def listener_callback(self, frame):
        # Calculate dynamic dt
        current_time = self.get_clock().now()
        dt = self.kalman_initial_dt_val
        if self.last_frame_time is not None:
            time_diff_s = (current_time - self.last_frame_time).nanoseconds / 1e9
            if 0.001 < time_diff_s < 1.0:  # Sanity check for dt
                dt = time_diff_s
                # Aggregate FPS over 10 frames
                self.fps_frame_count += 1
                self.fps_time_accum_s += time_diff_s
                if self.fps_frame_count >= 10:
                    avg_fps = (
                        self.fps_frame_count / self.fps_time_accum_s
                        if self.fps_time_accum_s > 0
                        else 0.0
                    )
                    self.get_logger().info(f"FPS (avg/10): {avg_fps:.2f}")
                    self.fps_frame_count = 0
                    self.fps_time_accum_s = 0.0
            else:
                pass
        if dt > 0:
            pass  # Avoid per-frame logging for FPS to reduce overhead
        self.last_frame_time = current_time

        original_height, original_width = frame.shape[:2]

        # Robust Padding/Resizing (Letterboxing)
        if self.lb_ready:
            try:
                resized_image = cv2.resize(
                    frame, self.lb_target_size, interpolation=cv2.INTER_LINEAR
                )
                rs0, rs1 = self.lb_slice_rows
                cs0, cs1 = self.lb_slice_cols
                self.prealloc_padded_image[rs0:rs1, cs0:cs1] = resized_image
                padded_image = self.prealloc_padded_image
            except Exception:
                # Fallback to dynamic path on failure
                target_h, target_w = self.imgsz_height, self.imgsz_width
                scale = 0.0
                if original_width > 0 and original_height > 0:
                    scale = min(target_w / original_width, target_h / original_height)
                new_w, new_h = int(original_width * scale), int(original_height * scale)
                padded_image = np.full(
                    (self.imgsz_height, self.imgsz_width, 3),
                    (114, 114, 114),
                    dtype=np.uint8,
                )
                if new_w > 0 and new_h > 0:
                    resized_image = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                    )
                    dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
                    padded_image[dh : dh + new_h, dw : dw + new_w] = resized_image
                else:
                    self.get_logger().warn(
                        f"Invalid resize dimensions ({new_w}x{new_h}). Resizing directly (may distort)."
                    )
                    padded_image = cv2.resize(
                        frame,
                        (self.imgsz_width, self.imgsz_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
        else:
            target_h, target_w = self.imgsz_height, self.imgsz_width
            scale = 0.0
            if original_width > 0 and original_height > 0:
                scale = min(target_w / original_width, target_h / original_height)
            new_w, new_h = int(original_width * scale), int(original_height * scale)
            padded_image = np.full(
                (self.imgsz_height, self.imgsz_width, 3),
                (114, 114, 114),
                dtype=np.uint8,
            )
            if new_w > 0 and new_h > 0:
                resized_image = cv2.resize(
                    frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )
                dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
                padded_image[dh : dh + new_h, dw : dw + new_w] = resized_image
            else:
                self.get_logger().warn(
                    f"Invalid resize dimensions ({new_w}x{new_h}). Resizing directly (may distort)."
                )
                padded_image = cv2.resize(
                    frame,
                    (self.imgsz_width, self.imgsz_height),
                    interpolation=cv2.INTER_LINEAR,
                )

        # YOLO Model Tracking
        # Use single integer for imgsz if square, otherwise use tuple (height, width) format
        if self.imgsz_width == self.imgsz_height:
            imgsz_value = self.imgsz_width
        else:
            imgsz_value = (self.imgsz_height, self.imgsz_width)  # YOLO expects (H, W)

        if self.enable_tracking:
            results = self.model.track(
                padded_image,
                half=self.half,
                imgsz=imgsz_value,
                show=False,  # Manual visualization
                conf=self.conf,
                persist=self.persist_tracking,  # Keep tracks between frames
                tracker="bytetrack.yaml",  # Specify a robust tracker
            )
        else:
            results = self.model(
                padded_image,
                half=self.half,
                imgsz=imgsz_value,
                show=False,
                conf=self.conf,
                verbose=False,
            )

        vis_frame = padded_image.copy() if self.show else None
        current_frame_processed_track_ids = set()
        published_bbox_this_frame = False
        best_enemy_bbox_xyxy = None  # For direct YOLO publish when Kalman is disabled

        if results and len(results) > 0:
            res_obj = results[0]  # Results for the first (and likely only) image

            boxes_data = None  # To hold r.boxes or r.obb
            if hasattr(res_obj, "boxes") and res_obj.boxes is not None:
                boxes_data = res_obj.boxes
            elif (
                hasattr(res_obj, "obb")
                and res_obj.obb is not None
                and self.task == "obb"
            ):
                boxes_data = res_obj.obb

            if (
                boxes_data is not None
                and hasattr(boxes_data, "xyxy")
                and boxes_data.xyxy is not None
            ):
                bbox_coords_list = boxes_data.xyxy.cpu().int().tolist()

                track_ids_list = []
                if hasattr(boxes_data, "id") and boxes_data.id is not None:
                    track_ids_list = boxes_data.id.cpu().int().tolist()
                else:  # Fallback if no track IDs
                    track_ids_list = [None] * len(bbox_coords_list)

                for i, bbox_xyxy in enumerate(bbox_coords_list):
                    track_id = track_ids_list[i]
                    x1, y1, x2, y2 = bbox_xyxy

                    if self.show and vis_frame is not None:  # Draw raw YOLO box
                        cv2.rectangle(
                            vis_frame,
                            (x1, y1),
                            (x2, y2),
                            BBOX_RAW_COLOR,
                            BBOX_LINE_SIZE,
                        )
                        if track_id is not None:
                            cv2.putText(
                                vis_frame,
                                f"ID:{track_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                BBOX_RAW_COLOR,
                                1,
                            )

                    roi_img = padded_image[
                        max(0, y1) : min(y2, self.imgsz_height),
                        max(0, x1) : min(x2, self.imgsz_width),
                    ]
                    if roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
                        continue  # Skip empty ROIs

                    is_enemy, _ = self.check_enemy(
                        roi_img
                    )  # check_enemy logs its own decision

                    if self.use_kalman:
                        if (
                            is_enemy and track_id is not None
                        ):  # Must be an enemy AND have a track ID for KF
                            current_frame_processed_track_ids.add(track_id)
                            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            width_det, height_det = float(abs(x2 - x1)), float(
                                abs(y2 - y1)
                            )

                            if track_id not in self.active_trackers:
                                self.get_logger().info(
                                    f"Creating new Kalman tracker for enemy ID: {track_id}"
                                )
                                self.active_trackers[track_id] = TrackerWrapper(
                                    ArmorPlateTracker(
                                        center_x, center_y, width_det, height_det
                                    )
                                )
                            else:
                                self.active_trackers[track_id].tracker.update(
                                    center_x, center_y, width_det, height_det
                                )
                            self.active_trackers[track_id].lost_frames = 0
                    else:
                        if is_enemy and best_enemy_bbox_xyxy is None:
                            best_enemy_bbox_xyxy = (x1, y1, x2, y2)

        if self.use_kalman:
            # Manage and predict from active Kalman trackers
            active_ids_to_remove = []
            for track_id, tw_wrapper in self.active_trackers.items():
                if track_id not in current_frame_processed_track_ids:
                    tw_wrapper.lost_frames += 1

                if tw_wrapper.lost_frames >= self.tracker_timeout_val:
                    active_ids_to_remove.append(track_id)
                    self.get_logger().info(
                        f"Removing tracker ID {track_id} (lost {tw_wrapper.lost_frames} frames)."
                    )
                    continue

                if (
                    not published_bbox_this_frame and tw_wrapper.lost_frames == 0
                ):  # Publish for first active, non-lost track
                    pred_x_curr, pred_y_curr = tw_wrapper.tracker.predict(
                        dt
                    )  # Standard KF prediction for t+dt

                    # Get latest corrected velocity and acceleration for lookahead
                    last_corrected_state = tw_wrapper.tracker.kf.statePost.flatten()
                    vx, vy = last_corrected_state[2], last_corrected_state[3]
                    ax, ay = last_corrected_state[4], last_corrected_state[5]

                    additional_lookahead_time = self.prediction_lookahead_steps_val * dt

                    pred_x_final = (
                        pred_x_curr
                        + vx * additional_lookahead_time
                        + 0.5 * ax * (additional_lookahead_time**2)
                    )
                    pred_y_final = (
                        pred_y_curr
                        + vy * additional_lookahead_time
                        + 0.5 * ay * (additional_lookahead_time**2)
                    )

                    kf_width, kf_height = (
                        tw_wrapper.tracker.width,
                        tw_wrapper.tracker.height,
                    )
                    pub_x1 = int(pred_x_final - kf_width / 2)
                    pub_y1 = int(pred_y_final - kf_height / 2)
                    pub_x2 = int(pred_x_final + kf_width / 2)
                    pub_y2 = int(pred_y_final + kf_height / 2)

                    # Clamp coordinates to image boundaries
                    pub_x1 = max(0, min(pub_x1, self.imgsz_width - 1))
                    pub_y1 = max(0, min(pub_y1, self.imgsz_height - 1))
                    pub_x2 = max(0, min(pub_x2, self.imgsz_width - 1))
                    pub_y2 = max(0, min(pub_y2, self.imgsz_height - 1))

                    if pub_x1 < pub_x2 and pub_y1 < pub_y2:  # Check for valid box
                        self.bbox_msg.data = [pub_x1, pub_y1, pub_x2, pub_y2]
                        self.override.data = True
                        self.bbox_publisher.publish(self.bbox_msg)
                        self.override_publisher.publish(self.override)
                        published_bbox_this_frame = True
                        self.get_logger().info(
                            f"Published KF BBox ID {track_id} (Lookahead): {[pub_x1, pub_y1, pub_x2, pub_y2]}"
                        )

                        # Send to STM32 via serial if enabled
                        if (
                            self.stm32_serial is not None
                            and self.stm32_serial.connected
                        ):
                            try:
                                yaw, pitch = self.calculate_angles(
                                    pub_x1, pub_y1, pub_x2, pub_y2
                                )
                                has_target = 1  # Valid target detected
                                success = self.stm32_serial.send_auto_aim(
                                    pitch, yaw, has_target
                                )
                                if not success and self.serial_verbose:
                                    self.get_logger().warn(
                                        f"Failed to send serial data (pitch={pitch:.4f}, yaw={yaw:.4f})"
                                    )
                            except Exception as e:
                                self.get_logger().warn(
                                    f"Exception sending serial data: {e}"
                                )

                        if self.show and vis_frame is not None:  # Draw KF box
                            cv2.rectangle(
                                vis_frame,
                                (pub_x1, pub_y1),
                                (pub_x2, pub_y2),
                                BBOX_KF_COLOR,
                                BBOX_LINE_SIZE,
                            )
                            text_y_pos = (
                                pub_y1 - 25
                                if pub_y1 > 25
                                else pub_y1 + int(kf_height) + 15
                            )  # Adjust text position
                            cv2.putText(
                                vis_frame,
                                f"KF_ID:{track_id}",
                                (pub_x1, text_y_pos),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                BBOX_KF_COLOR,
                                1,
                            )
                    else:
                        self.get_logger().warn(
                            f"KF ID {track_id} produced invalid lookahead box ({[pub_x1, pub_y1, pub_x2, pub_y2]}). Not publishing."
                        )

            for track_id_to_remove in active_ids_to_remove:
                if track_id_to_remove in self.active_trackers:
                    del self.active_trackers[track_id_to_remove]

            if (
                not published_bbox_this_frame
            ):  # Publish default if no valid KF track was published
                default_center_x = self.imgsz_width // 2
                default_center_y = self.imgsz_height // 2
                self.bbox_msg.data = [
                    default_center_x,
                    default_center_y,
                    default_center_x,
                    default_center_y,
                ]
                self.override.data = False
                self.bbox_publisher.publish(self.bbox_msg)
                self.override_publisher.publish(self.override)
        else:
            # Direct YOLO-based publish when Kalman is disabled
            if best_enemy_bbox_xyxy is not None:
                x1, y1, x2, y2 = best_enemy_bbox_xyxy
                pub_x1 = max(0, min(int(x1), self.imgsz_width - 1))
                pub_y1 = max(0, min(int(y1), self.imgsz_height - 1))
                pub_x2 = max(0, min(int(x2), self.imgsz_width - 1))
                pub_y2 = max(0, min(int(y2), self.imgsz_height - 1))
                if pub_x1 < pub_x2 and pub_y1 < pub_y2:
                    self.bbox_msg.data = [pub_x1, pub_y1, pub_x2, pub_y2]
                    self.override.data = True
                    self.bbox_publisher.publish(self.bbox_msg)
                    self.override_publisher.publish(self.override)
                    published_bbox_this_frame = True

                    # Send to STM32 via serial if enabled
                    if self.stm32_serial is not None and self.stm32_serial.connected:
                        try:
                            yaw, pitch = self.calculate_angles(
                                pub_x1, pub_y1, pub_x2, pub_y2
                            )
                            has_target = 1  # Valid target detected
                            success = self.stm32_serial.send_auto_aim(
                                pitch, yaw, has_target
                            )
                            if not success and self.serial_verbose:
                                self.get_logger().warn(
                                    f"Failed to send serial data (pitch={pitch:.4f}, yaw={yaw:.4f})"
                                )
                        except Exception as e:
                            self.get_logger().warn(
                                f"Exception sending serial data: {e}"
                            )

                    if self.show and vis_frame is not None:
                        cv2.rectangle(
                            vis_frame,
                            (pub_x1, pub_y1),
                            (pub_x2, pub_y2),
                            BBOX_RAW_COLOR,
                            BBOX_LINE_SIZE,
                        )
            if not published_bbox_this_frame:
                default_center_x = self.imgsz_width // 2
                default_center_y = self.imgsz_height // 2
                self.bbox_msg.data = [
                    default_center_x,
                    default_center_y,
                    default_center_x,
                    default_center_y,
                ]
                self.override.data = False
                self.bbox_publisher.publish(self.bbox_msg)
                self.override_publisher.publish(self.override)

        if self.show and vis_frame is not None:
            cv2.imshow("3. YOLOv8 Kalman Tracking", vis_frame)
            cv2.waitKey(1)  # Essential for imshow to refresh

    # --- ROS Callback for Robot ID ---
    def id_listener_callback(self, msg: Int32):
        robot_id = msg.data
        self.get_logger().info(f"Received robot_id: {robot_id}")
        if robot_id in RED_TEAM:
            self.target_class = BLUE_CLASS
            self.get_logger().info(
                f"Robot ID {robot_id} is RED_TEAM. Targeting BLUE_CLASS."
            )
        elif robot_id in BLUE_TEAM:
            self.target_class = RED_CLASS
            self.get_logger().info(
                f"Robot ID {robot_id} is BLUE_TEAM. Targeting RED_CLASS."
            )
        else:
            current_target_str = "BLUE" if self.target_class == BLUE_CLASS else "RED"
            self.get_logger().warn(
                f"Robot ID {robot_id} not in known teams. Maintaining current target_class: {current_target_str}"
            )

    # --- Enemy Validation Method (from user's original code) ---
    def check_enemy(self, roi_image):  # roi_image is a crop from padded_image
        if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
            self.get_logger().warn(
                "check_enemy: Invalid ROI shape (0 dimension); returning False."
            )
            return False, None  # Return a tuple

        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        target_ranges = BLUE_RANGES if self.target_class == BLUE_CLASS else RED_RANGES

        final_mask = np.zeros_like(hsv_roi[:, :, 0])
        for hsv_range in target_ranges:
            current_mask = cv2.inRange(hsv_roi, hsv_range[0], hsv_range[1])
            final_mask = cv2.bitwise_or(final_mask, current_mask)

        color_area = np.count_nonzero(final_mask)

        # Using the 0.01 (1%) area threshold from your original code. This is tunable.
        min_color_area_threshold = roi_image.shape[0] * roi_image.shape[1] * 0.01

        is_it_enemy = color_area > min_color_area_threshold

        return is_it_enemy, color_area  # Return tuple

    def destroy_node(self):
        self.get_logger().info("Stopping Basler camera...")
        try:
            if hasattr(self, "camera") and self.camera:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.Close()
                self.camera = None
        except Exception as e:
            self.get_logger().warn(f"Error stopping camera: {e}")

        # Close serial connection if enabled
        if hasattr(self, "stm32_serial") and self.stm32_serial is not None:
            try:
                self.stm32_serial.close()
            except Exception as e:
                self.get_logger().warn(f"Error closing serial connection: {e}")

        # Close CV windows before calling parent
        cv2.destroyAllWindows()

        super().destroy_node()


# --- Main Execution ---
def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received, shutting down.")
    finally:
        try:
            node.destroy_node()
        except Exception as e:
            print(f"Error in cleanup: {e}")
        finally:
            if rclpy.ok():  # Check if rclpy is still initialized
                rclpy.shutdown()


if __name__ == "__main__":
    main()
