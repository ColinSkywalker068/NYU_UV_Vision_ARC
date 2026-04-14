import atexit
import signal
import traceback

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# Import camera components from the same directory (both backends in camera.py)
try:
    from .camera import BaslerCamera, RealSenseCamera
except (ImportError, ValueError) as package_import_error:
    try:
        from camera import BaslerCamera, RealSenseCamera
    except Exception as fallback_error:
        raise ImportError(
            "Failed to import camera backends. "
            f"Package import error: {package_import_error}; "
            f"fallback import error: {fallback_error}"
        ) from package_import_error


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.get_logger().info("Initializing Camera Node...")
        self._shutting_down = False

        # --- Parameters ---
        self.declare_parameter("camera_type", "basler")
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("imgsz_height", 640)

        # Basler specific
        self.declare_parameter("basler_exposure_us", 5000.0)
        self.declare_parameter("basler_gain_db", 5.0)
        self.declare_parameter("basler_fps", 60)
        self.declare_parameter("basler_use_full_sensor", True)
        self.declare_parameter("basler_centered_roi", True)
        self.declare_parameter("basler_binning", 1)
        self.declare_parameter("basler_decimation", 1)
        self.declare_parameter("basler_auto_exposure", False)
        self.declare_parameter("basler_auto_gain", False)
        self.declare_parameter("basler_auto_white_balance", False)
        self.declare_parameter("basler_balance_red", 1.6)  # R/G/B channel gain
        self.declare_parameter(
            "basler_balance_green", 1.0
        )  # (tune to remove yellow tint)
        self.declare_parameter("basler_balance_blue", 2.2)
        self.declare_parameter("basler_enable_grab_thread", True)
        self.declare_parameter("basler_queue_timeout_ms", 50.0)

        # Realsense specific
        self.declare_parameter("realsense_width", 640)
        self.declare_parameter("realsense_height", 480)
        self.declare_parameter("realsense_fps", 60)
        self.declare_parameter(
            "realsense_exposure", 80.0
        )  # 80 = 8ms, good for fast motion
        self.declare_parameter("realsense_gain", 64.0)
        self.declare_parameter("realsense_auto_exposure", False)
        self.declare_parameter("realsense_auto_white_balance", True)
        self.declare_parameter("realsense_white_balance", 6500.0)
        self.declare_parameter("realsense_use_poll", False)
        self.declare_parameter("realsense_use_max_native_fps", False)

        # Retrieve common params
        self.camera_type = self.get_parameter("camera_type").value
        imgsz_width = self.get_parameter("imgsz_width").value
        imgsz_height = self.get_parameter("imgsz_height").value

        # --- Camera Init ---
        try:
            if self.camera_type == "basler":
                fps = self.get_parameter("basler_fps").value
                queue_timeout_ms = self.get_parameter("basler_queue_timeout_ms").value
                self.camera = BaslerCamera(
                    width=imgsz_width,
                    height=imgsz_height,
                    fps=fps,
                    exposure_us=self.get_parameter("basler_exposure_us").value,
                    gain_db=self.get_parameter("basler_gain_db").value,
                    use_full_sensor=self.get_parameter("basler_use_full_sensor").value,
                    centered_roi=self.get_parameter("basler_centered_roi").value,
                    binning=max(1, int(self.get_parameter("basler_binning").value)),
                    decimation=max(
                        1, int(self.get_parameter("basler_decimation").value)
                    ),
                    auto_exposure=self.get_parameter("basler_auto_exposure").value,
                    auto_gain=self.get_parameter("basler_auto_gain").value,
                    auto_white_balance=self.get_parameter(
                        "basler_auto_white_balance"
                    ).value,
                    balance_red=self.get_parameter("basler_balance_red").value,
                    balance_green=self.get_parameter("basler_balance_green").value,
                    balance_blue=self.get_parameter("basler_balance_blue").value,
                    enable_grab_thread=self.get_parameter(
                        "basler_enable_grab_thread"
                    ).value,
                    queue_timeout_sec=max(1.0, float(queue_timeout_ms)) / 1000.0,
                )
                self.fps = float(fps)

            elif self.camera_type == "realsense":
                rs_fps = self.get_parameter("realsense_fps").value
                self.camera = RealSenseCamera(
                    width=self.get_parameter("realsense_width").value,
                    height=self.get_parameter("realsense_height").value,
                    fps=rs_fps,
                    exposure=self.get_parameter("realsense_exposure").value,
                    gain=self.get_parameter("realsense_gain").value,
                    auto_exposure=self.get_parameter("realsense_auto_exposure").value,
                    auto_white_balance=self.get_parameter(
                        "realsense_auto_white_balance"
                    ).value,
                    white_balance=float(
                        self.get_parameter("realsense_white_balance").value
                    ),
                    use_poll=self.get_parameter("realsense_use_poll").value,
                    use_max_native_fps=self.get_parameter(
                        "realsense_use_max_native_fps"
                    ).value,
                )
                self.fps = float(rs_fps)
            else:
                raise ValueError(f"Unknown camera type: {self.camera_type}")

            self.camera.initialize()

            # Negotiate actual FPS after initialization
            if self.camera_type == "realsense":
                self._log_realsense_streams()
                # Log the pixel format prominently via ROS logger
                fmt = getattr(self.camera, "_stream_format", None)
                if fmt is not None:
                    try:
                        import pyrealsense2 as rs

                        if fmt == rs.format.bgr8:
                            fmt_name = "BGR8 (native, no conversion)"
                        elif fmt == rs.format.rgb8:
                            fmt_name = "RGB8 (channel swap only)"
                        elif fmt == rs.format.yuyv:
                            fmt_name = "YUYV (OpenCV YUV->BGR conversion)"
                        else:
                            fmt_name = str(fmt)
                    except Exception:
                        fmt_name = str(fmt)
                    self.get_logger().info(
                        f"========== PIXEL FORMAT: {fmt_name} =========="
                    )
                negotiated_fps = float(getattr(self.camera, "actual_fps", 0.0))
                if negotiated_fps > 0.0:
                    self.fps = negotiated_fps
                else:
                    self.fps = float(max(1, int(self.fps)))
            elif self.camera_type == "basler":
                if self.fps <= 0.0:
                    resulting_fps = float(getattr(self.camera, "actual_fps", 0.0))
                    self.fps = resulting_fps if resulting_fps > 0.0 else 60.0

            self.get_logger().info(
                f"{self.camera_type.capitalize()} Camera Initialized."
            )

        except Exception as e:
            self.get_logger().error(f"Failed to init camera: {e}")
            self.get_logger().error(traceback.format_exc())
            raise

        # --- Publisher ---
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_publisher = self.create_publisher(Image, "image_raw", image_qos)
        self.bridge = CvBridge()

        # --- Timer ---
        timer_period = 1.0 / self.fps if self.fps > 0 else 1.0 / 60.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Camera node publishing at {1 / timer_period:.2f} Hz.")

    def _log_realsense_streams(self):
        requested = getattr(self.camera, "requested_stream", None)
        negotiated = getattr(self.camera, "negotiated_stream", None)
        if requested is not None:
            self.get_logger().info(
                "RealSense requested color stream: "
                f"{requested[0]}x{requested[1]} @ {requested[2]} FPS"
            )
        if negotiated is not None:
            self.get_logger().info(
                "RealSense active color stream: "
                f"{negotiated[0]}x{negotiated[1]} @ {negotiated[2]} FPS"
            )
        else:
            fallback = float(getattr(self.camera, "actual_fps", 0.0))
            self.get_logger().info(
                f"RealSense active color FPS unavailable; using fallback {fallback:.2f} FPS"
            )

    def timer_callback(self):
        if self._shutting_down:
            return
        try:
            frame = self.camera.get_frame()
            if frame is not None:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = self.camera_type
                self.image_publisher.publish(msg)
            else:
                self.get_logger().warn("Failed to get frame from camera.")
        except Exception as e:
            self.get_logger().error(f"Error in camera timer callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def destroy_node(self):
        if self._shutting_down:
            return
        self._shutting_down = True
        self.get_logger().info("Shutting down camera node.")
        if hasattr(self, "timer") and self.timer is not None:
            self.timer.cancel()
        if hasattr(self, "camera") and self.camera:
            try:
                self.camera.release()
            except Exception as exc:
                self.get_logger().warn(f"Camera release error (non-fatal): {exc}")
        super().destroy_node()


def _shutdown(camera_node, executor=None):
    """Shared shutdown sequence — called from signals, atexit, and finally."""
    if camera_node is None:
        return
    try:
        camera_node.destroy_node()
    except Exception:
        pass
    try:
        if executor is not None:
            executor.shutdown()
    except Exception:
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()

    # Register atexit so the camera is released even if Python exits
    # without hitting our finally block (e.g. os._exit, fatal signal).
    atexit.register(lambda: _shutdown(camera_node))

    # Install signal handlers so Ctrl+C / kill always triggers a clean
    # shutdown instead of relying on rclpy's internal signal handling
    # (which can be unreliable inside containers or with busy executors).
    def _signal_handler(signum, _frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[camera_node] Received {sig_name}, shutting down cleanly...")
        _shutdown(camera_node)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        rclpy.spin(camera_node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as exc:
        camera_node.get_logger().error(f"Unexpected error: {exc}")
    finally:
        _shutdown(camera_node)


if __name__ == "__main__":
    main()
