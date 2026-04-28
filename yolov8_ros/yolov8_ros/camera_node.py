import faulthandler
import sys
import traceback
from dataclasses import dataclass

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

# Import camera components from the same directory (both backends in camera.py)
try:
    from .camera import BaslerCamera, RealSenseCamera
    from .shutdown_utils import installShutdownHandlers, shutdownRcl
except (ImportError, ValueError) as package_import_error:
    try:
        from camera import BaslerCamera, RealSenseCamera
        from shutdown_utils import installShutdownHandlers, shutdownRcl
    except Exception as fallback_error:
        raise ImportError(
            "Failed to import camera backends. "
            f"Package import error: {package_import_error}; "
            f"fallback import error: {fallback_error}"
        ) from package_import_error


@dataclass(frozen=True)
class BaslerInitOptions:
    width: int
    height: int
    fps: int
    exposure_us: float
    gain_db: float
    use_full_sensor: bool
    centered_roi: bool
    binning: int
    decimation: int
    auto_exposure: bool
    auto_gain: bool
    auto_white_balance: bool
    balance_red: float
    balance_green: float
    balance_blue: float
    enable_grab_thread: bool
    enable_transport_tuning: bool
    queue_timeout_sec: float


@dataclass(frozen=True)
class RealSenseInitOptions:
    width: int
    height: int
    fps: int
    exposure: float
    gain: float
    auto_exposure: bool
    auto_white_balance: bool
    white_balance: float
    use_poll: bool
    use_max_native_fps: bool


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.get_logger().info("Initializing Camera Node...")
        self._shutting_down = False

        # ─── parameters ───
        self.declare_parameter("camera_type", "basler")
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("imgsz_height", 640)

        # Basler specific
        self.declare_parameter("basler_exposure_us", 8000.0)
        self.declare_parameter("basler_gain_db", 10.0)
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
        self.declare_parameter("basler_enable_transport_tuning", True)
        self.declare_parameter("basler_queue_timeout_ms", 50.0)

        # Realsense specific
        self.declare_parameter("realsense_width", 640)
        self.declare_parameter("realsense_height", 480)
        self.declare_parameter("realsense_fps", 60)
        self.declare_parameter(
            "realsense_exposure", 40.0
        )  # 40 = 4ms, shorter exposure for less motion blur
        self.declare_parameter("realsense_gain", 64.0)
        self.declare_parameter("realsense_auto_exposure", False)
        self.declare_parameter("realsense_auto_white_balance", True)
        self.declare_parameter("realsense_white_balance", 6500.0)
        self.declare_parameter("realsense_use_poll", False)
        self.declare_parameter("realsense_use_max_native_fps", False)

        # Retrieve common params
        self.camera_type = self._get_string_parameter("camera_type")
        imgsz_width = self._get_int_parameter("imgsz_width")
        imgsz_height = self._get_int_parameter("imgsz_height")

        # ─── camera init ───
        try:
            if self.camera_type == "basler":
                fps = self._get_int_parameter("basler_fps")
                queue_timeout_ms = self._get_float_parameter("basler_queue_timeout_ms")
                basler_options = BaslerInitOptions(
                    width=imgsz_width,
                    height=imgsz_height,
                    fps=fps,
                    exposure_us=self._get_float_parameter("basler_exposure_us"),
                    gain_db=self._get_float_parameter("basler_gain_db"),
                    use_full_sensor=self._get_bool_parameter("basler_use_full_sensor"),
                    centered_roi=self._get_bool_parameter("basler_centered_roi"),
                    binning=max(1, self._get_int_parameter("basler_binning")),
                    decimation=max(1, self._get_int_parameter("basler_decimation")),
                    auto_exposure=self._get_bool_parameter("basler_auto_exposure"),
                    auto_gain=self._get_bool_parameter("basler_auto_gain"),
                    auto_white_balance=self._get_bool_parameter(
                        "basler_auto_white_balance"
                    ),
                    balance_red=self._get_float_parameter("basler_balance_red"),
                    balance_green=self._get_float_parameter("basler_balance_green"),
                    balance_blue=self._get_float_parameter("basler_balance_blue"),
                    enable_grab_thread=self._get_bool_parameter(
                        "basler_enable_grab_thread"
                    ),
                    enable_transport_tuning=self._get_bool_parameter(
                        "basler_enable_transport_tuning"
                    ),
                    # Guard against very small timeouts that cause frame starvation under load.
                    queue_timeout_sec=max(1.0, float(queue_timeout_ms)) / 1000.0,
                )
                self.camera = BaslerCamera(
                    width=basler_options.width,
                    height=basler_options.height,
                    fps=basler_options.fps,
                    exposure_us=basler_options.exposure_us,
                    gain_db=basler_options.gain_db,
                    use_full_sensor=basler_options.use_full_sensor,
                    centered_roi=basler_options.centered_roi,
                    binning=basler_options.binning,
                    decimation=basler_options.decimation,
                    auto_exposure=basler_options.auto_exposure,
                    auto_gain=basler_options.auto_gain,
                    auto_white_balance=basler_options.auto_white_balance,
                    balance_red=basler_options.balance_red,
                    balance_green=basler_options.balance_green,
                    balance_blue=basler_options.balance_blue,
                    enable_grab_thread=basler_options.enable_grab_thread,
                    enable_transport_tuning=basler_options.enable_transport_tuning,
                    queue_timeout_sec=basler_options.queue_timeout_sec,
                )
                self.fps = float(fps)

            elif self.camera_type == "realsense":
                rs_fps = self.get_parameter("realsense_fps").value
                realsense_options = RealSenseInitOptions(
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
                self.camera = RealSenseCamera(
                    width=realsense_options.width,
                    height=realsense_options.height,
                    fps=realsense_options.fps,
                    exposure=realsense_options.exposure,
                    gain=realsense_options.gain,
                    auto_exposure=realsense_options.auto_exposure,
                    auto_white_balance=realsense_options.auto_white_balance,
                    white_balance=realsense_options.white_balance,
                    use_poll=realsense_options.use_poll,
                    use_max_native_fps=realsense_options.use_max_native_fps,
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
                    # Keep timer sane even when the backend cannot report negotiated FPS.
                    self.fps = float(max(1, int(self.fps)))
            elif self.camera_type == "basler":
                if self.fps <= 0.0:
                    resulting_fps = float(getattr(self.camera, "actual_fps", 0.0))
                    self.fps = resulting_fps if resulting_fps > 0.0 else 60.0
                self._log_basler_settings()

            self.get_logger().info(
                f"{self.camera_type.capitalize()} Camera Initialized."
            )

        except Exception as e:
            self.get_logger().error(f"Failed to init camera: {e}")
            self.get_logger().error(traceback.format_exc())
            raise

        # ─── publisher ───
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.image_publisher = self.create_publisher(Image, "image_raw", image_qos)
        self.bridge = CvBridge()

        # ─── timer ───
        timer_period = 1.0 / self.fps if self.fps > 0 else 1.0 / 60.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Camera node publishing at {1 / timer_period:.2f} Hz.")

    def _get_parameter_value(self, name):
        return self.get_parameter(name).value

    def _get_string_parameter(self, name):
        return str(self._get_parameter_value(name)).strip().lower()

    def _get_int_parameter(self, name):
        return int(self._get_parameter_value(name))

    def _get_float_parameter(self, name):
        return float(self._get_parameter_value(name))

    def _get_bool_parameter(self, name):
        value = self._get_parameter_value(name)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

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

    def _log_basler_settings(self):
        settings = getattr(self.camera, "get_applied_settings", lambda: {})()
        ordered_keys = [
            "Width",
            "Height",
            "OffsetX",
            "OffsetY",
            "PixelFormat",
            "ResultingFrameRate",
            "AcquisitionFrameRateEnable",
            "AcquisitionFrameRate",
            "ExposureAuto",
            "ExposureTime",
            "GainAuto",
            "Gain",
            "BalanceWhiteAuto",
            "DeviceLinkThroughputLimitMode",
            "enable_grab_thread",
            "enable_transport_tuning",
        ]
        for key in ordered_keys:
            if key not in settings and key == "ResultingFrameRate":
                settings[key] = float(getattr(self.camera, "actual_fps", 0.0))
            if key in settings:
                self.get_logger().info(f"Basler {key}: {settings[key]}")

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


def main(args=None):
    faulthandler.enable(file=sys.stderr, all_threads=True)
    rclpy.init(args=args)
    camera_node = CameraNode()
    installShutdownHandlers(
        on_shutdown=lambda: shutdownRcl(node=camera_node),
        logger_name="camera_node",
    )
    try:
        rclpy.spin(camera_node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as exc:
        camera_node.get_logger().error(f"Unexpected error: {exc}")
    finally:
        shutdownRcl(node=camera_node)


if __name__ == "__main__":
    main()
