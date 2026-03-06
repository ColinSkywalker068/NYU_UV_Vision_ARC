import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import traceback

# Import camera components from the same directory (both backends in camera.py)
try:
    from .camera import BaslerCamera, RealSenseCamera
except (ImportError, ValueError):
    from camera import BaslerCamera, RealSenseCamera


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.get_logger().info("Initializing Camera Node...")

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

        # Realsense specific
        self.declare_parameter("realsense_width", 640)
        self.declare_parameter("realsense_height", 480)
        self.declare_parameter("realsense_fps", 60)
        self.declare_parameter("realsense_exposure", 120.0)
        self.declare_parameter("realsense_gain", 64.0)
        self.declare_parameter("realsense_auto_exposure", False)
        self.declare_parameter("realsense_auto_white_balance", False)
        self.declare_parameter("realsense_white_balance", 4600.0)
        self.declare_parameter("realsense_use_poll", False)

        # Retrieve params
        self.camera_type = self.get_parameter("camera_type").value
        self.imgsz_width = self.get_parameter("imgsz_width").value
        self.imgsz_height = self.get_parameter("imgsz_height").value

        # --- Components Init ---
        try:
            if self.camera_type == "basler":
                basler_exposure = self.get_parameter("basler_exposure_us").value
                basler_gain = self.get_parameter("basler_gain_db").value
                basler_fps = self.get_parameter("basler_fps").value
                basler_use_full_sensor = self.get_parameter(
                    "basler_use_full_sensor"
                ).value
                basler_centered_roi = self.get_parameter("basler_centered_roi").value
                basler_binning = self.get_parameter("basler_binning").value
                basler_decimation = self.get_parameter("basler_decimation").value
                basler_auto_exposure = self.get_parameter("basler_auto_exposure").value
                basler_auto_gain = self.get_parameter("basler_auto_gain").value

                self.camera = BaslerCamera(
                    width=self.imgsz_width,
                    height=self.imgsz_height,
                    fps=basler_fps,
                    exposure_us=basler_exposure,
                    gain_db=basler_gain,
                    use_full_sensor=bool(basler_use_full_sensor),
                    centered_roi=bool(basler_centered_roi),
                    binning=max(1, int(basler_binning)),
                    decimation=max(1, int(basler_decimation)),
                    auto_exposure=bool(basler_auto_exposure),
                    auto_gain=bool(basler_auto_gain),
                )
                self.fps = basler_fps
            elif self.camera_type == "realsense":
                realsense_width = self.get_parameter("realsense_width").value
                realsense_height = self.get_parameter("realsense_height").value
                realsense_fps = self.get_parameter("realsense_fps").value
                realsense_exposure = self.get_parameter("realsense_exposure").value
                realsense_gain = self.get_parameter("realsense_gain").value
                realsense_auto_exposure = self.get_parameter(
                    "realsense_auto_exposure"
                ).value
                realsense_auto_white_balance = self.get_parameter(
                    "realsense_auto_white_balance"
                ).value
                realsense_white_balance = self.get_parameter(
                    "realsense_white_balance"
                ).value
                realsense_use_poll = self.get_parameter("realsense_use_poll").value

                self.camera = RealSenseCamera(
                    width=realsense_width,
                    height=realsense_height,
                    fps=realsense_fps,
                    exposure=realsense_exposure,
                    gain=realsense_gain,
                    auto_exposure=bool(realsense_auto_exposure),
                    auto_white_balance=bool(realsense_auto_white_balance),
                    white_balance=float(realsense_white_balance),
                    use_poll=bool(realsense_use_poll),
                )
                self.fps = realsense_fps
            else:
                raise ValueError(f"Unknown camera type: {self.camera_type}")

            self.camera.initialize()
            self.get_logger().info(
                f"{self.camera_type.capitalize()} Camera Initialized."
            )

        except Exception as e:
            self.get_logger().error(f"Failed to init camera: {e}")
            self.get_logger().error(traceback.format_exc())
            raise e

        # --- Publisher ---
        self.image_publisher = self.create_publisher(Image, "image_raw", 10)
        self.bridge = CvBridge()

        # --- Timer ---
        timer_period = 1.0 / self.fps if self.fps > 0 else 1.0 / 60.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"Camera node publishing at {1/timer_period:.2f} Hz.")

    def timer_callback(self):
        try:
            frame = self.camera.get_frame()
            if frame is not None:
                # The 'passthrough' encoding just copies the data. If the camera output is BGR8, this is efficient.
                image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                image_message.header.stamp = self.get_clock().now().to_msg()
                image_message.header.frame_id = self.camera_type
                self.image_publisher.publish(image_message)
            else:
                self.get_logger().warn("Failed to get frame from camera.")

        except Exception as e:
            self.get_logger().error(f"Error in camera timer callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def destroy_node(self):
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        self.get_logger().info("Shutting down camera node.")
        # Cancel the timer first to prevent callbacks during teardown
        if hasattr(self, "timer") and self.timer is not None:
            self.timer.cancel()
        if hasattr(self, "camera") and self.camera:
            try:
                self.camera.release()
            except Exception as exc:
                self.get_logger().warn(f"Camera release error (non-fatal): {exc}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        camera_node.get_logger().error(f"Unexpected error: {exc}")
    finally:
        camera_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
