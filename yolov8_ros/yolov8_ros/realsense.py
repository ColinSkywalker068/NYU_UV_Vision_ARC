import pyrealsense2 as rs
import numpy as np
import time
import logging


class RealSenseCamera:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 60,
        exposure: float = 120.0,
        gain: float = 64.0,
        auto_exposure: bool = False,
        auto_white_balance: bool = False,
        white_balance: float = 4600.0,
        use_poll: bool = False,
        timeout_ms: int = 1000,
        poll_idle_sleep_ms: int = 2,
        power_line_frequency: int = 2,
        frames_queue_size: int = 1,
    ):
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.exposure = exposure
        self.gain = gain

        self.pipeline = None
        self.ctx = None
        self.config = None
        self.profile = None
        self.is_grabbing = False

        self.timeout_ms = timeout_ms
        self.use_poll = use_poll
        self.poll_idle_sleep_ms = poll_idle_sleep_ms

        self.auto_exposure = auto_exposure
        self.auto_white_balance = auto_white_balance
        self.white_balance = white_balance
        self.power_line_frequency = power_line_frequency
        self.frames_queue_size = frames_queue_size

        self.logger = logging.getLogger("RealSenseCamera")

    def initialize(self):
        self.ctx = rs.context()
        devices = self.ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense camera found")

        requested_profile = (
            int(self.target_width),
            int(self.target_height),
            int(self.target_fps),
        )
        fallback_profile = (640, 480, 30)
        profile_attempts = [requested_profile]
        if requested_profile != fallback_profile:
            profile_attempts.append(fallback_profile)

        self.profile = None
        last_error = None
        for width, height, fps in profile_attempts:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(
                rs.stream.color,
                width,
                height,
                rs.format.bgr8,
                fps,
            )
            try:
                self.profile = self.pipeline.start(self.config)
                self.target_width = width
                self.target_height = height
                self.target_fps = fps
                break
            except Exception as e:
                last_error = e

        if self.profile is None:
            raise RuntimeError(f"Failed to start RealSense camera: {last_error}")

        self.is_grabbing = True
        self._configure_sensor()
        self._verify_streams()

    def _configure_sensor(self):
        try:
            device = self.profile.get_device()
            color_sensor = None
            for sensor in device.sensors:
                try:
                    sname = sensor.get_info(rs.camera_info.name)
                except Exception:
                    sname = ""
                if "RGB" in sname or "Color" in sname:
                    color_sensor = sensor
                    break

            if color_sensor is not None:
                # Frames Queue Size
                if color_sensor.supports(rs.option.frames_queue_size):
                    color_sensor.set_option(
                        rs.option.frames_queue_size,
                        float(max(1, self.frames_queue_size)),
                    )

                # Auto Exposure / Manual Exposure
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    color_sensor.set_option(
                        rs.option.enable_auto_exposure,
                        1.0 if self.auto_exposure else 0.0,
                    )

                if not self.auto_exposure:
                    if color_sensor.supports(rs.option.exposure):
                        color_sensor.set_option(
                            rs.option.exposure, float(self.exposure)
                        )
                    if color_sensor.supports(rs.option.gain):
                        color_sensor.set_option(rs.option.gain, float(self.gain))

                # Auto White Balance / Manual White Balance
                if color_sensor.supports(rs.option.enable_auto_white_balance):
                    color_sensor.set_option(
                        rs.option.enable_auto_white_balance,
                        1.0 if self.auto_white_balance else 0.0,
                    )

                if not self.auto_white_balance and color_sensor.supports(
                    rs.option.white_balance
                ):
                    color_sensor.set_option(
                        rs.option.white_balance, float(self.white_balance)
                    )

                # Power Line Frequency
                if color_sensor.supports(rs.option.power_line_frequency):
                    color_sensor.set_option(
                        rs.option.power_line_frequency,
                        float(self.power_line_frequency),
                    )
            else:
                self.logger.warning(
                    "Color sensor not found; skipping option configuration."
                )

        except Exception as e:
            self.logger.warning(f"Failed applying color sensor options: {e}")

    def _verify_streams(self):
        # Check if unintended streams are active
        try:
            _ = self.profile.get_stream(rs.stream.depth)
            self.logger.warning("Depth stream is active (unexpected).")
        except Exception:
            pass

    def get_frame(self):
        if not self.pipeline or not self.is_grabbing:
            return None

        try:
            if self.use_poll:
                frames = self.pipeline.poll_for_frames()
                if not frames:
                    if self.poll_idle_sleep_ms > 0:
                        time.sleep(self.poll_idle_sleep_ms / 1000.0)
                    return None
            else:
                frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)

            color_frame = frames.get_color_frame()
            if not color_frame:
                return None

            # IMPORTANT: copy the data — np.asanyarray creates a zero-copy
            # view into the SDK's internal buffer. If the SDK recycles that
            # buffer (e.g. next frame arrives) while YOLO inference still
            # holds the array, the same memory is freed twice → "double free".
            frame = np.array(color_frame.get_data(), copy=True)
            return frame

        except RuntimeError as e:
            # Handle timeouts or frame drops gracefully
            if "timeout" in str(e).lower() or "Frame didn't arrive" in str(e):
                return None
            print(f"Error getting frame: {e}")
            return None
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None

    def release(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        self.is_grabbing = False
