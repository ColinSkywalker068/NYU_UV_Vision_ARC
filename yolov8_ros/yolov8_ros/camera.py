from pypylon import pylon
import cv2
import numpy as np
import time
import logging

try:
    import pyrealsense2 as rs
except ImportError:
    pass


def _record_setting(applied_settings, name, getter):
    try:
        applied_settings[name] = getter()
    except Exception:
        applied_settings[name] = "(error)"


def _set_node_value_if_writable(node, value):
    if node is None:
        return False
    try:
        if hasattr(node, "IsWritable") and not node.IsWritable():
            return False
        node.SetValue(value)
        return True
    except Exception:
        return False


def _get_int_node_value(node, attr_name, default_value):
    try:
        attribute = getattr(node, attr_name)
        return int(attribute() if callable(attribute) else attribute)
    except Exception:
        return int(default_value)


def _clamp_to_node_range(node, requested_value):
    minimum = _get_int_node_value(node, "GetMin", requested_value)
    maximum = _get_int_node_value(node, "GetMax", requested_value)
    return max(minimum, min(maximum, int(requested_value)))


def _align_to_node_increment(node, requested_value):
    clamped_value = _clamp_to_node_range(node, requested_value)
    increment = _get_int_node_value(node, "GetInc", 1)
    if increment <= 1:
        return clamped_value
    minimum = _get_int_node_value(node, "GetMin", 0)
    aligned_value = minimum + ((clamped_value - minimum) // increment) * increment
    return _clamp_to_node_range(node, aligned_value)


def _set_downsample_factor(camera, node_name, factor, applied_settings):
    if factor <= 1:
        return
    try:
        node = getattr(camera, node_name)
    except Exception:
        return
    aligned_factor = _align_to_node_increment(node, factor)
    if _set_node_value_if_writable(node, aligned_factor):
        _record_setting(applied_settings, node_name, lambda: node.GetValue())


def _compute_centered_offset(full_size, roi_size, node):
    centered_offset = max(0, (int(full_size) - int(roi_size)) // 2)
    return _align_to_node_increment(node, centered_offset)


def _get_pixel_format_symbolics(camera):
    try:
        if not hasattr(camera, "PixelFormat"):
            return []
        node = camera.PixelFormat
        if hasattr(node, "GetSymbolics"):
            return list(node.GetSymbolics())
        if hasattr(node, "Symbolics"):
            return list(node.Symbolics)
        return []
    except Exception:
        return []


def _choose_preferred_pixel_format(symbolics):
    preferred_order = [
        "BayerRG8",
        "BayerBG8",
        "BayerGB8",
        "BayerGR8",
        "RGB8",
        "BGR8",
        "Mono8",
    ]
    for pixel_format in preferred_order:
        if pixel_format in symbolics:
            return pixel_format
    return None


class BaslerCamera:
    def __init__(
        self,
        width=640,
        height=480,
        fps=0,
        exposure_us=5000.0,
        gain_db=5.0,
        use_full_sensor=True,
        centered_roi=True,
        binning=1,
        decimation=1,
        auto_exposure=False,
        auto_gain=False,
    ):
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.use_full_sensor = use_full_sensor
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.use_full_sensor = bool(use_full_sensor)
        self.centered_roi = bool(centered_roi)
        self.binning = max(1, int(binning))
        self.decimation = max(1, int(decimation))
        self.auto_exposure = bool(auto_exposure)
        self.auto_gain = bool(auto_gain)
        self.camera = None
        self.converter = None
        self.is_grabbing = False
        self.applied_settings = {}

    def initialize(self):
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera found")

        self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
        self.camera.Open()

        self._configure_camera()

        # Build converter
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.is_grabbing = True

    def _configure_camera(self):
        camera = self.camera
        self.applied_settings = {}
        _set_downsample_factor(
            camera, "BinningHorizontal", self.binning, self.applied_settings
        )
        _set_downsample_factor(
            camera, "BinningVertical", self.binning, self.applied_settings
        )
        _set_downsample_factor(
            camera, "DecimationHorizontal", self.decimation, self.applied_settings
        )
        _set_downsample_factor(
            camera, "DecimationVertical", self.decimation, self.applied_settings
        )
        max_width = camera.WidthMax.GetValue()
        max_height = camera.HeightMax.GetValue()
        target_width = (
            max_width if self.use_full_sensor else min(self.target_width, max_width)
        )
        target_height = (
            max_height if self.use_full_sensor else min(self.target_height, max_height)
        )
        target_width = _align_to_node_increment(camera.Width, int(target_width))
        target_height = _align_to_node_increment(camera.Height, int(target_height))
        offset_x = 0
        offset_y = 0
        if not self.use_full_sensor and self.centered_roi:
            offset_x = _compute_centered_offset(
                int(max_width), int(target_width), camera.OffsetX
            )
            offset_y = _compute_centered_offset(
                int(max_height), int(target_height), camera.OffsetY
            )
        _set_node_value_if_writable(camera.OffsetX, 0)
        _set_node_value_if_writable(camera.OffsetY, 0)
        _set_node_value_if_writable(camera.Width, target_width)
        _set_node_value_if_writable(camera.Height, target_height)
        _set_node_value_if_writable(camera.OffsetX, int(offset_x))
        _set_node_value_if_writable(camera.OffsetY, int(offset_y))
        _record_setting(self.applied_settings, "Width", lambda: camera.Width.GetValue())
        _record_setting(
            self.applied_settings, "Height", lambda: camera.Height.GetValue()
        )
        _record_setting(
            self.applied_settings, "OffsetX", lambda: camera.OffsetX.GetValue()
        )
        _record_setting(
            self.applied_settings, "OffsetY", lambda: camera.OffsetY.GetValue()
        )
        symbolics = _get_pixel_format_symbolics(camera)
        self.applied_settings["PixelFormat_available"] = (
            ", ".join(symbolics) if symbolics else "(none)"
        )
        chosen_format = _choose_preferred_pixel_format(symbolics)
        if chosen_format is not None:
            _set_node_value_if_writable(camera.PixelFormat, chosen_format)
        _record_setting(
            self.applied_settings, "PixelFormat", lambda: camera.PixelFormat.GetValue()
        )
        if self.target_fps > 0:
            try:
                if (
                    camera.AcquisitionFrameRateEnable.GetValue()
                    and camera.AcquisitionFrameRate.IsWritable()
                ):
                    camera.AcquisitionFrameRate.SetValue(float(self.target_fps))
            except Exception:
                pass
        if self.auto_exposure:
            _set_node_value_if_writable(camera.ExposureAuto, "Continuous")
        else:
            _set_node_value_if_writable(camera.ExposureAuto, "Off")
            _set_node_value_if_writable(camera.ExposureTime, float(self.exposure_us))
        if self.auto_gain:
            _set_node_value_if_writable(camera.GainAuto, "Continuous")
        else:
            _set_node_value_if_writable(camera.GainAuto, "Off")
            _set_node_value_if_writable(camera.Gain, float(self.gain_db))

    def get_frame(self):
        if not self.camera or not self.is_grabbing:
            return None

        grab = self.camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
        if grab and grab.GrabSucceeded():
            image = self.converter.Convert(grab)
            frame = image.GetArray()
            grab.Release()
            return frame

        if grab:
            grab.Release()
        return None

    def get_applied_settings(self):
        return dict(self.applied_settings)

    def release(self):
        if self.camera:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if self.camera.IsOpen():
                self.camera.Close()
            self.camera = None
            self.is_grabbing = False


class RealSenseCamera:
    def __init__(
        self,
        width=640,
        height=480,
        fps=60,
        exposure=120.0,
        gain=64.0,
        auto_exposure=False,
        auto_white_balance=False,
        white_balance=4600.0,
        use_poll=False,
    ):
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.exposure = exposure
        self.gain = gain
        self.auto_exposure = auto_exposure
        self.auto_white_balance = auto_white_balance
        self.white_balance = white_balance
        self.use_poll = use_poll

        self.pipeline = None
        self.config = None
        self.profile = None
        self.is_grabbing = False

        self.timeout_ms = 1000
        self.poll_idle_sleep_ms = 2
        self.power_line_frequency = 2  # 1=50Hz, 2=60Hz
        self.frames_queue_size = 1

        self.logger = logging.getLogger("RealSenseCamera")

    def initialize(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Check for devices
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense camera found")

        # Disable all streams first to ensure clean state
        try:
            self.config.disable_all_streams()
        except Exception:
            pass

        # Enable color stream
        self.config.enable_stream(
            rs.stream.color,
            self.target_width,
            self.target_height,
            rs.format.bgr8,
            self.target_fps,
        )

        # Start streaming
        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            raise RuntimeError(f"Failed to start RealSense camera: {e}")

        self.is_grabbing = True
        self._configure_sensor()

        # Verify streams
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
