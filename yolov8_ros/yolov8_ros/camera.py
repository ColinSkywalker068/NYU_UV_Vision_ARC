import logging
import threading
import time

import cv2
import numpy as np
from pypylon import pylon

try:
    import pyrealsense2 as rs
except ImportError:
    pass


class LatestFrameQueue:
    """Single-slot queue where consumers always get the latest frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._event = threading.Event()

    def put(self, frame):
        with self._lock:
            self._frame = frame
        self._event.set()

    def get(self, timeout_sec):
        if not self._event.wait(timeout_sec):
            return None
        with self._lock:
            frame = self._frame
            self._frame = None
            self._event.clear()
        return frame


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
            # Some transport nodes exist but are firmware-locked at runtime.
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
    # Hardware nodes reject off-grid values; floor-to-grid avoids set-value failures.
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


# Format preference order: YUYV supports the highest FPS on most D4xx
# color sensors, BGR8/RGB8 avoid conversion but cap at ~30fps.
# NOTE: Y8 (infrared) is excluded — YOLO needs colour channels.
_RS_COLOR_FORMAT_PREFERENCE = [
    "bgr8",  # native BGR, no conversion needed (if USB 3.0 supports 60fps)
    "rgb8",  # trivial channel swap
    "yuyv",  # fallback for USB 2.0 or older firmware
]


def _rs_format_by_name(rs_module, name):
    """Look up an rs.format enum value by short name."""
    return getattr(rs_module.format, name, None)


def _realsense_best_color_stream(ctx, width, height, min_fps=0):
    """Find the fastest color stream at the given resolution.

    Searches colour sensors only, across multiple pixel formats.
    Returns (width, height, fps, rs_format) or None.
    """
    try:
        devs = list(ctx.query_devices())
    except Exception:
        return None
    if not devs:
        return None

    try:
        sensors = devs[0].query_sensors()
    except Exception:
        return None

    # Build a set of acceptable formats
    fmt_map = {}  # rs_format_enum -> preference_rank
    for rank, name in enumerate(_RS_COLOR_FORMAT_PREFERENCE):
        fmt_enum = _rs_format_by_name(rs, name)
        if fmt_enum is not None:
            fmt_map[fmt_enum] = rank

    best = None
    best_fps = -1
    best_rank = len(_RS_COLOR_FORMAT_PREFERENCE)  # worse than any real rank

    for sensor in sensors:
        # Only scan colour sensors — infrared is single-channel and
        # unusable for downstream colour-based detection (YOLO, etc.).
        try:
            if not sensor.is_color_sensor():
                continue
        except Exception:
            continue
        for profile in sensor.get_stream_profiles():
            try:
                vp = profile.as_video_stream_profile()
                fmt = vp.format()
                if fmt not in fmt_map:
                    continue
                if vp.width() != int(width) or vp.height() != int(height):
                    continue
                fps = int(vp.fps())
                if fps < min_fps:
                    continue
                rank = fmt_map[fmt]
                # Prefer higher FPS; at equal FPS prefer better format rank
                if fps > best_fps or (fps == best_fps and rank < best_rank):
                    best_fps = fps
                    best_rank = rank
                    best = (int(vp.width()), int(vp.height()), fps, fmt)
            except Exception:
                continue
    return best


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
        auto_white_balance=False,
        balance_red=1.6,
        balance_green=1.0,
        balance_blue=2.2,
        enable_grab_thread=False,
        queue_timeout_sec=0.05,
        retrieve_timeout_ms=1000,
        enable_transport_tuning=False,
    ):
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.use_full_sensor = bool(use_full_sensor)
        self.centered_roi = bool(centered_roi)
        self.binning = max(1, int(binning))
        self.decimation = max(1, int(decimation))
        self.auto_exposure = bool(auto_exposure)
        self.auto_gain = bool(auto_gain)
        self.auto_white_balance = bool(auto_white_balance)
        self.balance_red = float(balance_red)
        self.balance_green = float(balance_green)
        self.balance_blue = float(balance_blue)
        self.enable_grab_thread = bool(enable_grab_thread)
        self.enable_transport_tuning = bool(enable_transport_tuning)
        self.queue_timeout_sec = max(0.001, float(queue_timeout_sec))
        self.retrieve_timeout_ms = max(1, int(retrieve_timeout_ms))

        self.camera = None
        self.converter = None
        self.is_grabbing = False
        self.applied_settings = {}
        self._native_bgr8 = False
        self._initialized = False
        self._frame_queue = LatestFrameQueue()
        self._grab_stop = threading.Event()
        self._grab_thread = None
        self.actual_fps = 0.0
        self.logger = logging.getLogger("BaslerCamera")
        self._release_lock = threading.Lock()

    def initialize(self):
        # Defensive: if re-initializing, release previous resources first.
        if self.camera is not None:
            self.release()

        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No Basler camera found")

        self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
        self.camera.Open()

        self._configure_camera()
        if self.enable_transport_tuning:
            self._configure_transport()

        # Build converter BEFORE grabbing starts so it is ready for the
        # first RetrieveResult call — avoids racing the grab thread.
        if not self._native_bgr8:
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
        else:
            self.converter = None

        # Start acquisition.
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.is_grabbing = True
        self._grab_stop.clear()

        # Warmup: retrieve and discard one frame synchronously.  This
        # forces pylon to fully initialise its internal transport buffers
        # before the grab thread starts hammering RetrieveResult, which
        # eliminates the double-free / corruption on startup.
        self._warmup_grab()

        self._initialized = True

        if self.enable_grab_thread:
            self._grab_thread = threading.Thread(
                target=self._grab_thread_fn,
                name="BaslerGrabThread",
                daemon=True,
            )
            self._grab_thread.start()

    # ------------------------------------------------------------------
    # Camera configuration (unchanged logic, cleaned up)
    # ------------------------------------------------------------------

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
        self._native_bgr8 = False
        if hasattr(camera, "PixelFormat"):
            _set_node_value_if_writable(camera.PixelFormat, "BGR8")
            try:
                self._native_bgr8 = camera.PixelFormat.GetValue() == "BGR8"
            except Exception:
                self._native_bgr8 = False
            if not self._native_bgr8:
                symbolics = _get_pixel_format_symbolics(camera)
                self.applied_settings["PixelFormat_available"] = (
                    ", ".join(symbolics) if symbolics else "(none)"
                )
                chosen_format = _choose_preferred_pixel_format(symbolics)
                if chosen_format is not None:
                    _set_node_value_if_writable(camera.PixelFormat, chosen_format)
            else:
                self.applied_settings["PixelFormat_available"] = "BGR8"
        _record_setting(
            self.applied_settings, "PixelFormat", lambda: camera.PixelFormat.GetValue()
        )
        try:
            if self.target_fps > 0:
                if hasattr(camera, "AcquisitionFrameRateEnable"):
                    camera.AcquisitionFrameRateEnable.SetValue(True)
                if hasattr(camera, "AcquisitionFrameRate"):
                    camera.AcquisitionFrameRate.SetValue(float(self.target_fps))
            else:
                if hasattr(camera, "AcquisitionFrameRateEnable"):
                    camera.AcquisitionFrameRateEnable.SetValue(False)
        except Exception as exc:
            self.logger.warning(f"Failed to configure AcquisitionFrameRate: {exc}")
        if self.auto_exposure:
            _set_node_value_if_writable(camera.ExposureAuto, "Continuous")
        else:
            _set_node_value_if_writable(camera.ExposureAuto, "Off")
            try:
                exposure_node = camera.ExposureTime
                clamped_exposure = float(
                    max(
                        float(exposure_node.GetMin()),
                        min(float(exposure_node.GetMax()), float(self.exposure_us)),
                    )
                )
            except Exception:
                clamped_exposure = float(self.exposure_us)
            _set_node_value_if_writable(camera.ExposureTime, clamped_exposure)
        if self.auto_gain:
            _set_node_value_if_writable(camera.GainAuto, "Continuous")
        else:
            _set_node_value_if_writable(camera.GainAuto, "Off")
            _set_node_value_if_writable(camera.Gain, float(self.gain_db))
        if self.auto_white_balance:
            _set_node_value_if_writable(camera.BalanceWhiteAuto, "Continuous")
        else:
            _set_node_value_if_writable(camera.BalanceWhiteAuto, "Off")
            # Set explicit R/G/B channel gains so the image isn't yellow
            for channel, ratio in [
                ("Red", self.balance_red),
                ("Green", self.balance_green),
                ("Blue", self.balance_blue),
            ]:
                try:
                    _set_node_value_if_writable(camera.BalanceRatioSelector, channel)
                    _set_node_value_if_writable(camera.BalanceRatio, float(ratio))
                except Exception:
                    pass

        # Disable the internal Color Correction Matrix (LightSourcePreset).
        # When active, the camera applies scene-dependent colour shifts
        # (e.g. red objects cool the image, blue objects warm it).
        try:
            _set_node_value_if_writable(camera.LightSourcePreset, "Off")
        except Exception:
            pass

        # Ace 2 / USB3 Specific: Disable throughput limit so camera uses full USB bandwidth
        try:
            _set_node_value_if_writable(camera.DeviceLinkThroughputLimitMode, "Off")
        except Exception:
            pass
        try:
            self.actual_fps = float(camera.ResultingFrameRate.GetValue())
            self.logger.info(
                f"Basler ResultingFrameRate (camera ceiling): {self.actual_fps:.2f} fps"
            )
        except Exception:
            self.actual_fps = 0.0

    def _configure_transport(self):
        camera = self.camera
        try:
            sg = camera.GetStreamGrabberParams()
            try:
                sg.MaxTransferSize.SetValue(65536)
            except Exception as exc:
                self.logger.warning(f"MaxTransferSize unsupported: {exc}")
            try:
                sg.NumMaxQueuedUrbs.SetValue(32)
            except Exception as exc:
                self.logger.warning(f"NumMaxQueuedUrbs unsupported: {exc}")
        except Exception as exc:
            self.logger.warning(f"StreamGrabberParams unavailable: {exc}")

        try:
            camera.MaxNumBuffer.SetValue(32)
        except Exception as exc:
            self.logger.warning(f"MaxNumBuffer unsupported: {exc}")

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def _warmup_grab(self):
        """Retrieve and discard frames until one succeeds (max 10 attempts).

        This forces pylon's transport layer to fully initialise its
        internal buffer pool, which prevents the double-free / heap
        corruption that occurs when the grab thread retrieves before
        the buffers are ready.
        """
        for _ in range(10):
            try:
                grab = self.camera.RetrieveResult(
                    self.retrieve_timeout_ms, pylon.TimeoutHandling_Return
                )
            except Exception:
                continue
            if grab is None:
                continue
            succeeded = grab.GrabSucceeded()
            grab.Release()
            if succeeded:
                return
        self.logger.warning("Basler warmup: no successful frame in 10 attempts")

    def _convert_grab_to_array(self, grab):
        """Convert a successful grab result to a contiguous BGR8 numpy array.

        The array is fully copied into Python-owned memory before the
        grab result is released, so no dangling pointers remain.
        """
        if self._native_bgr8:
            frame = np.array(grab.GetArray(), copy=True)
            grab.Release()
            return frame

        if self.converter is None:
            grab.Release()
            return None

        converted = self.converter.Convert(grab)
        # Materialize into Python-owned memory immediately.
        frame = np.array(converted.GetArray(), copy=True)
        grab.Release()
        return frame

    def _retrieve_single_frame(self):
        camera = self.camera
        if camera is None or not self.is_grabbing or self._grab_stop.is_set():
            return None
        try:
            grab = camera.RetrieveResult(
                self.retrieve_timeout_ms, pylon.TimeoutHandling_Return
            )
        except Exception:
            return None
        if not grab:
            return None
        if not grab.GrabSucceeded():
            grab.Release()
            return None
        return self._convert_grab_to_array(grab)

    def _grab_thread_fn(self):
        while not self._grab_stop.is_set():
            frame = self._retrieve_single_frame()
            if frame is not None:
                self._frame_queue.put(frame)

    def get_frame(self):
        if not self._initialized or self._grab_stop.is_set():
            return None
        if self.enable_grab_thread:
            return self._frame_queue.get(self.queue_timeout_sec)
        return self._retrieve_single_frame()

    def get_applied_settings(self):
        return dict(self.applied_settings)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def release(self):
        with self._release_lock:
            camera = self.camera
            if camera is None:
                return

            # 1. Signal everything to stop.
            self._initialized = False
            self.is_grabbing = False
            self._grab_stop.set()

            # 2. Stop grabbing so RetrieveResult unblocks in the thread.
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
            except Exception as exc:
                self.logger.warning(f"Basler StopGrabbing during release failed: {exc}")

            # 3. Wait for the grab thread to exit BEFORE touching the
            #    camera or converter — the thread may still be mid-Convert.
            if self._grab_thread is not None:
                self._grab_thread.join(
                    timeout=max(2.0, self.retrieve_timeout_ms / 1000.0 + 1.0)
                )
                if self._grab_thread.is_alive():
                    self.logger.warning(
                        "Basler grab thread did not exit cleanly before release."
                    )
                self._grab_thread = None

            # 4. Close and destroy the device FIRST — while Python still
            #    holds live references to the camera and converter objects.
            #    Dropping references before Close() lets C++ destructors
            #    run on pylon internals in the wrong order, which corrupts
            #    the USB transport state and causes double-free on re-run.
            try:
                if camera.IsOpen():
                    camera.Close()
            except Exception as exc:
                self.logger.warning(f"Basler Close during release failed: {exc}")

            try:
                camera.DestroyDevice()
            except Exception:
                pass

            # 5. NOW safe to drop Python references — the C++ side is
            #    already fully torn down so destructors are no-ops.
            self.converter = None
            self.camera = None

    def __del__(self):
        # Safety net: if release() was never called (e.g. unhandled
        # exception during startup), try to clean up.
        try:
            if self.camera is not None:
                self.release()
        except Exception:
            pass


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
        use_max_native_fps=False,
        enable_grab_thread=False,
        queue_timeout_sec=0.05,
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
        self.use_max_native_fps = bool(use_max_native_fps)
        self.enable_grab_thread = bool(enable_grab_thread)
        self.queue_timeout_sec = max(0.001, float(queue_timeout_sec))

        self.pipeline = None
        self.config = None
        self.profile = None
        self.is_grabbing = False
        self._stream_format = None  # rs.format enum of the active stream

        self.timeout_ms = 1000
        self.poll_idle_sleep_ms = 2
        self.power_line_frequency = 2  # 1=50Hz, 2=60Hz
        self.frames_queue_size = 1

        self.logger = logging.getLogger("RealSenseCamera")
        self.actual_fps = 0.0
        self.requested_stream = None
        self.negotiated_stream = None
        self._frame_queue = LatestFrameQueue()
        self._grab_stop = threading.Event()
        self._grab_thread = None
        self._release_lock = threading.Lock()

    def initialize(self):
        ctx = rs.context()
        self.pipeline = rs.pipeline(ctx)
        self.config = rs.config()
        requested_stream = None

        # Check for devices
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense camera found")

        # Disable all streams first to ensure clean state
        try:
            self.config.disable_all_streams()
        except Exception:
            pass

        # Discover the best available stream across all formats.
        discovered = _realsense_best_color_stream(
            ctx,
            self.target_width,
            self.target_height,
            min_fps=self.target_fps if not self.use_max_native_fps else 0,
        )

        if discovered is not None:
            req_w, req_h, req_fps, req_fmt = discovered
            self.logger.info(
                "RealSense discovered best color stream: %dx%d @ %d FPS (format=%s)",
                req_w,
                req_h,
                req_fps,
                req_fmt,
            )
        else:
            # Fallback: request YUYV on color at target FPS.
            req_w = int(self.target_width)
            req_h = int(self.target_height)
            req_fps = max(1, int(round(self.target_fps)))
            req_fmt = rs.format.yuyv
            self.logger.warning(
                "No matching RealSense color stream found; falling back to %dx%d @ %d YUYV",
                req_w,
                req_h,
                req_fps,
            )

        requested_stream = (req_w, req_h, req_fps)
        self._stream_format = req_fmt

        self.config.enable_stream(
            rs.stream.color,
            req_w,
            req_h,
            req_fmt,
            req_fps,
        )

        try:
            self.profile = self.pipeline.start(self.config)
        except Exception as exc:
            raise RuntimeError(f"Failed to start RealSense camera: {exc}")

        self.is_grabbing = True
        self._grab_stop.clear()
        self._configure_sensor()
        try:
            stream_profile = self.profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
            self.actual_fps = float(stream_profile.fps())
            negotiated_stream = (
                int(stream_profile.width()),
                int(stream_profile.height()),
                int(stream_profile.fps()),
            )
        except Exception:
            self.actual_fps = float(max(1, int(round(self.target_fps))))
            negotiated_stream = None

        if requested_stream is not None:
            self.requested_stream = requested_stream
            self.logger.info(
                "RealSense requested color stream: %dx%d @ %d FPS",
                requested_stream[0],
                requested_stream[1],
                requested_stream[2],
            )
        if negotiated_stream is not None:
            self.negotiated_stream = negotiated_stream
            self.logger.info(
                "RealSense active color stream: %dx%d @ %d FPS",
                negotiated_stream[0],
                negotiated_stream[1],
                negotiated_stream[2],
            )
        else:
            self.negotiated_stream = None
            self.logger.info(
                "RealSense active color FPS unavailable; using fallback %.2f FPS",
                self.actual_fps,
            )

        # Log the pixel format prominently so we can debug colour issues
        fmt = self._stream_format
        fmt_name = "UNKNOWN"
        if fmt == rs.format.bgr8:
            fmt_name = "BGR8 (native, no conversion needed)"
        elif fmt == rs.format.rgb8:
            fmt_name = "RGB8 (channel swap only)"
        elif fmt == rs.format.yuyv:
            fmt_name = "YUYV (software YUV->BGR conversion via OpenCV)"
        else:
            fmt_name = str(fmt)
        self.logger.info("========== RealSense PIXEL FORMAT: %s ==========", fmt_name)

        # Verify streams
        self._verify_streams()

    def _configure_sensor(self):
        try:
            device = self.profile.get_device()
            color_sensor = None
            try:
                sensors = device.query_sensors()
            except Exception:
                sensors = list(getattr(device, "sensors", []))
            for sensor in sensors:
                try:
                    if sensor.is_color_sensor():
                        color_sensor = sensor
                        break
                except Exception:
                    continue

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

    def _convert_to_bgr(self, raw):
        """Convert a raw frame array to BGR8 based on the active stream format."""
        fmt = self._stream_format

        if fmt == rs.format.bgr8:
            return raw

        if fmt == rs.format.rgb8:
            return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

        if fmt == rs.format.yuyv:
            # librealsense buffer is 16-bit (2 bytes) per pixel macropixel.
            # We must view as uint8 and reshape to (H, W, 2) for OpenCV.
            # CRITICAL: call .copy() AFTER reshape so the final array owns its own
            # memory and does NOT share the buffer with `raw`. Without this, both
            # the view and the original array reference the same allocation, causing
            # a double-free / glibc heap corruption when both go out of scope.
            if self.negotiated_stream:
                w, h = self.negotiated_stream[0], self.negotiated_stream[1]
            else:
                w = self.target_width
                h = raw.view(np.uint8).size // (w * 2)
            yuyv = raw.view(np.uint8).reshape((h, w, 2)).copy()
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

        # Unknown format — try to use as-is.
        self.logger.warning(f"No conversion for format {fmt}, returning raw frame")
        return raw

    def _retrieve_single_frame(self):
        pipeline = self.pipeline
        if pipeline is None or not self.is_grabbing or self._grab_stop.is_set():
            return None

        try:
            if self.use_poll:
                frames = pipeline.poll_for_frames()
                if not frames:
                    if self.poll_idle_sleep_ms > 0:
                        time.sleep(self.poll_idle_sleep_ms / 1000.0)
                    return None
            else:
                frames = pipeline.wait_for_frames(timeout_ms=self.timeout_ms)

            color_frame = frames.get_color_frame()
            if not color_frame:
                return None

            # Materialize an independent contiguous array before any RealSense
            # frame objects go out of scope to avoid SDK buffer lifetime bugs.
            raw = np.asanyarray(color_frame.get_data()).copy()
            del color_frame
            del frames

            # Convert to BGR8 if the stream isn't already BGR.
            frame = self._convert_to_bgr(raw)
            return frame

        except RuntimeError as e:
            # Handle timeouts or frame drops gracefully
            if "timeout" in str(e).lower() or "Frame didn't arrive" in str(e):
                return None
            self.logger.warning(f"Error getting RealSense frame: {e}")
            return None
        except Exception as e:
            if self._grab_stop.is_set():
                return None
            self.logger.warning(f"Error getting RealSense frame: {e}")
            return None

    def _grab_thread_fn(self):
        while not self._grab_stop.is_set():
            frame = self._retrieve_single_frame()
            if frame is None:
                continue
            self._frame_queue.put(frame)

    def get_frame(self):
        if self._grab_stop.is_set():
            return None
        if self.enable_grab_thread:
            return self._frame_queue.get(self.queue_timeout_sec)
        return self._retrieve_single_frame()

    def release(self):
        with self._release_lock:
            pipeline = self.pipeline
            if pipeline is None:
                return

            self.is_grabbing = False
            self._grab_stop.set()

            if self._grab_thread is not None:
                self._grab_thread.join(timeout=max(2.0, self.timeout_ms / 1000.0 + 1.0))
                if self._grab_thread.is_alive():
                    self.logger.warning(
                        "RealSense grab thread did not exit cleanly before release."
                    )
                self._grab_thread = None

            self.profile = None
            self.config = None
            self.pipeline = None

            try:
                pipeline.stop()
            except Exception as exc:
                self.logger.warning(
                    f"RealSense pipeline stop during release failed: {exc}"
                )
