from pypylon import pylon
import cv2


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
