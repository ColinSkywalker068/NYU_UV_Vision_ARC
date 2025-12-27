from pypylon import pylon
import cv2

class BaslerCamera:
    def __init__(self, width=640, height=480, fps=0, exposure_us=5000.0, gain_db=5.0):
        self.target_width = width
        self.target_height = height
        self.target_fps = fps
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.camera = None
        self.converter = None
        self.is_grabbing = False

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
        # Basic configuration logic from basler_test.py
        camera = self.camera
        
        max_width = camera.WidthMax.GetValue()
        max_height = camera.HeightMax.GetValue()
        
        # Use full sensor logic if needed, or target dimensions
        # Here we try to set target, capped by max
        w = min(self.target_width, max_width)
        h = min(self.target_height, max_height)
        
        try:
            camera.OffsetX.SetValue(0)
            camera.OffsetY.SetValue(0)
        except Exception:
            pass
            
        camera.Width.SetValue(w)
        camera.Height.SetValue(h)
        
        try:
            if camera.PixelFormat.IsWritable():
                camera.PixelFormat.SetValue("Mono8") # As per basler_test.py logic, though we convert to BGR later
        except Exception:
            pass

        if self.target_fps > 0:
            try:
                if camera.AcquisitionFrameRateEnable.GetValue():
                    camera.AcquisitionFrameRate.SetValue(float(self.target_fps))
            except Exception:
                pass
        
        # Manual Exposure/Gain as default (auto=False)
        try:
            camera.ExposureAuto.SetValue("Off")
            camera.ExposureTime.SetValue(float(self.exposure_us))
        except Exception:
            pass
            
        try:
            camera.GainAuto.SetValue("Off")
            camera.Gain.SetValue(float(self.gain_db))
        except Exception:
            pass

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

    def release(self):
        if self.camera:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            if self.camera.IsOpen():
                self.camera.Close()
            self.camera = None
            self.is_grabbing = False
