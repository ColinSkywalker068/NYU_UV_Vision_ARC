import pyrealsense2.pyrealsense2 as rs
import numpy as np

# Configure depth and color streams


class DepthCamera:

    # Constructor
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        depth_sensor=device.query_sensors()[0]
        depth_sensor.set_option(rs.option.laser_power, 0)
        
        # Start streaming
        self.pipeline.start(config)

    # Get Depth and Color Frame
    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
        except:
            return False, None, None
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if not depth_frame or not color_frame:
            return False, None, None

        return True, depth_image, color_image

    def release(self):
        self.pipeline.stop()
