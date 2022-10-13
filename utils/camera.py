#import pyrealsense2.pyrealsense2 as rs
#import numpy as np
import cv2


#camera class to read in input from webcam
class camera:
    def __init__(self, camera_index):
        self.cap = cv2.VideoCapture(camera_index)
        
    #Get Color frame
    def get_frame(self):
        ret, frame = self.cap.read()
        depth_image = None
        
        return ret, depth_image, frame
    
    def release(self):
        self.cap.release()


#camera class to read in input from realsense depth camera
class depth_camera:

    def __init__(self, frame_size=(640,480), frame_rate=30):
        
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        
        #Init streams
        config.enable_stream(rs.stream.depth, frame_size[0], frame_size[1], rs.format.z16, frame_rate)
        config.enable_stream(rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, frame_rate)

        #Disable laser
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
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