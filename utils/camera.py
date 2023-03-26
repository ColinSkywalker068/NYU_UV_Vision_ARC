import numpy as np
import cv2
import config as cfg
import video_record

if cfg.camera['DEPTH_CAPTURE']:
    try:
        import pyrealsense2.pyrealsense2 as rs
    except:
        import pyrealsense2 as rs


class imu_sensor:
    def __init__(self):

        # Make another pipeline for IMU
        self.imu_pipeline = rs.pipeline()
        imu_config = rs.config()
        imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
        imu_config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

        self.imu_pipeline.start(imu_config)

    def get_frame(self):
        try:
            imu_frames = self.imu_pipeline.wait_for_frames()
            acceleration = imu_frames[rs.stream.accel].as_motion_frame(
            ).get_motion_data()
            gyroscope = imu_frames[rs.stream.gyro].as_motion_frame(
            ).get_motion_data()
        except (RuntimeError, ValueError) as e:
            return False, None, None

        # TODO: potentially return time data along with accel/gyro value
        return True, acceleration, gyroscope


# camera class to read in input from webcam
class camera:
    def __init__(self, camera_index):
        """
        Constructor.
        Attributes:
            cap = A VideoCapture object of cv2 module
        """
        self.cap = cv2.VideoCapture(camera_index)
        # optimize camera capture for only compressed color frames
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    # Get Color frame
    def get_frame(self):
        """
        Obtains a frame from the camera and returns it.
        Returns:
        ret = True if frame was successfully obtained.
        depth_image = False always as this is class is for a normal camera.
        frame = frame ectracted from the camera feed.
        """
        if not self.cap.isOpened():
            return False, None, None
        ret, frame = self.cap.read()
        depth_image = None
        return ret, depth_image, frame

    def release(self):
        """
        Destructor for the VideoCapture
        """
        self.cap.release()


# camera class to read in input from realsense depth camera
class depth_camera:

    def __init__(self, frame_size=(640, 480), frame_rate=30):
        """
        Constructor. Intializes the depth camera object and prep it for capture the frames.
        Parameters: 
        frame_size: Frame size/ resoltuion specification for the the cam feed.
        frame_rate: rate at which the camera should capture frames.
        """

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)

        # Init streams
        config.enable_stream(
            rs.stream.depth, frame_size[0], frame_size[1], rs.format.z16, frame_rate)
        config.enable_stream(
            rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, frame_rate)
        # Disable laser
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.laser_power, 0)

        # color_sensor = next(sensor for sensor in pipeline_profile.get_device(
        # ).sensors if not sensor.is_depth_sensor())

        # color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        # color_sensor.set_option(rs.option.exposure, 60)
        # color_sensor.set_option(rs.option.brightness, 0)
        # color_sensor.set_option(rs.option.contrast, 50)
        # color_sensor.set_option(rs.option.gain, 128)
        # color_sensor.set_option(rs.option.gamma, 500)
        # color_sensor.set_option(rs.option.saturation, 64)

        # Start streaming
        self.pipeline.start(config)

        # algin depth frame to color frame
        self.align = rs.align(rs.stream.color)

        # video recorder
        if cfg.camera['RECORD']:
            self.video_recorder = video_record.VideoRecord()
        else:
            self.video_recorder = None

    # Get Depth and Color Frame

    def get_frame(self):
        """
        Extracts depth and color frame from camera feed.
        Returns:
        ret = True if success in obtaining the frames
        depth_image = Depth frame 
        color_image = RGB image frame
        """
        try:
            frames = self.pipeline.wait_for_frames()
        except:
            return False, None, None

        # Align the depth frame to color frame
        # See https://dev.intelrealsense.com/docs/rs-align
        frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Check if a valid frame is available to render
        if color_frame is None or depth_frame is None:
            return False, None, None

        depth_image = np.asarray(depth_frame.get_data())
        color_image = np.asarray(color_frame.get_data())

        # if self.video_recorder:
        # self.video_recorder.process_frame(color_image)

        return True, depth_image, color_image

    def release(self):
        """
        Stops the capture pipeline.
        """
        self.pipeline.stop()
