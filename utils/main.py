import os
from pathlib import Path
import logging
import config
import camera
import plate_detector
import monitor
import cv2
import flagger
from video_record import VideoRecord
from uart_distance import UartDistance
if not os.path.exists("../logs/"):
    os.makedirs("../logs/")

# check if the file exists, create it if it doesn't
if not os.path.exists(config.general["LOG_PATH"]):
    open(config.general["LOG_PATH"], "a").close()
logging.basicConfig(
    filename=config.general["LOG_PATH"], filemode="a", level=logging.DEBUG
)
logger = logging.getLogger(Path(__file__).stem)


def main():
    # Create camera object
    if config.camera["DEPTH_CAPTURE"]:
        # creates depth camera object
        cam = camera.depth_camera(
            config.camera["FRAME_SIZE"], config.camera["FRAME_RATE"]
        )
    else:
        # creates vamera object (without depth capture)
        cam = camera.camera(config.camera["CAMERA_INDEX"])

    # create flagger object
    flagger_obj = flagger.Flagger(
        config.gameplay["OPPONENT_TEAM_COLOR"],
        config.gameplay["HSV_LOWER_RED"],
        config.gameplay["HSV_UPPER_RED"],
        config.gameplay["HSV_LOWER_RED_2"],
        config.gameplay["HSV_UPPER_RED_2"],
        config.gameplay["HSV_LOWER_BLUE"],
        config.gameplay["HSV_UPPER_BLUE"],
    )

    # create detector object
    # check if tensorrt is available
    if config.yolo_tensorrt["ENABLE"]:
        try:
            import tensorrt as trt
            model_weights_path = config.yolo_tensorrt["MODEL_WEIGHTS_PATH"]
        except ImportError:
            logger.debug("TensorRT not available, using YOLOv5")
            model_weights_path = config.yolo["MODEL_WEIGHTS_PATH"]
    else:
        model_weights_path = config.yolo["MODEL_WEIGHTS_PATH"]

    detector = plate_detector.plate_detector(
        config.yolo_tensorrt["REPO_PATH"],
        config.yolo_tensorrt["MODEL_SOURCE"],
        model_weights_path,
        tracking=config.detection["TRACKING"],
        flagger=flagger_obj,
        uart=config.detection["UART"],
    )
    if config.detection["UART"]:
       count = 0
       while detector.uart.Uart.get_referee_feedback()['robotID'] in [0, -1] and count <= 100:
           os.system('clear')   
           count += 1
           logger.debug("Robot ID not found")
           if count > 100:
               logger.debug("Robot ID not found")
               break
       if count <= 100:
           flagger_obj.opponent_color = (
            "RED" if detector.uart.Uart.get_referee_feedback()['robotID'] > 100 else "BLUE"
                #"RED" if detector.uart.Uart.get_robot_id() else "BLUE"
           )
           if config.general["DEBUG_LOGS"]:
               print("opponent color: ", flagger_obj.opponent_color)
       else:
            flagger_obj.opponent_color = config.gameplay["OPPONENT_TEAM_COLOR"]
            logger.debug("Robot ID not found, using default color")

    # create monitor object to display annotated images
    if config.detection["IMU"]:
        imu = camera.imu_sensor()
        # Might need another file for encoder help, which we also pass into Localization
        # localization = Localization(imu)
        # localizationThread = Thread(target=localization.main)
        # localizationThread.start()

    mon = monitor.monitor()
    if config.camera["RECORD"]:
        video_saver = VideoRecord()



    while True:
        # obtain frame
        ret, depth_image, color_image = cam.get_frame()
        if not ret:
            logger.debug("No frame received")
            continue
        # make deep copy of color image
        if config.camera["RECORD"]:
            video_saver.add_frame(color_image.copy())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # process frame to obtain object detection predicitons
        bboxes, messages = detector.process_frame(
            color_image,
            config.detection["DETECTION_CONFIDENCE_CUTOFF"],
            config.detection["NO_TO_DETECT"],
            config.general["DEBUG_LOGS"],
        )

        if detector.tracker is not None:
            bboxes = detector.update_tracker(bboxes)

        depth_values = None
        # @REFACTOR: why send messages to controls here???
        # sort bboxes by median distance
        # if depth_image is not None and len(bboxes) > 0:
        #     bboxes, depth_values = plate_detector.sort_bboxes_by_distance(bboxes, depth_image)

        # if len(bboxes) > 0:
        #     bboxes, messages = detector.send_msg(bboxes, messages, depth_values)

        # show detections on screen
        if config.detection["DISPLAY"]:
           
            # draw line of center
            center_x = int(config.camera_intrinsics[config.camera['CameraVersion']]['PPX'] )
            center_y = int(config.camera_intrinsics[config.camera['CameraVersion']]['PPY'])
            c_size=40
            cv2.line( color_image, (center_x-c_size, center_y), (center_x+c_size, center_y), (0,255,0), 2)
            cv2.line( color_image, (center_x, center_y-c_size), (center_x, center_y+c_size), (0,255,0), 2)
            mon.draw_on_image(color_image, bboxes, messages)

    # if config.detection["IMU"]:
    #     localizationThread.stop()



if __name__ == "__main__":
    main()
