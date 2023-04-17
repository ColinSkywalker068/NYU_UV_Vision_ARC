import json
from threading import Thread
import config
import camera
import plate_detector
import monitor
from localization import *
import cv2
import flagger
from video_record import VideoRecord

# from tracking import Tracker
# from tracking.sort import Sort


def main():

    # Create camera object
    if config.camera['DEPTH_CAPTURE']:
        # creates depth camera object
        cam = camera.depth_camera(
            config.camera['FRAME_SIZE'], config.camera['FRAME_RATE'])
    else:
        # creates vamera object (without depth capture)
        cam = camera.camera(config.camera['CAMERA_INDEX'])

    # create flagger object
    flagger_obj = flagger.Flagger(
        config.gameplay['OPPONENT_TEAM_COLOR'], config.gameplay['HSV_LOWER_RED'], config.gameplay['HSV_UPPER_RED'], config.gameplay['HSV_LOWER_RED_2'], config.gameplay['HSV_UPPER_RED_2'], config.gameplay['HSV_LOWER_BLUE'], config.gameplay['HSV_UPPER_BLUE'])

    # create detector object
    # check if tensorrt is available
    if config.yolo_tensorrt['ENABLE']:
        try:
            import tensorrt as trt
            model_weights_path = config.yolo_tensorrt['MODEL_WEIGHTS_PATH']
        except:
            print('TensorRT not available, using YOLOv5')
            model_weights_path = config.yolo['MODEL_WEIGHTS_PATH']
    else:
        model_weights_path = config.yolo['MODEL_WEIGHTS_PATH']

    detector = plate_detector.plate_detector(
        config.yolo_tensorrt['REPO_PATH'],
        config.yolo_tensorrt['MODEL_SOURCE'],
        model_weights_path,
        tracking=config.detection['TRACKING'],
        flagger=flagger_obj,
        uart=config.detection['UART'])

    # create monitor object to display annotated images
    if config.detection['IMU']:
        imu = camera.imu_sensor()
        # Might need another file for encoder help, which we also pass into Localization
        localization = Localization(imu)
        localizationThread = Thread(target=localization.main)
        localizationThread.start()

        # tracker = Tracker(localization)
        # tracker = Sort()

    mon = monitor.monitor()
    if config.camera['RECORD']:
        video_saver = VideoRecord()

    while True:
        # obtain frame
        ret, depth_image, color_image = cam.get_frame()
        if not ret:
            continue
        # make deep copy of color image
        if config.camera['RECORD']:
            video_saver.add_frame(color_image.copy())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # process frame to obtain object detection predicitons
        bboxes, messages = detector.process_frame(
            color_image, config.detection['DETECTION_CONFIDENCE_CUTOFF'], config.detection['NO_TO_DETECT'], config.general['DEBUG_LOGS'])

        # show detections on screen
        if config.detection['DISPLAY']:
            mon.draw_on_image(color_image, bboxes, messages)

        # TODO: Waiting here for refactoring of main code:
        #      get_coordinates in the original main.py currently only returns one bounding box. We would be returning mulitple from it. Then, pass it onto process_frame() to get each of their depths
        #      With the coords. and the depth, we THEN pass those pairs onto the Tracker object to update its state

        # tracker.updateTrackerState(coords_distances)

        # this is subject to change, but its just here to show potential structure of how things would be called
        #closest_target = tracker.get_closest_tracked_robot()
        #final_coords = tracker.get_coords_to_move(closest_target)
        # ---> Send final coords through serial

    if config.detection['IMU']:
        localizationThread.stop()


if __name__ == '__main__':
    main()
