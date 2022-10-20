import json
from threading import Thread
import config
import camera
import plate_detector
import monitor
from localization.localization import *
import cv2
from utils.localization.tracking import Tracker

def main():
    
    if config.camera['DEPTH_CAPTURE']:
        cam = camera.depth_camera(config.camera['FRAME_SIZE'], config.camera['FRAME_RATE'])
    else:
        cam = camera.camera(config.camera['CAMERA_INDEX'])
        
    imu = camera.imu()
    #Might need another file for encoder help, which we also pass into Localization
    localization = Localization(imu)
    localizationThread = Thread(target = localization.main)
    localizationThread.start()

    tracker = Tracker(localization)

    detector = plate_detector.plate_detector(config.yolo['REPO_PATH'], config.yolo['MODEL_SOURCE'], config.yolo['MODEL_WEIGHTS_PATH'])
    
    mon = monitor.monitor()
    
    while True:
        ret, depth_image, color_image = cam.get_frame()
        if ret:
            key = cv2.waitKey(1)
            if key==27:
                break

            bboxes, messages = detector.process_frame(color_image, config.detection['DETECTION_CONFIDENCE_CUTOFF'], config.detection['NO_TO_DETECT'], config.general['DEBUG_LOGS'])

            if config.detection['DISPLAY']:
                mon.draw_on_image(color_image, bboxes, messages) 

            #TODO: Waiting here for refactoring of main code:
            #      get_coordinates in the original main.py currently only returns one bounding box. We would be returning mulitple from it. Then, pass it onto process_frame() to get each of their depths
            #      With the coords. and the depth, we THEN pass those pairs onto the Tracker object to update its state

            #tracker.updateTrackerState(coords_distances)

            # this is subject to change, but its just here to show potential structure of how things would be called
            closest_target = tracker.get_closest_tracked_robot()
            final_coords = tracker.get_coords_to_move(closest_target)
            # ---> Send final coords through serial

    localizationThread.stop()

if __name__=='__main__':
    main()