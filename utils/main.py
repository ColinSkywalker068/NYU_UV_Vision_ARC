import json
from threading import Thread
import config
import camera
import plate_detector
import monitor
from localization.localization import *
import cv2

def main():
    
    if config.camera['DEPTH_CAPTURE']:
        cam = camera.depth_camera(config.camera['FRAME_SIZE'], config.camera['FRAME_RATE'])
    else:
        cam = camera.camera(config.camera['CAMERA_INDEX'])
        
    imu = camera.imu()
    localization = Localization(imu)
    localizationThread = Thread(target = localization.main)
    localizationThread.start()

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

    localizationThread.stop()

if __name__=='__main__':
    main()