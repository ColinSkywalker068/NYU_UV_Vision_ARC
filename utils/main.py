import json
import config
import camera
import plate_detector
import monitor
import cv2

def main():
    
    #Create camera object
    if config.camera['DEPTH_CAPTURE']:
        #creates depth camera object
        cam = camera.depth_camera(config.camera['FRAME_SIZE'], config.camera['FRAME_RATE'])
    else:
        # creates vamera object (without depth capture)
        cam = camera.camera(config.camera['CAMERA_INDEX'])
        
    # create detector object
    detector = plate_detector.plate_detector(config.yolo['REPO_PATH'], config.yolo['MODEL_SOURCE'], config.yolo['MODEL_WEIGHTS_PATH'])
    
    # create monitor object to display annotated images
    mon = monitor.monitor()
    
    while True:
        #obtain frame
        ret, depth_image, color_image = cam.get_frame()
        if ret:
            key = cv2.waitKey(1)
            if key==27:
                break

            # process frame to obtain object detection predicitons
            bboxes, messages = detector.process_frame(color_image, config.detection['DETECTION_CONFIDENCE_CUTOFF'], config.detection['NO_TO_DETECT'], config.general['DEBUG_LOGS'])
            
            # show detections on screen
            if config.detection['DISPLAY']:
                mon.draw_on_image(color_image, bboxes, messages) 

if __name__=='__main__':
    main()