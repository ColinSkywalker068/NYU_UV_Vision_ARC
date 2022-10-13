import cv2
import torch
import os

class plate_detector:
    
    def __init__(self, repo='ultralytics/yolov5', model='custom', model_file_path='./Algorithm/pt_files/best.pt', tracking=True):
        self.model = torch.hub.load(repo, 'custom', path=model_file_path)  # or yolov5m, yolov5l, yolov5x, custom
        if tracking:
            self.tracking = True
            self.tracker = cv2.legacy.TrackerMedianFlow_create()
            self.tracker_initialized = False
        else:
            self.tracking = False


    # Process a color frame
    def process_frame(self, color_image, conf_thres = 0.25, k=1, debug=False): 
        
        # Get bounding boxes
        results = self.model(color_image)

        # Post process bounding boxes
        detections_rows = results.pandas().xyxy

        for i in range(len(detections_rows)):
            rows = detections_rows[i].to_numpy()

        # Go through all detections
        top_k_detections = rows[:k]
        
        bboxes = []
        messages = []
        
        for i in range(len(top_k_detections)):
            # Get the bounding box of the first object (most confident)
            x_min, y_min, x_max, y_max, conf, cls, label = top_k_detections[i]
            if self.tracking:
                if not self.tracker_initialized:
                    tracker_success = self.tracker.init(color_image, (x_min, y_min, x_max, y_max))
                    self.tracker_initialized = True
                else:
                    
                    # Update tracker
                    tracker_success, bbox = self.tracker.update(color_image)
                    bboxes.append(bbox)
#                     x_min, y_min, w, h = bbox
#                     x_max = x_min + w
#                     y_max = y_min + h
                    if tracker_success:
                        bboxes.append([x_min, y_min, x_max, y_max, conf, label])
                        messages.append("Tracking Success")
                    else:
                        print('Tracking Failure')
                        messages.append('Tracking failed for {}'.format(i))
            else:
                bbox = [x_min, y_min, x_max, y_max, conf, label]
                bboxes.append(bbox)
                messages.append("")
        
                
            if debug:
                print("Object {} detected at ({},{}) \n\n\n                     ({},{})".format(i, x_min, y_min, x_max, y_max))
                os.system('cls')
                os.system('clear')
                
        return bboxes, messages
                    
