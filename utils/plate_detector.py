import cv2
import torch
import os
import flagger
import uart_server
import torch.backends.cudnn as cudnn
import uart_comms
import math
import config

# deep_sort
from Sort import *
from Detection import *
import logging
import matplotlib.pyplot as plt
from pathlib import Path
logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
#  Autoselect kernel with best performance on current hardware
cudnn.benchmark = True


class plate_detector:

    # Constructor
    def __init__(self, repo='ultralytics/yolov5', model='custom', model_file_path='./Algorithm/pt_files/best.engine', tracking=True, flagger=None, uart=False):
        """
        Constructor. Initializes attributes. Uses torch.hub.load() of PyTorch module to load a pytorch model.
        Parameters:
            repo:  url of the repository that has the load scripts for the model
            model: type of model
            model_file_path: path to the weights of the model
            tracking: True if tracking is to be done else False
            flagger: flagger object (Default: None)
        Attributes:
            model: Object detection model object
            tracker: tracker object
            tracking: True if tracking is enabled else False
            tracker_initialized: True if tracker is already initialized else False
        """

        # Create object detection model
        # or yolov5m, yolov5l, yolov5x, custom
        self.model = torch.hub.load(
            repo, 'custom', path=model_file_path).eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Deactivate Dropout layers
        # self.model.eval()  # set model to evaluation mode

        # @TODO: convert to tensorrt

        # Check if tracking is required
        self.tracker_initialized = False
        if tracking:
            # Create the tracker object
            self.tracking = True

            self.tracker = Sort(max_age=30, min_hits=1)
            cmap = plt.get_cmap('tab20b')
            self.colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        else:
            self.tracking = False

        if flagger is not None:
            self.flagger = flagger

        if uart:
            # create a uart object to send/receive data
            self.uart = uart_comms.Publisher_node()
        else:
            self.uart = None

        # Color FOV for Intel D400 Series cameras
        self.HFOV = config.camera['HFOV']
        self.VFOV = config.camera['VFOV']
        self.camera_width = config.camera['FRAME_SIZE'][0]
        self.center_x = self.camera_width / 2
        self.camera_height = config.camera['FRAME_SIZE'][1]
        self.center_y = self.camera_height / 2

    # Process a color frame
    def process_frame(self, color_image, conf_thres=0.25, k=1, debug=False):
        """
        Performs object detection on the input image.
        Filters out predictions.
        Performs tracking.
        Paramters:
        color_image = input image on which the object detection is to be performed.
        conf_thresh = The threshold value for the confidence score of the predictions by the model. The predicitons below the conf_thresh are discarded.
        k = No. of objects to be outputed from the object detection predictions
        debug = Prints debug logs if True

        Returns:
            bboxes: A list of tuples, each tuple containing the 6 specs of each  bounding box that is predicted
            messages: List of strings that are to be printed on the image
        """

        # Get bounding boxes
        results = self.model(color_image)

        # Post process bounding boxes

        # use numpy (faster)
        # x_min, y_min, x_max, y_max, conf, class (0 for plate)
        # no cls (only in pandas)

        # detections_rows = [results.xyxy[i].detach().cpu().numpy()
        # for i in range(len(results.xyxy))]

        # 4th column is confidence; for every row in the results (xyxy) that we
        # copy to main memory
        top_k_detections = [row[row[:, 4] >= conf_thres] for row in [
            result.detach().cpu().numpy() for result in results.xyxy]][0][:k]
        # use [0] bc tensor, [:k] bc top k

        bboxes = []
        messages = []

        if self.tracking:
            dets = [bbox[:6] for bbox in top_k_detections
                    # check only track enemy plates and label is 0
                    # we have to use self.flagger.check_enemy_plate() to check
                    if (self.flagger.check_enemy(color_image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]) and bbox[5] == 0)
                    ]
            dets = np.array(dets)
            # ensure [[],] format
            if len(dets) == 0:
                dets = np.array([[]])
            elif len(dets.shape) == 1:
                dets = np.array([dets])

            # if length of each row is not 7, append 0s
            if len(dets.shape) == 2 and dets.shape[1] != 7:
                dets = np.append(dets, np.zeros(
                    (dets.shape[0], 7 - dets.shape[1])), axis=1)
            # print(dets)
            tracker_output = self.tracker.update(dets)
            # print(tracker_output)
            bboxes = [bbox[:5] for bbox in tracker_output
                      # ensure no NaNs
                      if not np.isnan(bbox).any()]

        # @TODO: we currently only send and print detections that are only inferred by the model, not tracked by DeepSORT. We should send and print both. Code needs to be refactored.
        for i in range(len(top_k_detections)):
            # Get the bounding box of the first object (most confident)
            x_min, y_min, x_max, y_max, conf, label = top_k_detections[i]

            # check if plate is enemy plate
            if label == 0 and self.flagger is not None:
                # (y_min, y_max, x_min, x_max)
                if not self.flagger.check_enemy(color_image[int(y_min): int(y_max), int(x_min): int(x_max)]):
                    continue

            if not self.tracking:
                # Add bounding boxes to be drawn on the image
                bbox = [x_min, y_min, x_max, y_max, conf, label]
                bboxes = np.append(bboxes, bbox)
                # bboxes.append(bbox)
                messages.append("")

            # @TODO: only sends most confident plate
            # update metric used
            if self.uart is not None and i == 0:
                # 1. Check get_auto_aim_request() in uart_server.py if we shold send data. if true then we send angle offsets

                # @TODO this might block, test performance
                if self.uart.Uart.get_auto_aim_request():
                    # get the pitch and yaw offsets
                    x_rad_offset, y_rad_offset = self.calculate_angle_offsets(
                        ((x_max+x_min)//2, (y_max+y_min)//2))

                    # 2. send the offsets to the uart server
                    # pitch, yaw
                    msg = [y_rad_offset, x_rad_offset, 1]
                    self.uart.send_auto_aim(msg)

            if debug:
                print("Object {} detected at ({},{}) \n\n\n ({},{})".format(
                    i, x_min, y_min, x_max, y_max))
                # print for windows
                if os.name == 'nt':
                    os.system('cls')
                else:
                    os.system('clear')

        return bboxes, messages

    def calculate_angle_offsets(self, roi_center_offsets):
        """
        Returns the pitch and yaw offsets of the detected plate
        Parameters:
            bbox: bounding box of the detected plate
        Returns:
            pitch_offset: pitch offset of the detected plate
            yaw_offset: yaw offset of the detected plate
            returned values are in radians from the camera's center
        """
        x_roi, y_roi = roi_center_offsets
        y_roi = self.camera_width - y_roi
        move_x = x_roi - self.center_x
        move_y = y_roi - self.center_y
        if(move_x != 0):
            move_x /= self.center_x
        if(move_y != 0):
            move_y /= self.center_y
        move_x *= self.HFOV/2
        move_y *= self.VFOV/2
        move_x = math.radians(move_x)
        move_y = math.radians(move_y)
        # print("x angle offset %f y angle offset %f", move_x, move_y)
        return (move_x, move_y)
