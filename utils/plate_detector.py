# import cv2 #unused
import torch
import os

# import flagger #unused
# import uart_server #unused
import torch.backends.cudnn as cudnn
import uart_comms
import math
import config
import numpy as np
from tracking.sort import Sort
import matplotlib.pyplot as plt

import logging
from pathlib import Path

# use jit
import torch.jit as jit

logger = logging.getLogger(Path(__file__).stem)
logger.setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
#  Autoselect kernel with best performance on current hardware
cudnn.benchmark = True


class plate_detector:
    # Constructor
    def __init__(
        self,
        repo="ultralytics/yolov5",
        model="custom",
        model_file_path="./Algorithm/pt_files/best.engine",
        tracking=True,
        flagger=None,
        uart=False,
    ):
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
        # self.model = torch.hub.load(
        #     repo, 'custom', path=model_file_path).eval()

        # TODO: start using python logging module
        device = torch.device("cpu")

        try:
            if torch.backends.mps.is_available():
                print("MPS available")
                device = torch.device("mps")
        except:
            print("MPS not available")
            if torch.cuda.is_available():
                print("CUDA available")
                device = torch.device("cuda:0")
            else:
                print("CUDA/MPS not available")

        self.model = torch.hub.load(repo, "custom", path=model_file_path, device=device)

        # Deactivate Dropout layers
        self.model.eval()  # set model to evaluation mode

        # Check if tracking is required
        self.tracking = tracking

        self.tracker = Sort(max_age=5, min_hits=1) if self.tracking else None

        self.flagger = flagger

        self.uart = uart_comms.Publisher_node() if uart else None

        # keep track of last offset (y_offset, x_offset)
        self.last_rad_offset = (0, 0)

        # Color FOV for Intel D400 Series cameras
        self.HFOV = config.camera["HFOV"]
        self.VFOV = config.camera["VFOV"]
        self.camera_width = config.camera["FRAME_SIZE"][0]
        self.center_x = self.camera_width / 2
        self.camera_height = config.camera["FRAME_SIZE"][1]
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
        stop_shooting = True
        results = self.model(color_image)
        bboxes, messages = self.get_bboxes_from_results(
            color_image, results, conf_thres, k
        )

        if self.tracker is not None:
            bboxes = self.update_tracker(bboxes)

        # @TODO: only sends most confident plate
        # update metric used
        # send bbox at beginning of frame
        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = [0, 0, 0, 0]

        if len(bboxes) > 0:
            x_min, y_min, x_max, y_max, _, _ = bboxes[0]
            stop_shooting = False

            # 1. Check get_auto_aim_request() in uart_server.py if we shold send data. if true then we send angle offsets

            # @TODO this might block, test performance
            # if self.uart.Uart.get_auto_aim_request():
        if self.uart is not None:
            try:
                self.send_angles(
                    x_min, x_max, y_min, y_max, stop=stop_shooting, debug=debug
                )
            except Exception as e:
                print("Error sending angles: ", e)

        return bboxes, messages

    def get_bboxes_from_results(self, color_image, results, conf_thres=0.25, k=1):
        """
        Returns bounding boxes from the results of the model
        Parameters:
            results: results of the model
            conf_thres: confidence threshold
            k: no. of bounding boxes to return
        Returns:
            bboxes: list of bounding boxes
            messages: list of messages to be printed on the image
        """
        messages = []
        # use numpy (faster)
        # 4th column is confidence; for every row in the results (xyxy) that we
        # copy to main memory
        top_k_detections = [
            row[row[:, 4] >= conf_thres]
            for row in [result.detach().cpu().numpy() for result in results.xyxy]
        ][0][:k]
        # use [0] bc tensor, [:k] bc top k
        # print(top_k_detections)
        assert self.flagger is not None, "Flagger not initialized"
        bboxes = [
            bbox[:6]
            for bbox in top_k_detections
            # check only track enemy plates and label is 0
            # we have to use self.flagger.check_enemy_plate() to check
            if (
                self.flagger.check_enemy(
                    color_image[
                        int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
                    ]
                )
                and bbox[5] == 0
            )
        ]

        return bboxes, messages

    def update_tracker(self, bboxes):
        """
        Updates the tracker with the new bounding boxes and returns the updated bounding boxes
        Parameters:
            bboxes: list of bounding boxes
        """
        assert self.tracker is not None, "Tracker not initialized"

        dets = np.array(bboxes)
        # ensure [[],] format
        if len(dets) == 0:
            dets = np.array([[]])
        elif len(dets.shape) == 1:
            dets = np.array([dets])

        # if length of each row is not 7, append 0s
        if len(dets.shape) == 2 and dets.shape[1] != 7:
            dets = np.append(dets, np.zeros((dets.shape[0], 7 - dets.shape[1])), axis=1)
        # print(dets)
        tracker_output = self.tracker.update(dets)
        # print(tracker_output)
        # add label to each row
        bboxes = [bbox.tolist() for bbox in tracker_output if not np.isnan(bbox).any()]
        # print(bboxes)
        return bboxes

    def send_angles(self, x_min, x_max, y_min, y_max, stop=False, debug=False):
        """
        Sends the pitch and yaw offsets to the uart server
        Parameters:
        """
        if stop:
            msg = [0, 0, 0]

            # @TODO see if norm is correct for already normalized values
            if np.linalg.norm(self.last_rad_offset) > 1.5:
                # check if last_rad_offset was large (target went out of frame)
                # then we should send the last rad offset to continue tracking
                # send the last rad offset /6
                msg = [
                    self.last_rad_offset[0] / 6,
                    self.last_rad_offset[1] / 6,
                    0,
                ]
                self.last_rad_offset = (0, 0)
        else:
            # get the pitch and yaw offsets
            x_rad_offset, y_rad_offset = self.calculate_angle_offsets(
                ((x_max + x_min) // 2, (y_max + y_min) // 2)
            )
            # 2. send the offsets to the uart server
            # pitch, yaw
            # if the L2 norm of the offsets from 0 is <= 0.175 rad (+- 10 degrees x, y), start firing

            has_target = int(np.linalg.norm([x_rad_offset, y_rad_offset]) <= 0.175)
            y_rad_offset /= 3
            x_rad_offset /= -3
            msg = [y_rad_offset, x_rad_offset, has_target]
            self.last_rad_offset = (y_rad_offset, x_rad_offset)
        assert self.uart is not None
        for i in range(2):
            self.uart.send_auto_aim(msg)

        if debug and not stop:
            print(
                "Plate detected at \n({:.3f},{:.3f})\n\n({:.3f},{:.3f})".format(
                    x_min, y_min, x_max, y_max
                ),
                end="\r",
            )

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
        y_roi = self.camera_height - y_roi
        move_x = x_roi - self.center_x
        move_y = y_roi - self.center_y
        if move_x != 0:
            move_x /= self.center_x
        if move_y != 0:
            move_y /= self.center_y
        move_x *= self.HFOV / 2
        move_y *= self.VFOV / 2
        move_x = math.radians(move_x)
        move_y = math.radians(move_y)
        # print("x angle offset %f y angle offset %f", move_x, move_y)
        return (move_x, move_y)
