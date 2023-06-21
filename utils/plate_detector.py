#import cv2 #unused
import torch
torch.no_grad()
import os

import flagger 
import uart_server 
import torch.backends.cudnn as cudnn
import uart_comms
import math
import config
import numpy as np
from tracking.sort import Sort

import logging
from pathlib import Path

# use njit
from numba import njit,prange
from pathlib import Path
import logging

logging.basicConfig(
    filename=config.general["LOG_PATH"], filemode="a", level=logging.DEBUG
)
logger = logging.getLogger(Path(__file__).stem)

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
                logger.debug("MPS available")
                device = torch.device("mps")
        except:
            logger.debug("MPS not available")
            if torch.cuda.is_available():
                logger.debug("CUDA available")
                device = torch.device("cuda:0")
            else:
                logger.debug("CUDA/MPS not available")

        self.model = torch.hub.load(repo, "custom", path=model_file_path, device=device)

        # Deactivate Dropout layers
        self.model.eval()  # set model to evaluation mode

        # Check if tracking is required
        self.tracking = tracking

        self.tracker = Sort(max_age=10, min_hits=0) if self.tracking else None

        self.uart = uart_comms.Publisher_node() if uart else None

        # pass reference uart to flagger object
        self.flagger = flagger

        # keep track of last offset (y_offset, x_offset)
        self.last_rad_offset = (0, 0)

        # Color FOV for Intel D400 Series cameras
        self.HFOV = config.camera["HFOV"]
        self.VFOV = config.camera["VFOV"]
        self.camera_width = config.camera["FRAME_SIZE"][0]
        self.center_x = self.camera_width / 2#326.653747558594#
        self.camera_height = config.camera["FRAME_SIZE"][1]
        self.center_y = self.camera_height / 2#237.997650146484#

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
        #if self.uart is not None and not self.uart.Uart.get_auto_aim_request():
        #    return [], []

        # Get bounding boxes
        results = self.model(color_image)
        bboxes, messages = self.get_bboxes_from_results(
            color_image, results, conf_thres, k
        )
        return bboxes, messages

    def send_msg(self, bboxes, messages, debug=False):
        # @TODO: only sends most confident plate
        # update metric used
        # send bbox at beginning of frame
        (
            x_min,
            y_min,
            x_max,
            y_max,
        ) = [0, 0, 0, 0]

        stop_shooting = True
        if len(bboxes) > 0:
            x_min, y_min, x_max, y_max, _, _ = bboxes[0]
            stop_shooting = False


        #@TODO: will refactor this later
        (x_rad_offset, y_rad_offset) = calculate_angle_offsets(
            ((x_max + x_min) // 2, (y_max + y_min) // 2),
            self.center_x,
            self.center_y,
            self.camera_height,
            self.HFOV,
            self.VFOV,
        )
        # 2. send the offsets to the uart server
        # pitch, yaw
        # if the L2 norm of the offsets from 0 is <= 0.175 rad (+- 10 degrees x, y), start firing

        has_target = int(np.linalg.norm([x_rad_offset, y_rad_offset]) <= 0.175)
        #print("XRAD: ", x_rad_offset)
        #print("YRAD: ", y_rad_offset)
        logger.debug(f"x= {-x_rad_offset}, y= {y_rad_offset}, target= {has_target} ")

        #if self.uart is not None:
        try:
            self.send_angles(
                x_min, x_max, y_min, y_max, stop=stop_shooting, debug=debug
            )
        except Exception as e:
            logger.debug(f"Error sending angles: {e}")

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
                        bbox[1].astype(int) : bbox[3].astype(int),
                        bbox[0].astype(int) : bbox[2].astype(int),
                    ]
                )
                # and bbox[5] == 0
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
        valid_mask = np.logical_not(np.isnan(tracker_output).any(axis=1))
        valid_bboxes = tracker_output[valid_mask]
        bboxes = np.array(valid_bboxes)

        # print(bboxes)
        return bboxes

    def send_angles(self, x_min, x_max, y_min, y_max, stop=False, debug=True):
        """
        Sends the pitch and yaw offsets to the uart server
        Parameters:
        """
        if stop:
            return
            msg = [0, 0, 0]

            # @TODO see if norm is correct for already normalized values
            #if np.linalg.norm(self.last_rad_offset) > 1.5:
                # check if last_rad_offset was large (target went out of frame)
                # then we should send the last rad offset to continue tracking
                # send the last rad offset /6
                #msg = [
                #    self.last_rad_offset[0] / 6,
                #    self.last_rad_offset[1] / 6,
                #    0,
                #]
                #self.last_rad_offset = (0, 0)
        else:
            # get the pitch and yaw offsets
            x_rad_offset, y_rad_offset = calculate_angle_offsets(
                ((x_max + x_min) / 2, (y_max + y_min) / 2),
                self.center_x,
                self.center_y,
                self.camera_height,
                self.HFOV,
                self.VFOV,
            )
            # 2. send the offsets to the uart server
            # pitch, yaw
            # if the L2 norm of the offsets from 0 is <= 0.175 rad (+- 10 degrees x, y), start firing

            has_target = int(np.linalg.norm([x_rad_offset, y_rad_offset]) <= 0.175)
            
            if y_rad_offset < -0.5:
                y_rad_offset = -0.5
            elif y_rad_offset > 0.5:
                y_rad_offset = 0.5

            if x_rad_offset < -0.5:
                x_rad_offset = -0.5
            elif x_rad_offset > 0.5:
                x_rad_offset = 0.5
            # bound x and y offset each by 0.1
            msg = [y_rad_offset/3, -x_rad_offset/3, has_target]

            self.last_rad_offset = (y_rad_offset, -x_rad_offset)
        assert self.uart is not None
        logger.debug(f"MSG: {msg}")
        #for i in range(2):
        self.uart.send_auto_aim(msg)

        if debug and not stop:
            logger.debug(
                "Plate detected at \n({:.3f},{:.3f})\n\n({:.3f},{:.3f})".format(
                    x_min, y_min, x_max, y_max
                ),
                end="\r",
            )


@njit(cache=True)
def calculate_angle_offsets(
    roi_center_offsets, center_x, center_y, camera_height, HFOV, VFOV
):
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
    y_roi = camera_height - y_roi
    move_x = x_roi - center_x
    move_y = y_roi - (center_y * 0.8)
    if move_x != 0:
        move_x /= center_x
    if move_y != 0:
        move_y /= center_y
    move_x *= HFOV / 2
    move_y *= VFOV / 2
    move_x, move_y = math.radians(move_x), math.radians(move_y)
    return move_x, move_y


@njit(cache=True)
def sort_bboxes_by_distance(bboxes, depth_image):
    """
    Sorts the bboxes by distance from the camera
    Parameters:
        bboxes: list of bounding boxes
        depth_image: depth image
    Returns:
        sorted_bboxes: sorted list of bounding boxes by median plate depth
    """
    num_bboxes = len(bboxes)
    depth_values = np.zeros((num_bboxes,))
    sorted_bboxes = np.zeros((num_bboxes, 6))

    for i in range(num_bboxes):
        bbox = bboxes[i]
        x_min, y_min, x_max, y_max, _, _ = bbox
        # get the depth values of the plate
        depth_values[i] = np.median(
            depth_image[int(y_min) : int(y_max), int(x_min) : int(x_max)].flatten()
        )
    # sort the bboxes by median depth
    sorted_indices = np.argsort(depth_values)
    
    for i in range(num_bboxes):
        sorted_bboxes[i, :] = bboxes[sorted_indices[i]]
    return sorted_bboxes
