import cv2
import os

# import flagger #unused
# import uart_server #unused
# import torch.backends.cudnn as cudnn
import uart_comms
import math
import config
import numpy as np

# from tracking.sort import Sort #unused - no tracking on classic CV
from plate_detector import calculate_angle_offsets


import logging
from pathlib import Path

# use njit
from numba import njit, prange
from pathlib import Path
import logging

logging.basicConfig(
    filename=config.general["LOG_PATH"], filemode="a", level=logging.DEBUG
)
logger = logging.getLogger(Path(__file__).stem)

#  Autoselect kernel with best performance on current hardware
# cudnn.benchmark = True

# @TODO: clean up interface


class classic_plate_detector:
    def __init__(
        self,
        tracking=False,
        flagger=None,
        uart=False,
    ):
        # we dont need to predict tracking - fast enough, keeping for now to maintain interface
        self.tracking = tracking

        self.uart = uart_comms.Publisher_node() if uart else None

        # we do not use flagger - keeping for now to maintain interface
        self.flagger = flagger

        # keep track of last offset (y_offset, x_offset)
        self.last_rad_offset = (0, 0)

        # Color FOV for Intel D400 Series cameras
        self.HFOV = config.camera["HFOV"]
        self.VFOV = config.camera["VFOV"]
        self.camera_width = config.camera["FRAME_SIZE"][0]
        self.center_x = self.camera_width / 2  # 326.653747558594#
        self.camera_height = config.camera["FRAME_SIZE"][1]
        self.center_y = self.camera_height / 2  # 237.997650146484#

    def merge_boxes(self, boxes, x_val, y_val):
        size = len(boxes)
        if size < 2:
            return boxes

        if size == 2:
            if self.boxes_mergeable(boxes[0], boxes[1], x_val, y_val):
                boxes[0] = self.merge_bounding_boxes(boxes[0], boxes[1])
                del boxes[1]
            return boxes

        boxes = sorted(boxes, key=lambda r: r[0])
        i = size - 2
        while i >= 0:
            if self.boxes_mergeable(boxes[i], boxes[i + 1], x_val, y_val):
                boxes[i] = self.merge_bounding_boxes(boxes[i], boxes[i + 1])
                del boxes[i + 1]
            i -= 1
        return boxes

    def boxes_mergeable(self, box1, box2, x_val, y_val):
        (x1, y1, w1, h1) = box1
        (x2, y2, w2, h2) = box2
        return (
            max(x1, x2) - min(x1, x2) - self.minx_w(x1, w1, x2, w2) < x_val
            and max(y1, y2) - min(y1, y2) - self.miny_h(y1, h1, y2, h2) < y_val
        )

    def minx_w(self, x1, w1, x2, w2):
        return w1 if x1 <= x2 else w2

    def miny_h(self, y1, h1, y2, h2):
        return h1 if y1 <= y2 else h2

    def merge_bounding_boxes(self, box1, box2):
        # Get the minimum and maximum x and y coordinates.
        min_x = min(box1[0], box2[0])
        min_y = min(box1[1], box2[1])
        max_x = max(box1[0] + box1[2], box2[0] + box2[2])
        max_y = max(box1[1] + box1[3], box2[1] + box2[3])
        # Return the new bounding box.
        return np.int0((min_x, min_y, (max_x - min_x), (max_y - min_y)))

    # keeping variables to maintain interface
    def process_frame(self, color_image, conf_thres=0.25, k=1, debug=False):
        messages = []

        lab = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        prepared_frame = cv2.GaussianBlur(src=enhanced_img, ksize=(5, 5), sigmaX=0)

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(prepared_frame, cv2.COLOR_BGR2HSV)

        # reduce brightness
        value = -50.0
        hsv[:, :, 2] = hsv[:, :, 2] - value

        # Define the lower and upper bounds of the blue color range
        # lower_blue = np.array([70, 200, 190])
        # upper_blue = np.array([140, 255, 200])
        lower_blue = config.gameplay["HSV_LOWER_BLUE_C"]
        upper_blue = config.gameplay["HSV_UPPER_BLUE_C"]

        # Create a mask to identify blue pixels
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find all contours in the mask
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create a list to store the blue rectangles
        bboxes = []

        # Iterate over all contours
        for contour in contours:
            # Find the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            # x, y, w, h = x/self.camera_width, y/self.camera_height, w/self.camera_width, h/self.camera_height
            # Add the rectangle to the list of blue rectangles
            #   if abs( w/h - 1.8) <= 0.001:
            #   if w >= 3 and w <= 650 and h >= 1 and h <= 450:
            bboxes.append((x, y, w, h))

            # if len(blue_rectangles) > 0:
            #   blue_rectangles = remove_overlapping_boxes(blue_rectangles)
            # print(len(blue_rectangles))
            # Draw the blue rectangles on the frame
            bboxes = self.merge_boxes(bboxes, x_val=100, y_val=10)

        # print(type(bboxes))
        # maintain the interface
        dummy = np.array([1, 1])
        bboxes = [np.append(sublist, dummy, axis=0) for sublist in bboxes]
        return bboxes, messages

        # used to print bbox and center of bbox
        # for i, rectangle in enumerate(blue_rectangles):
        #     roi = frame_c[rectangle[1]:rectangle[1]+rectangle[3], rectangle[0]:rectangle[0]+rectangle[2]]
        #     # check if roi positive classification from the model
        #     # test_dl = model(TF.to_tensor(cv2.resize(roi,(32, 32), interpolation = cv2.INTER_AREA)).unsqueeze(0))
        #     # threshold = 0.7

        #     # # Apply sigmoid activation
        #     # sigmoid_tensor = torch.sigmoid(test_dl)
        #     # print(sigmoid_tensor)
        #     # classification = np.argmax(torch.where(sigmoid_tensor > threshold, torch.tensor(1), torch.tensor(0)).numpy())
        #     # print(classification)
        #     # if classification == 1:
        #     cv2.rectangle(frame, rectangle, (0, 255, 0), thickness=2)
        #     cv2.circle(frame, (rectangle[0]+rectangle[2]//2,rectangle[1]+rectangle[3]//2), radius=3, color=(0, 0, 255), thickness=-1)
        #     # roi = frame[rectangle[1]:rectangle[1]+rectangle[3], rectangle[0]:rectangle[0]+rectangle[2]]
        #     # cv2.imwrite(f'frames/roi{i}.png', roi)

    # @TODO: write unified detection library with common utils
    def send_angles(self, x_min, x_max, y_min, y_max, stop=False, debug=True):
        """
        Sends the pitch and yaw offsets to the uart server
        Parameters:
        """
        if stop:
            return
            msg = [0, 0, 0]

            # @TODO see if norm is correct for already normalized values
            # if np.linalg.norm(self.last_rad_offset) > 1.5:
            # check if last_rad_offset was large (target went out of frame)
            # then we should send the last rad offset to continue tracking
            # send the last rad offset /6
            # msg = [
            #    self.last_rad_offset[0] / 6,
            #    self.last_rad_offset[1] / 6,
            #    0,
            # ]
            # self.last_rad_offset = (0, 0)
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
            msg = [y_rad_offset / 3, -x_rad_offset / 3, has_target]

            self.last_rad_offset = (y_rad_offset, -x_rad_offset)
        assert self.uart is not None
        logger.debug(f"MSG: {msg}")
        # for i in range(2):
        self.uart.send_auto_aim(msg)

        if debug and not stop:
            logger.debug(
                "Plate detected at \n({:.3f},{:.3f})\n\n({:.3f},{:.3f})".format(
                    x_min, y_min, x_max, y_max
                ),
                end="\r",
            )

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

        # @TODO: will refactor this later
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
        # print("XRAD: ", x_rad_offset)
        # print("YRAD: ", y_rad_offset)
        logger.debug(f"x= {-x_rad_offset}, y= {y_rad_offset}, target= {has_target} ")

        # if self.uart is not None:
        try:
            self.send_angles(
                x_min, x_max, y_min, y_max, stop=stop_shooting, debug=debug
            )
        except Exception as e:
            logger.debug(f"Error sending angles: {e}")

        return bboxes, messages
