import cv2
import numpy as np
from threading import Thread


class Flagger:

    def __init__(self, opponent_color, red_lower, red_upper, red_lower_2, red_upper_2, blue_lower, blue_upper):
        """
        Constructor.
        opponent_color: 'RED' or "BLUE' indicating the color of the opponent team.
        red_lower: tuple indicating a HSV value indicating the lower limit of red color
        red_upper: tuple indicating a HSV value indicating the upper limit of red color
        blue_lower: tuple indicating a HSV value indicating the lower limit of blue color
        blue_upper: tuple indicating a HSV value indicating the upper limit of blue color
        """
        self.opponent_color = opponent_color

        # we will use bitwise or to combine the two masks
        # (there is a better way to do this, we will refactor later)

        self.red_lower = np.array(red_lower, dtype=np.uint8)
        self.red_upper = np.array(red_upper, dtype=np.uint8)
        self.red_lower_2 = np.array(red_lower, dtype=np.uint8)
        self.red_upper_2 = np.array(red_upper, dtype=np.uint8)
        self.blue_lower = np.array(blue_lower, dtype=np.uint8)
        self.blue_upper = np.array(blue_upper, dtype=np.uint8)

    # Check if the robot in the frame is enemy robot or not. Returns true if it is enemy robot
    def check_enemy(self, frame):
        """
        Checks if the frame contains the robot of enemy team or ally.
        Paramter:
        frame = input image

        Returns:
        True if enemy robot detected else False
        """
        color = self.red_or_blue(frame)
        return color == self.opponent_color and color != 'NONE'

    # determines if the robot in the frame is of red or blue color

    def red_or_blue(self, frame):
        """
        Checks what is the color of the robot in the image.
        Parameters:
        frame = input image

        Returns:
        "RED" if red is detected the most else "BLUE"
        """

        # Convert BGR format ot HSV format
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get Red masks
        red_mask = cv2.inRange(
            hsv, self.red_lower, self.red_upper)

        red_mask_2 = cv2.inRange(
            hsv, self.red_lower_2, self.red_upper_2)

        red_mask = cv2.bitwise_or(red_mask, red_mask_2)

        # Get Blue masks
        blue_mask = cv2.inRange(
            hsv, self.blue_lower, self.blue_upper)

        # Get the contours of the red and blue regions
        red_contours, _ = cv2.findContours(
            red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        r_area = np.sum([cv2.contourArea(
            c) for c in red_contours if (cv2.contourArea(c)) != 0])
        b_area = np.sum([cv2.contourArea(
            c) for c in blue_contours if (cv2.contourArea(c)) != 0])

        if r_area > b_area:
            return "RED"
        elif b_area > r_area:
            return "BLUE"
        return "NONE"
