import cv2
import numpy as np


class Flagger:

    def __init__(self, opponent_color, red_lower, red_upper, blue_lower, blue_upper):
        """
        Constructor.
        opponent_color: 'RED' or "BLUE' indicating the color of the opponent team.
        red_lower: tuple indicating a HSV value indicating the lower limit of red color
        red_upper: tuple indicating a HSV value indicating the upper limit of red color
        blue_lower: tuple indicating a HSV value indicating the lower limit of blue color
        blue_upper: tuple indicating a HSV value indicating the upper limit of blue color
        """
        self.opponent_color = opponent_color
        self.color_ranges = {'red':
                             {
                                 'lower': np.array(red_lower),
                                 'upper': np.array(red_upper)
                             },
                             'blue':
                                 {
                                     'lower': np.array(blue_lower),
                                     'upper': np.array(blue_upper)
                             },
                             }

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
        if color == self.opponent_color:
            return True
        return False

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
        print(len(self.color_ranges['red']['lower']))

        # Get Red masks
        red_mask = cv2.inRange(
            hsv, self.color_ranges['red']['lower'], self.color_ranges['red']['upper'])

        # Get Blue masks
        blue_mask = cv2.inRange(
            hsv, self.color_ranges['blue']['lower'], self.color_ranges['blue']['upper'])

        # Get the contours of the red and blue regions
        red_contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Determine which is greater
        if len(red_contours) > 0 and len(blue_contours) > 0:
            r_area = 0
            b_area = 0
            for c in red_contours:
                r_area += cv2.contourArea(c)
            for c in blue_contours:
                b_area += cv2.contourArea(c)
            if r_area > b_area:
                return 'RED'
            else:
                return 'BLUE'
        elif len(red_contours) > 0:
            return 'RED'

        return 'BLUE'
