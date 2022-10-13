import cv2

class flagger:
    
    def __init__(self, opponent_color, red_lower, red_upper, blue_lower, blue_upper):
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
                                     
    def check_enemy(self, frame):
        color = red_or_blue(self, frame)
        if color==opponent_color:
            return True
        else:
            return False
        
    def red_or_blue(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, self.color_ranges.red_lower, self.color_ranges.red_upper)
        blue_mask = cv2.inRange(hsv, self.color_ranges.blue_lower, self.color_ranges.blue_upper)
        _, red_contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, blue_contours, _ = cv2.findContours(
            blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(red_contours) > 0 and len(blue_contours) > 0:
            r_area = 0
            b_area = 0
            for c in red_contours:
                r_area += cv2.contourArea(c)
            for c in blue_contours:
                b_area += cv2.contourArea(c)
            if r_area > b_area:
                return 'r'
            else:
                return 'b'
        elif len(red_contours) > 0:
            return 'r'

        return 'b'