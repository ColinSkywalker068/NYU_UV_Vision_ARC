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
                 
    #Check if the robot in the frame is enemy robot or not. Returns true if it is enemy robot
    def check_enemy(self, frame):
        color = red_or_blue(self, frame)
        if color==opponent_color:
            return True
        else:
            return False
        
        
    #determines if the robot in the frame is of red or blue color
    def red_or_blue(self, frame):     
        
        # Convert BGR format ot HSV format
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get Red masks 
        red_mask = cv2.inRange(hsv, self.color_ranges.red_lower, self.color_ranges.red_upper)
        
        # Get Blue masks
        blue_mask = cv2.inRange(hsv, self.color_ranges.blue_lower, self.color_ranges.blue_upper)
        
        #Get the contours of the red and blue regions
        _, red_contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _, blue_contours, _ = cv2.findContours(
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