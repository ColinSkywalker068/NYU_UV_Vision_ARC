import cv2
import math

 
class Ballistics:
    def __init__(self,x_offset,y_offset,distance):
        self.coords = [0,0,0]
        self.velocity = [0,0,0]
        self.acceleration = [0,0,0]

        self.target_coords = [0,0,0]
        self.target_velocity = [0,0,0]
        self.target_acceleration = [0,0,0]
