

from typing import Tuple
from camera import imu_sensor


# Class responsible for localization of our robot using IMU/encoders
class Localization:
    
    def __init__(self,imu_pipeline: imu_sensor) -> None:
        self.position = (0.0,0.0,0.0) #x,y,angle
        self.imu_pipeline = imu_pipeline

    #TODO: need to make sure this is thread safe
    def get_robot_position(self) -> Tuple[float,float,float]:
        return self.position

    #Main loop for localization, will be run on separate thread within the original main loop of our program
    def main(self) -> None:

        while True:
            #Each loop will be occuring at the fastest polling rate (based on IMU)
            print(self.imu_pipeline.get_frame())

            #TODO: Need to implement math to update position
    
