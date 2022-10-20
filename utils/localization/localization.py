

from typing import Tuple
from utils.camera import imu_sensor


class Localization:
    
    def __init__(self,imu_pipeline: imu_sensor) -> None:
        self.position = (0.0,0.0,0.0) #x,y,angle
        self.imu_pipeline = imu_pipeline

    def get_robot_position(self) -> Tuple[float,float,float]:
        return self.position

    #Main loop for localization
    def main(self) -> None:

        while True:
            #Each loop will be occuring at the fastest polling rate (based on IMU)
            print(self.imu_pipeline.get_frame())


            #Need to have a thread-safe access for main thread to access the robot's position
            #TODO: Need to implement math to update position
    
