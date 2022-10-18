

from utils.camera import imu


class Localization:
    
    def __init__(self,imu: imu):
        self.position = (0.0,0.0,0.0) #x,y,angle
        self.imu = imu

    #Main loop for localization
    def main(self):

        while True:
            #Each loop will be occuring at the fastest polling rate (based on IMU)
            print(self.imu.get_frame())


        #Need to have a thread-safe access for main thread to access the robot's position


    
