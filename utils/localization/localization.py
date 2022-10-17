#Main loop for localization



def main():
    while True:
        #Each loop will be occuring at the fastest polling rate (based on IMU)
        pass


class Localization:
    
    def __init__(self):
        self.position = (0.0,0.0,0.0) #x,y,angle


    #Need to have a thread-safe access for main thread to access the robot's position


    
