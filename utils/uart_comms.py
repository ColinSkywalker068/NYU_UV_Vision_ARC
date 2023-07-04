import uart_server
import time
import threading
import struct
import uart_server
import config
import os
from pathlib import Path
import logging
from uart_distance import UartDistance
import numpy as np
import multiprocessing as mp
logging.basicConfig(
    filename=config.general["LOG_PATH"],
    filemode='a',
    level = logging.DEBUG)
logger = logging.getLogger(Path(__file__).stem)


class Publisher_node:
    def __init__(self, Uart_server=None):
        self.Uart = Uart_server
        if self.Uart is None:
            self.Uart = uart_server.Uart()
        self.header = 0xE7


        # Autonomous
        if config.gameplay['Autonomous']: 
            self.UD = UartDistance()
            self.thread_ultrasonic = threading.Thread(target=self._start_ultrasonic)
            self.thread_ultrasonic.start()
        

        # start thread
        self.thread = threading.Thread(target=self._start)
        self.thread.start()
        

    def _start_ultrasonic(self):
        
        while True:
            if(self.UD.connection.in_waiting >0):
                self.UD.read_sensor()
                self.ultrasonic_callback(self.UD.distance)
            
            

    # called as a thread
    def _start(self):
        while True:
            if self.Uart is None:
                try:
                    self.Uart = uart_server.Uart()
                except:
                    if config.detection["DEBUG"]:

                        logger.debug("Uart not initialized1")
                    continue
            try:
                self.Uart.listener()

            except:
                # print("Uart not initialized2")
                if config.detection["DEBUG"]:
                    logger.debug("Uart not initialized2")

            # self.Uart.listener()
            # delay to prevent cpu overload
    # AI-powered autonomous driving (it's just if-statements, give me a break)
    def ultrasonic_callback(self, msg):
        # msg=[50 if number > 1200 else number for number in msg]
        print("Ultrasonic: ", msg)
        # return
        # msg = [front, left, right, back]
        # we send an align request as appropriate depending on the distance
        # we can send 1 message at a time, each with the forward and backward velocity, then we send a message 0.2 seconds later to stop the motors



        obsticals= [x < config.uart_ultrasonic["MIN_DISTANCE"] for x in msg]
        power = 0.2
        r_power=0.7
        time_s=0.3
        # 0: no-bayblade, 1: bayblade, 2: reverse-bayblade
        beyblade=2

        # random 0.2 chance to turn left only
        # if np.random.rand() < 0.01:
        #     r_power = -1

        

        # obstical in front
        if obsticals[0]:
            # obstical in front, left, and not right => turn right
            if obsticals[1] and not obsticals[2]:
                for i in range(2):
                    # self.send_align_request([power, 0,0,beyblade])
                    self.send_align_request([0, 0,r_power,beyblade])
                # time.sleep(time_s*2)
            # obstical in front, right, and not left => turn left
            elif obsticals[2] and not obsticals[1]:
                for i in range(2):
                    # self.send_align_request([-power, 0,0,beyblade])
                    self.send_align_request([0, 0,r_power,beyblade])
                # time.sleep(time_s*2)
            # obstical in front, left, and right => turn around
            elif obsticals[2] and obsticals[1]:
                for i in range(2):
                    # self.send_align_request([ 0,-power,0,beyblade])
                    self.send_align_request([ 0,0,r_power,beyblade])
                # time.sleep(time_s*4)
            else:
                # obstical in front and not left or right
                # turn right
                for i in range(2):
                    self.send_align_request([power, 0,0,beyblade])
                    # self.send_align_request([0, 0,r_power,beyblade])
                # time.sleep(time_s*2)
        else:
            # no obstical in front
            # go forward
            if obsticals[1] and not obsticals[2]:
                # move right
                for i in range(2):
                    self.send_align_request([power/2, power/2,0,beyblade])
            elif obsticals[2] and not obsticals[1]:
                # move left
                for i in range(2):
                    self.send_align_request([-power, -power/2,0,beyblade])
            else:
                for i in range(2):
                    self.send_align_request([0,power,0,beyblade])

        # if obstical in front, left, and not right then turn right
        # if obsticals[0] and obsticals[1] and not obsticals[2]:
        #     for i in range(2):
        #         self.send_align_request([0, 0,power,0])
        # # if obstical in front, right, and not left then turn left
        # elif obsticals[0] and obsticals[2] and not obsticals[1]:
        #     for i in range(2):
        #         self.send_align_request([0, 0,-power,0])
        # # if obstical in front, right, left, and not back then turn around
        # elif obsticals[0] and obsticals[2] and obsticals[1]:
        #     for i in range(2):
        #         self.send_align_request([ -power/2.0,0,power/2,0])
        # # if no obstical in front, go forward
        # elif not obsticals[0]:
        #     for i in range(2):
        #         self.send_align_request([power,0,0,0])
        # time.sleep(time_s)
        # # if obstical in front, right, left, and back then stop
        # for i in range(2):
        #     self.send_align_finish(msg=[1])
        



    def send_auto_aim(self, msg):
        # msg = [pitch, yaw, has_target]
        # [y,x]
        length = 11
        msg_type = 1
        header = struct.pack("BHBBH", self.header, length, 0x00, 0x00, msg_type)

        body = struct.pack("iiB", int(msg[0] * 1000000), int(msg[1] * 1000000), int(msg[2]))
        footer = struct.pack("H", 0x0)

        msg = header + body + footer
        # print(msg)
        # check if uart is initialized
        # if so send the message, if not try to initialize and then send the message
        if self.Uart is None:
            try:
                self.Uart = uart_server.Uart()
            except:
                if config.detection["DEBUG"]:
                    logger.debug("Uart not initialized3")
                return
        try:
            self.Uart.send_data(msg)
        except:
            if config.detection["DEBUG"]:
                logger.debug("Uart not initialized4")

    def send_align_request(self, msg):
        # msg = [x,y,r, type]
        length = 9
        msg_type = 2

        header = struct.pack("BHBBH", self.header, length, 0x00, 0x00, msg_type)

        body = struct.pack(
            "hhhB",
            int(msg[0] * 1000),
            int(msg[1] * 1000),
            int(msg[2] * 1000),
            int(msg[3]),
        )
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        self.Uart.send_data(msg)

    def send_align_finish(self, msg):
        if not msg[0]:
            return

        length = 0x03
        msg_type = 0x03
        header = struct.pack("BHBBH", self.header, length, 0x00, 0x00, msg_type)
        body = struct.pack("B", 0x01)
        footer = struct.pack("H", 0x0)
        msg = header + body + footer

        self.Uart.send_data(msg)

    def send_gimbal_movement(self, msg):
        # msg = [gimbal_yaw, gimbal_pitch,mode= 0: base, 1: beyblade, 2: backward beyblade]
        # yaw and pitch should be between -1 and 1
        # yaw (x, + is anti-clockwise)
        # pitch (y, + is up)
        length = 7
        msg_type = 4
        header = struct.pack("BHBBH", self.header, length, 0x00, 0x00, msg_type)
        body = struct.pack("hhB", int(msg[0] * 1000), int(msg[1] * 1000), int(msg[2]))
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        self.Uart.send_data(msg)

# 5: move straight
    def send_move_straight(self,msg):
        # [int distance (mm), int velocity*1000 (m/s)]

        length = 6
        msg_type=5
        header = struct.pack("BHBBH", self.header, length, 0x00, 0x00, msg_type)
        body = struct.pack("Hh", int(msg[0]), int(msg[1] * 1000))
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        self.Uart.send_data(msg)

# 6: spin chasis
    def send_spin_chassis(self,msg):
        # [angle (deg), velocity *1000(deg/s)]
        length = 6
        msg_type=6
        header = struct.pack("BHBBH", self.header, length, 0x00, 0x00, msg_type)
        body = struct.pack("hh", int(msg[0]), int(msg[1] * 1000))
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        self.Uart.send_data(msg)
# 7: set power
    def send_set_power(self,msg):
        # [xLinearPower*100, yLinearPower*100, xAnglePower*100, yAnglePower*100], all in range 0-100
        length = 10
        msg_type=7
        header = struct.pack("BHBBH", self.header, length, 0x00, 0x00, msg_type)
        body = struct.pack("HHHH", int(msg[0]*100), int(msg[1]*100),int(msg[2]*100),int(msg[3]*100))
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        self.Uart.send_data(msg)

# 8: set_velocity
    def send_set_velocity(self,msg):
        # [xVelocity,yVelocity,yawVelocity,pitchVelocity]
        length = 10
        msg_type=8
        header = struct.pack("BHBBH", self.header, length, 0x00, 0x00, msg_type)
        body = struct.pack("hhhh", int(msg[0]*100), int(msg[1]*100),int(msg[2]*100),int(msg[3]*100))
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        self.Uart.send_data(msg)
# 9: stop
    def send_stop_chassis(self,msg):
        length = 3
        msg_type=9
        header = struct.pack("BHBBH", self.header, length, 0x00, 0x00, msg_type)
        body = struct.pack("?", 1)
        footer = struct.pack("H", 0x0)
        msg = header + body + footer
        self.Uart.send_data(msg)


import sys

if __name__ == "__main__":
    node = Publisher_node(None)
    # count=1
    # for i in range(2*count):
    #     #node.send_align_request([-0.3, -0.3, 0, 0])
    #     node.send_gimbal_movement([0.00,0.0001,0])
    # for i in range(4):
    #     node.send_align_request([-0.3, -0.3, 0, 0])
    #     time.sleep(0.5)
    # for i in range(2):
    #     node.send_align_request([0.2, 0.0, 0, 0])
    # time.sleep(2.3)
    #     # node.send_gimbal_movement([0.1,-0.1,0])
    for i in range(2):
        node.send_align_finish([1])
    # # assert()
    # # print("ID: ", node.Uart.get_robot_id())
    # while True:
    #     pitch, yaw = node.Uart.get_turret_feedback()
    #     #print("(Yaw, Pitch): (", yaw, ", ", pitch, ")")
    #     print("(Pitch deg,  Yaw deg): (", pitch, ", ", yaw, ")",end='\r')
    #     #time.sleep(0.01)
    #     os.system('clear') 
    # while True:
    #     referee_feedback_data = node.Uart.get_referee_feedback()
    #     os.system('clear')
    #     print(referee_feedback_data, end='\r')
        
        # time.sleep(0.01)
        
    # print(node.Uart.handle_align_request())
    # print(node.Uart.get_auto_aim_request())
    # time.sleep(.2)
    #while True:
    #    node.send_auto_aim([-0.1, 0, 0])
    #    time.sleep(0.25)
    #    node.send_auto_aim([0.1, 0, 0])
    #    time.sleep(0.25)

    # node.send_align_request([1, 2, 3])
    # node.send_align_finish([1])
