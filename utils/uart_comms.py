import uart_server
import time
import threading
import struct
import uart_server
import config

from pathlib import Path
import logging
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

        # start thread
        self.thread = threading.Thread(target=self._start)
        self.thread.start()

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
                if config.detection["DEBUG"]:
                    logger.debug("Uart not initialized2")

            # self.Uart.listener()
            # delay to prevent cpu overload
            # time.sleep(0.1)

    def send_auto_aim(self, msg):
        # msg = [pitch, yaw, has_target]
        # [y,x]
        length = 11
        msg_type = 1
        header = struct.pack("BHBBH", self.header, length, 0x00, 0x00, msg_type)

        body = struct.pack("iiB", int(msg[0] * 10000), int(msg[1] * 10000), int(msg[2]))
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
    # for i in range(4):
    #     node.send_align_request([0.0, 0.0, 0, 0])
        #node.send_gimbal_movement([0.1,-0.1,0])
    # for i in range(2):
    #     node.send_align_finish([1])
    # # assert()
    # # print("ID: ", node.Uart.get_robot_id())
    # while True:
    #   pitch, yaw = node.Uart.get_turret_feedback()
    #   debugger.log(f"(Yaw, Pitch): ({yaw} , {pitch})")
    #   time.sleep(0.2)
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
