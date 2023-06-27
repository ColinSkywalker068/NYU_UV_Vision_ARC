import serial
import struct
import config
import os
from distance_mapper import distance_mapper
import threading
from pathlib import Path
import logging
logging.basicConfig(
    filename=config.general["LOG_PATH"],
    filemode='a',
    level = logging.DEBUG)
logger = logging.getLogger(Path(__file__).stem)
class UartDistance:
    def __init__(self, port="/dev/ttyACM0", baudrate=9600) -> None:
        i = 0
        self.port = port
        self.baudrate = baudrate
        self.read_mutex = threading.Lock()
        # front, left, right, back
        self.distance = [0, 0, 0]
        self.distance_mapper = distance_mapper(config.uart_ultrasonic["MAPPER"])
        self.connection = serial.Serial(port='/dev/ttyACM0', baudrate=baudrate)

    
    def read_sensor(self):
        try:
            # 4 elements of 2 bytes each
            raw_bytes = self.connection.read(3*2)
            # print("Raw bytes: ", raw_bytes)
            if raw_bytes:
                # self.read_mutex.acquire()
                distance = struct.unpack("<HHH", raw_bytes)
                self.distance[0] = distance[self.distance_mapper.front]
                self.distance[1] = distance[self.distance_mapper.left]
                self.distance[2] = distance[self.distance_mapper.right]
                # self.read_mutex.release()
        except:
            print("uart distance failed to read sensor")
            # logger.debug("uart distance failed to read sensor")


    

