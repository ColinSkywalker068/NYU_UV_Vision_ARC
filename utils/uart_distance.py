import serial
import struct
import config
import os
import distance
import threading
from pathlib import Path
import logging
logging.basicConfig(
    filename=config.general["LOG_PATH"],
    filemode='a',
    level = logging.DEBUG)
logger = logging.getLogger(Path(__file__).stem)
class uart_distance:
    def __init__(self, port="/dev/ttyACM0", baudrate=9600) -> None:
        i = 0
        self.port = port
        self.baudrate = baudrate
        self.read_mutex = threading.Lock()
        # front, left, right, back
        self.distance = [0, 0, 0, 0]
        self.distance_mapper = distance.distance_mapper(config.uart_ultrasonic["MAPPER"])
        while i < 5:
            try:
                self.connection = serial.Serial(port=port, baudrate=baudrate)
            except:
                if i == 0:
                    logger.debug(f"uart distance failed port ACM{i}")
                elif i == 5:
                    logger.debug(f"uart distance failed to connect")
                    raise Exception("uart distance failed to connect")
                i+=1
                port = '/dev/ttyACM' + str(i)
    
    def read_sensor(self):
        try:
            # 4 elements of 2 bytes each
            raw_bytes = self.connection.read(4*2)
            if raw_bytes:
                self.read_mutex.acquire()
                distance = struct.unpack("<HHHH", raw_bytes)
                self.distance[0] = distance[self.distance_mapper.front]
                self.distance[1] = distance[self.distance_mapper.left]
                self.distance[2] = distance[self.distance_mapper.right]
                self.distance[3] = distance[self.distance_mapper.back]
                self.read_mutex.release()
        except:
            logger.debug("uart distance failed to read sensor")


    

