from pyexpat.errors import messages
import threading
import time
import serial
from threading import Thread
from queue import Queue
import struct
import math


# UART constants
HEADER_LENGTH = 5
WAITING_FOR_HEADER = 0
READING_HEADER = 1
READING_DATA = 2


# UART class to handle serial communication with type A board
class Uart:
    # Constructor
    def __init__(self, port="/dev/ttyUSB0", baudrate=115200):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0
        )

        # use USB serial port

        self.HEADER = 0xE7

        self.state = WAITING_FOR_HEADER
        self.msg_len = 0
        self.msg_type = 0

        self.auto_aim_request_list = Queue.queue()

        # mutex for robot id

        self.robot_id = -1
        self.robot_id_mutex = threading.Lock()

        self.auto_aim_request = False
        self.auto_aim_request_mutex = threading.Lock()

        # IMU data
        # ax, ay, az, roll, pitch, yaw, wx, wy, wz, rf_rpm, lf_rpm, rb_rpm, lb_rpm
        self.imu_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.imu_data_mutex = threading.Lock()

        # Gimbal Feedback
        # pitch, yaw
        self.turret_feedback = [0, 0]
        self.turret_feedback_mutex = threading.Lock()

    # Send string of data to type A board
    # @param data: byte string of data to send

    def send_data(self, data):
        self.ser.write(data)

    ### BEGIN MESSAGE HANDLERS ###

    def handle_align_request(self, data):
        # data = [enable]
        try:
            data = struct.unpack(">B", data)
        except:
            print("Invalid Align Request Message")
            return

        request = []
        request[0] = data[0]
        self.auto_aim_request_list.put(request)

    def handle_robot_id(self, data):
        # data = [id]
        try:
            data = struct.unpack(">B", data)
        except:
            print("Invalid Robot ID Message")
            return

        self.robot_id_mutex.acquire()
        self.robot_id = data[0]
        self.robot_id_mutex.release()

    def handle_auto_aim_request(self, data):
        # data = [enable]
        try:
            data = struct.unpack(">B", data)
        except:
            print("Invalid Auto Aim Request Message")
            return

        self.auto_aim_request_mutex.acquire()
        self.auto_aim_request = data[0]
        self.auto_aim_request_mutex.release()

    def handle_imu_feedback(self, data):
        try:
            data = struct.unpack(">hhhhhhhhhhhhh", data)
        except:
            print("Invalid IMU Feedback Message")
            return

        self.imu_data_mutex.acquire()
        self.imu_data[0] = float(data[0])/100
        self.imu_data[1] = float(data[1])/100
        self.imu_data[2] = float(data[2])/100
        self.imu_data[3] = float(data[3]/100)
        self.imu_data[4] = float(data[4]/100)
        self.imu_data[5] = float(data[5]/100)
        self.imu_data[6] = float(data[6])/100
        self.imu_data[7] = float(data[7])/100
        self.imu_data[8] = float(data[8])/100
        self.imu_data[9] = data[9]
        self.imu_data[10] = data[10]
        self.imu_data[11] = data[11]
        self.imu_data[12] = data[12]
        self.imu_data_mutex.release()

    def handle_turret_feedback(self, data):
        # data = [pitch, yaw]
        try:
            data = struct.unpack(">hh", data)
        except:
            print("Invalid Turret Feedback Message")
            return
        feedback = [float(data[0])/100, float(data[1])/100]
        self.turret_feedback.lock()
        self.turret_feedback = feedback
        self.turret_feedback_mutex.release()

    ### END MESSAGE HANDLERS ###
    # listen for data from serial port

    def listener(self):
        if self.ser.in_waiting > 0:
            if self.state == WAITING_FOR_HEADER:
                data = struct.unpack("B", self.ser.read())[0]
                if data == self.HEADER:
                    self.state = READING_HEADER
            elif self.state == READING_HEADER:
                header_bytes = self.ser.read(HEADER_LENGTH-1)
                if len(header_bytes) != HEADER_LENGTH-1:
                    self.state = WAITING_FOR_HEADER
                    return

                header = struct.unpack(">BHBB", header_bytes)
                self.msg_len = header[0]
                self.msg_type = header[-1]
                self.state = READING_DATA

            elif self.state == READING_DATA:
                self.reset_listener()
                self.data = self.ser.read(self.msg_len)
                if self.msg_type == 1:
                    self.handle_align_request(self.data)
                elif self.msg_type == 2:
                    self.handle_robot_id(self.data)
                elif self.msg_type == 3:
                    self.handle_auto_aim_request(self.data)
                elif self.msg_type == 4:
                    self.handle_imu_feedback(self.data)
                elif self.msg_type == 5:
                    self.handle_turret_feedback(self.data)

    def reset_listener(self):
        self.state = WAITING_FOR_HEADER
        self.msg_len = 0
        self.msg_type = 0

    # get functions

    ### BEGIN GET FUNCTIONS ###
    def get_robot_id(self):
        self.robot_id_mutex.acquire()
        robot_id = self.robot_id
        self.robot_id_mutex.release()
        return robot_id

    def get_auto_aim_request(self):
        self.auto_aim_request_mutex.acquire()
        auto_aim_request = self.auto_aim_request
        self.auto_aim_request_mutex.release()
        return auto_aim_request

    def get_imu_data(self):
        self.imu_data_mutex.acquire()
        imu_data = self.imu_data
        self.imu_data_mutex.release()
        return imu_data

    def get_turret_feedback(self):
        self.turret_feedback_mutex.acquire()
        turret_feedback = self.turret_feedback
        self.turret_feedback_mutex.release()
        return turret_feedback

    ### END GET FUNCTIONS ###
    def __del__(self):
        self.ser.close()
