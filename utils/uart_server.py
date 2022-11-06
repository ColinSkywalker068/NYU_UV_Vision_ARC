from pyexpat.errors import messages
import threading
import time
import serial
from threading import Thread
from queue import Queue


# UART class to handle serial communication with type A board
class Uart:
    # Constructor
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600, message_queue_size=20):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            timeout=1
        )

        # Queue of recieved messages (message_queue_size max)
        self.recieved_message_queue = Queue(maxsize=message_queue_size)
        self.message_queue_size = message_queue_size

        # Start thread to read data from serial (listener)
        tr = Thread(target=self._read_data_thread)
        tr.start()

    # Send string of data to type A board
    # @param data: string of data to send
    def send_data(self, data: str):
        self.ser.write(data.encode())

    # Get a message from the queue of recieved messages (thread safe)
    def _get_data_queue_thread(self):
        response = ""
        if not self.recieved_message_queue.empty():
            response = self.recieved_message_queue.get()
        return response

    # Read data from type A board into internal queue as a thread
    def _read_data_thread(self):
        while True:
            # handle full queue
            if self.recieved_message_queue.full():
                self.recieved_message_queue.get()

            if self.ser.inWaiting() != 0:
                data = self.ser.readline()
                # print(data)
                self.recieved_message_queue.put_nowait(data)

    # Get a message from the queue of recieved messages
    # @param amount: amount of messages to get from queue
    def get_data(self, amount=1):
        # tmp list to store messages
        messages = []

        # pad amount to length of queue if amount is greater than queue length
        if amount > self.recieved_message_queue.qsize():
            amount = self.recieved_message_queue.qsize()

        # get messages from queue
        messages = [self._get_data_queue_thread() for i in range(amount)]

        return messages

    # Close serial connection

    def close(self):
        self.ser.close()
