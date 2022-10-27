from pyexpat.errors import messages
import threading
import time
import serial
from threading import Thread


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
        self.recieved_message_queue = []
        self.message_queue_size = message_queue_size

        # mutex for recieved message queue
        self.message_queue_mutex = threading.Lock()

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
        if len(self.recieved_message_queue) > 0:
            self.message_queue_mutex.acquire()
            response = self.recieved_message_queue.pop(0)
            self.message_queue_mutex.release()
        return response

    # Read data from type A board into internal queue as a thread
    def _read_data_thread(self):
        while True:
            # handle full queue
            if len(self.recieved_message_queue) > self.message_queue_size:
                _ = self._get_data_queue_thread()

            if self.ser.inWaiting() != 0:
                data = self.ser.readline()
                # print(data)
                self.message_queue_mutex.acquire()
                self.recieved_message_queue.append(data)
                self.message_queue_mutex.release()

    # Get a message from the queue of recieved messages
    # @param amount: amount of messages to get from queue
    def get_data(self, amount=1):
        # tmp list to store messages
        messages = []

        # pad amount to length of queue if amount is greater than queue length
        if amount > len(self.recieved_message_queue):
            amount = len(self.recieved_message_queue)

        # get messages from queue
        for i in range(amount):
            messages.append(self._get_data_queue_thread())

        return messages

    # Close serial connection

    def close(self):
        self.ser.close()
