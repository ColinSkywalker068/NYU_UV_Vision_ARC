import uart_server
import time
import threading
import struct
import numpy as np
import uart_server


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
            self.Uart.listener()
            # delay to prevent cpu overload
            time.sleep(0.1)

    def send_auto_aim(self, msg):
        # msg = [pitch, yaw, has_target]
        length = 0x05
        msg_type = 0x01
        header = struct.pack(">BHBBH", self.header,
                             length, 0x00, 0x00, msg_type)

        body = struct.pack(">ffB", msg[0], msg[1], 1)
        footer = struct.pack(">H", 0x0)

        msg = header+body+footer
        self.Uart.send_data(msg)

    def send_align_request(self, msg):
        #msg = [x,y,r]
        length = 0x06
        msg_type = 0x02

        header = struct.pack(">BHBBH", self.header,
                             length, 0x00, 0x00, msg_type)

        body = struct.pack(">HHH", msg[0], msg[1], msg[2])
        footer = struct.pack(">H", 0x0)
        msg = header+body+footer
        self.Uart.send_data(msg)

    def send_align_finish(self, msg):
        if not msg[0]:
            return

        length = 0x01
        msg_type = 0x03
        header = struct.pack(">BHBBH", self.header,
                             length, 0x00, 0x00, msg_type)
        body = struct.pack(">B", 0x01)
        footer = struct.pack(">H", 0x0)
        msg = header+body+footer

        self.Uart.send_data(msg)


if __name__ == "__main__":
    node = Publisher_node(None)
    node.send_auto_aim([1, 2, 1])
    node.send_align_request([1, 2, 3])
    node.send_align_finish([1])
