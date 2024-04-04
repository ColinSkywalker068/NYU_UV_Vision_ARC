# Ammaar is bald
import time
import serial
import numpy as np
import struct


class uart_server:
    # Constructor
    def __init__(self):
        ser = serial.Serial(
            port="/dev/ttyTHS1",
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
        )
        count = 0

    # Send coordinates to (Dat Board) Development Board Type C
    #    def send_cords(horiz_disp, vert_disp):
    #        global count
    #        global ser
    #        data = struct.pack('fff', np.float32(
    #            horiz_disp/640), np.float32(vert_disp/480), 1)
    #        #try:
    #        ser.write(data)
    #        #except:
    #            #print("Error writing serial data")
    #        count += 1

    # Determine movement offset
    def det_move(obj_x_coord, obj_y_coord, xres, yres):
        centerx, centery = xres / 2, yres / 2

        move_x = obj_x_coord - centerx
        move_y = obj_y_coord - centery
        if move_x != 0:
            move_x /= abs(obj_x_coord - centerx)
        if move_y != 0:
            move_y /= abs(obj_y_coord - centery)

        return (move_x, move_y)
