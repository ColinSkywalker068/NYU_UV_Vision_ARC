import os
import serial
import numpy as np
from turtle import color
import matplotlib
from Realsense.realsense_depth import *
from Realsense.realsense import *
from Algorithm.main import *
import cv2
import time
import argparse
import struct
from UART.uart import uart_server
opponent_team = 'r'  # 'r' or 'b'
count = 0

matplotlib.use('TKAgg')
# Disable tensorflow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


red_lower = np.array([0, 4, 226])
red_upper = np.array([60, 255, 255])
blue_lower = np.array([68, 38, 131])
blue_upper = np.array([113, 255, 255])


def red_or_blue(color_frame):
    hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    _, red_contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, blue_contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(red_contours) > 0 and len(blue_contours) > 0:
        r_area = 0
        b_area = 0
        for c in red_contours:
            r_area += cv2.contourArea(c)
        for c in blue_contours:
            b_area += cv2.contourArea(c)
        if r_area > b_area:
            return 'r'
        else:
            return 'b'
    elif len(red_contours) > 0:
        return 'r'

    return 'b'


def send_cords(ser, horiz_disp, vert_disp):
    global count
    data = struct.pack('fff', np.float32(
        horiz_disp/640), np.float32(vert_disp/480), 1)
    try:
        ser.write(data)
    except:
        print("Error writing serial data")
    count += 1


def det_move_(obj_x_coord, obj_y_coord, xres, yres):
    obj_y_coord = yres-obj_y_coord
    centerx, centery = xres/2.0, yres/2.0

    move_x = obj_x_coord-centerx
    move_y = obj_y_coord-centery
    if(move_x != 0):
        move_x /= centerx
    if(move_y != 0):
        move_y /= centery
    return(move_x, move_y)


def main(_argv):
    parser = argparse.ArgumentParser()
    # Initialize Camera Intel Realsense
    dc = DepthCamera()
    uartServer = uart_server()

    Debug_flag = 0

    # Parse arguments
    # if _argv.Debug == "1" or _argv.D == "1":
    #    Debug_flag = 1
    #    # Create window for video
    #    cv2.namedWindow("Video")
    #    cv2.namedWindow("Video_Depth")

    # elif _argv.Debug == "0" or _argv.D == "0":
    #   Debug_flag = 0

    # Load saved CV model
    model = get_model()

    ser = serial.Serial(
        port="/dev/ttyTHS1",
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1
    )

    # Initialize Algorithm
    oldCords = None
    depth = None

    while True:
        # Start Video Capture
        try:
            ret, depth_frame, color_frame = dc.get_frame()
        except:
            print("Error getting frame")

        # If frame is not empty
        if ret:

            key = cv2.waitKey(1)
            if key == 27:
                break

            # Get coordinates from color frame
            try:
                coordinates = get_coordinates(color_frame, model)
            except:
                print("Error getting cordinates\n")

            if coordinates != None:
                # Get Median Depth from depth frame
                try:
                    depth = process_frame(
                        depth_frame, coordinates[0], coordinates[1], coordinates[2], coordinates[3])
                except:
                    print("Error processing_frame")

                # Debug mode
                if Debug_flag == 1:
                    print(coordinates)
                    print(coordinates)
                    print(depth)
                    show_frame(color_frame, depth_frame, depth, coordinates)

               # try:
                final_cords = det_move_(
                    (coordinates[0]+coordinates[2])/2,
                    (coordinates[1]+coordinates[3])/2,
                    640,
                    480)

                if red_or_blue(color_frame) != opponent_team:
                    pass
                print(final_cords[0], final_cords[1])
                send_cords(ser, final_cords[0], final_cords[1])
               # except:
                #   print("Error sending coordinates", coordinates[0], coordinates[1], coordinates[2], coordinates[3],  det_move_(
                #           (coordinates[0]+coordinates[2])/2,
                #           (coordinates[1]+coordinates[3])/2,
                #           640,
                #           480))


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
