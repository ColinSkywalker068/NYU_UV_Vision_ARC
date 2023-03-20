import statistics
import cv2
from Realsense.realsense_depth import *

# Processes frame and returns median depth

# Get Median Depth from given Depth Frame and BBOX coordinates


def process_frame(depth_frame, x_min, y_min, x_max, y_max):
    values = []
    for x in range(x_min - 1, x_max):
        for y in range(y_min - 1, y_max):
            values.append(depth_frame[y, x])

    med = statistics.median(values)
    return med


# Display BBOX around detection with Estimated median depth.
def show_frame(color_frame, depth_frame, depth, coordinates):
    # Display Text for distance
    if coordinates != None:
        cv2.putText(color_frame, "Median: {}mm".format(
            depth), (coordinates[0], coordinates[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))

        # Display Rectangle overlay
        cv2.rectangle(color_frame, (coordinates[0], coordinates[1]),
                      (coordinates[2], coordinates[3]), (0, 0, 255), 10)
        cv2.rectangle(depth_frame, (coordinates[0], coordinates[1]),
                      (coordinates[2], coordinates[3]), (0, 0, 255), 10)


    # Show Both
    cv2.imshow("Video", color_frame)
    cv2.imshow("Video_Depth", depth_frame)


#point = (400, 300)

# def show_distance(event, x, y, args, params):
#    global point
#    point = (x, y)

# Initialize Camera Intel Realsense
#dc = DepthCamera()

# Create mouse event
#cv2.namedWindow("Color frame")


# while True:
#    ret, depth_frame, color_frame = dc.get_frame()
#
#    # Show distance for a specific point
#    cv2.circle(color_frame, point, 4, (0, 0, 255))
#    distance = depth_frame[point[1], point[0]]
#
#    # Calculate bounds of 100 x 100 square around the point
#    bounds = 50
#    minY = point[1]-bounds
#    maxY = point[1]+bounds
#    minX = point[0]-bounds
#    maxX = point[0]+bounds
#
#    values = []
#    for x in range(minX, maxX+1):
#        for y in range(minY, minY+1):
#            values.append(depth_frame[y, x])
#            #print(depth_frame[point[1], point[0]])
#    print(values)
#    med = statistics.median(values)
#
#    cv2.putText(color_frame, "{}mm".format(distance) + "Median: {}mm".format(med), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0))
#    cv2.rectangle(color_frame, (minX, minY), (maxX, maxY), (0, 0, 255), 10)
#
#    cv2.imshow("depth frame", depth_frame)
#    cv2.imshow("Color frame", color_frame)
#    key = cv2.waitKey(1)
#    if key == 27:
#        break
