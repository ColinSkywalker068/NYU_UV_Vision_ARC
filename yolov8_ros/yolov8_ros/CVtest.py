# Import libraries
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

# Enable OpenCV optimizations
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

# Load YOLO model
model = YOLO("/home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb2.pt")
# bestObb2.pt runs well, 120 inch is good 80-100 performance, 150 inch is about mid performance with 50-80 general conf, and 180 is the max for any ok results.
# Initialize webcam
cap = cv2.VideoCapture(0)


# Define HSV color ranges for blue and red
BLUE_RANGES = [
    (np.array([100, 150, 150], np.uint8), np.array([128, 255, 255], np.uint8))
]
RED_RANGES = [
    (np.array([0, 150, 150], np.uint8), np.array([10, 255, 255], np.uint8)),
    (np.array([170, 150, 150], np.uint8), np.array([180, 255, 255], np.uint8))
]

def reduce_white_regions(hsv):
    """Reduce brightness in white regions to enhance color visibility."""
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    hsv[:, :, 2][white_mask > 0] = (hsv[:, :, 2][white_mask > 0] * 0.7).astype(np.uint8)
    return white_mask

def create_color_mask(hsv, color_ranges):
    """Create a mask for specified color ranges."""
    mask = np.zeros_like(hsv[:, :, 0])
    for low, high in color_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    return mask

def enhance_image(frame):
    """Enhance image by reducing white regions and isolating red/blue colors."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = reduce_white_regions(hsv)

    red_mask = create_color_mask(hsv, RED_RANGES)
    blue_mask = create_color_mask(hsv, BLUE_RANGES)
    color_mask = cv2.bitwise_or(red_mask, blue_mask)

    hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(hsv[:, :, 2])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    color_regions = cv2.bitwise_and(enhanced, enhanced, mask=color_mask)
    result = cv2.addWeighted(enhanced, 0.7, color_regions, 0.3, 0)
    result = cv2.bilateralFilter(result, 9, 75, 75)

    color_vis = np.zeros_like(frame)
    color_vis[blue_mask > 0] = [255, 0, 0]
    color_vis[red_mask > 0] = [0, 0, 255]

    white_vis = np.zeros_like(frame)
    white_vis[white_mask > 0] = [255, 255, 255]

    return result, color_vis, white_vis

# Import libraries
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

# Enable OpenCV optimizations
cv2.setUseOptimized(True)
cv2.setNumThreads(2)

# Load YOLO model
model = YOLO("/home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb2.pt")
# bestObb2.pt runs well, 120 inch is good 80-100 performance, 150 inch is about mid performance with 50-80 general conf, and 180 is the max for any ok results.
# Initialize webcam
cap = cv2.VideoCapture(0)


# Define HSV color ranges for blue and red
BLUE_RANGES = [
    (np.array([100, 150, 150], np.uint8), np.array([128, 255, 255], np.uint8))
]
RED_RANGES = [
    (np.array([0, 150, 150], np.uint8), np.array([10, 255, 255], np.uint8)),
    (np.array([170, 150, 150], np.uint8), np.array([180, 255, 255], np.uint8))
]

def reduce_white_regions(hsv):
    """Reduce brightness in white regions to enhance color visibility."""
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    hsv[:, :, 2][white_mask > 0] = (hsv[:, :, 2][white_mask > 0] * 0.7).astype(np.uint8)
    return white_mask

def create_color_mask(hsv, color_ranges):
    """Create a mask for specified color ranges."""
    mask = np.zeros_like(hsv[:, :, 0])
    for low, high in color_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
    return mask

def enhance_image(frame):
    """Enhance image by reducing white regions and isolating red/blue colors."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = reduce_white_regions(hsv)

    red_mask = create_color_mask(hsv, RED_RANGES)
    blue_mask = create_color_mask(hsv, BLUE_RANGES)
    color_mask = cv2.bitwise_or(red_mask, blue_mask)

    hsv[:, :, 2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(hsv[:, :, 2])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    color_regions = cv2.bitwise_and(enhanced, enhanced, mask=color_mask)
    result = cv2.addWeighted(enhanced, 0.7, color_regions, 0.3, 0)
    result = cv2.bilateralFilter(result, 9, 75, 75)

    color_vis = np.zeros_like(frame)
    color_vis[blue_mask > 0] = [255, 0, 0]
    color_vis[red_mask > 0] = [0, 0, 255]

    white_vis = np.zeros_like(frame)
    white_vis[white_mask > 0] = [255, 255, 255]

    return result, color_vis, white_vis

def tile_inference(image, model, tile_size=160, overlap=0.2):
    stride = int(tile_size * (1 - overlap))
    H, W, _ = image.shape
    detections = []

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            tile = image[y:y+tile_size, x:x+tile_size]
            h, w, _ = tile.shape
            if h < tile_size or w < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded_tile[:h, :w] = tile
                tile = padded_tile


            result = model(tile)[0]
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                conf = box[4]
                cls = int(box[5])
                detections.append([x1 + x, y1 + y, x2 + x, y2 + y, conf, cls])

    return detections

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# # try:
# #     while True:
# #         # Wait for a coherent color frame
# #         frames = pipeline.wait_for_frames()
# #         color_frame = frames.get_color_frame()
# #         if not color_frame:
# #             print("Failed to grab frame. Exiting...")
# #             break
# #         frame = np.asanyarray(color_frame.get_data())  # Convert to numpy array

# #         enhanced_frame, color_visualization, white_visualization = enhance_image(frame)
# #         results = model(enhanced_frame)
# #         output_frame = results[0].plot()

# #         cv2.imshow("Original Frame", frame)
# #         cv2.imshow("Detection Results", output_frame)

# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break

# # except KeyboardInterrupt:
# #     print("Interrupted by user. Exiting...")

# # finally:
# #     pipeline.stop()  # Stop the RealSense pipeline
# #     cv2.destroyAllWindows()
# #     cap.release()
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)

try:
    while True:
        # Wait for a coherent color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("Failed to grab frame. Exiting...")
            break
        frame = np.asanyarray(color_frame.get_data())  # Convert to numpy array

        enhanced_frame, color_visualization, white_visualization = enhance_image(frame)
        detections = tile_inference(enhanced_frame, model, tile_size=640, overlap=0.5)


        output_frame = enhanced_frame.copy()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(output_frame, f"{model.names[cls]} {conf:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Detection Results", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user. Exiting...")

finally:
    pipeline.stop()  # Stop the RealSense pipeline
    cv2.destroyAllWindows()
    cap.release()