import onnxruntime as ort
import numpy as np
import pyrealsense2 as rs
import cv2

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load ONNX model once (outside the loop for efficiency)
session = ort.InferenceSession(
    "/home/ultraviolet/CV_Detection/yolov8_ros/models/HUST_model.onnx"
)
input_name = session.get_inputs()[0].name

try:
    # Warm up camera
    for _ in range(30):
        pipeline.wait_for_frames()

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert image to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Preprocess: resize and normalize
        img_resized = cv2.resize(color_image, (416, 416))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(img_transposed, axis=0).astype(np.float32)

        # Run inference
        outputs = session.run(None, {input_name: input_data})

        # (Optional) You can process outputs here to draw boxes, etc.

        # Show the frame
        cv2.imshow("RealSense Stream", img_resized)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
