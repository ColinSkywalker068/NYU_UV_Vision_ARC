import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Int32MultiArray, Bool
# --- ONNX/CV IMPORTS ---
import cv2
import numpy as np
import onnxruntime  # Import ONNX Runtime
# --- END ONNX/CV IMPORTS ---
import os
import pyrealsense2 as rs
import threading
import time

# Ensure the display is available for cv2.imshow if used
os.environ["GDK_SYNCHRONIZE"] = "1"

# --- Constants and Configuration (Unchanged) ---
RED_TEAM = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
BLUE_TEAM = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111)
RED_CLASS = 1
BLUE_CLASS = 0
BLUE_RANGES = [
    [np.array((100, 150, 150), np.uint8), np.array((128, 255, 255), np.uint8)]
]
RED_RANGES = [
    [np.array((0, 150, 150), np.uint8), np.array((10, 255, 255), np.uint8)],
    [np.array((170, 150, 150), np.uint8), np.array((180, 255, 255), np.uint8)],
]
BBOX_COLOR = (0, 255, 0)
BBOX_LINE_SIZE = 2
IMG_WIDTH = 640
IMG_HEIGHT = 480
FPS = 60

class yolov8_ONNX(Node):
    def __init__(self):
        super().__init__("yolov8_onnx_realsense_node")
        self.get_logger().info("Initializing YOLOv8 ONNX RealSense Node...")

        # --- ROS Parameter Declaration ---
        # Update default to point to an .onnx file
        self.declare_parameter("model_path", "/path/to/your/model.onnx")
        self.declare_parameter("half", True)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("show_video", False)
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("iou_threshold", 0.7) # Add IOU threshold for NMS
        self.declare_parameter("camera_exposure", 100)

        # --- Get ROS Parameters ---
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.is_half = self.get_parameter("half").get_parameter_value().bool_value
        self.imgsz = self.get_parameter("imgsz").get_parameter_value().integer_value
        self.show = self.get_parameter("show_video").get_parameter_value().bool_value
        self.conf_threshold = self.get_parameter("conf").get_parameter_value().double_value
        self.iou_threshold = self.get_parameter("iou_threshold").get_parameter_value().double_value
        self.target_class = BLUE_CLASS
        self.camera_exposure = self.get_parameter("camera_exposure").get_parameter_value().integer_value

        # --- ONNX Model Initialization ---
        self.get_logger().info(f"Loading ONNX model: {model_path}")
        try:
            # Use CUDAExecutionProvider for GPU acceleration
            self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
            # Get model input details
            model_inputs = self.session.get_inputs()
            self.input_name = model_inputs[0].name
            self.input_shape = model_inputs[0].shape
            self.get_logger().info(f"ONNX model loaded. Input Name: {self.input_name}, Input Shape: {self.input_shape}")
        except Exception as e:
            self.get_logger().error(f"Failed to load ONNX model: {e}")
            self.get_logger().info("Attempting to load with CPU provider.")
            try:
                # Fallback to CPU if CUDA fails
                self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                model_inputs = self.session.get_inputs()
                self.input_name = model_inputs[0].name
                self.input_shape = model_inputs[0].shape
                self.get_logger().info(f"ONNX model loaded on CPU. Input Name: {self.input_name}, Input Shape: {self.input_shape}")
            except Exception as e_cpu:
                 self.get_logger().error(f"Failed to load ONNX model on CPU as well: {e_cpu}")
                 raise

        # --- ROS Publishers and Subscribers (Unchanged) ---
        self.int_subscription = self.create_subscription(Int32, "robot_id", self.id_listener_callback, 10)
        self.bbox_publisher = self.create_publisher(Int32MultiArray, "bounding_box", 10)
        self.override_publisher = self.create_publisher(Bool, "override", 10)
        self.override = Bool()
        self.bbox_msg = Int32MultiArray()
        self.bbox_msg.data = [320, 320, 320, 320]

        # --- Threading and RealSense Setup (Largely Unchanged) ---
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = True
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, FPS)
        
        try:
            profile = self.pipeline.start(config)
            self.get_logger().info(f"RealSense camera started at {IMG_WIDTH}x{IMG_HEIGHT} @ {FPS} FPS.")
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                self.get_logger().info("Auto-exposure disabled.")
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, self.camera_exposure)
                self.get_logger().info(f"Manual exposure set to: {self.camera_exposure} us.")
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense camera: {e}")
            self.is_running = False
            return
        
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        self.get_logger().info("Camera polling thread started.")
        
        self.timer_period = 1.0 / (FPS * 2) 
        self.timer = self.create_timer(self.timer_period, self.process_frame_callback)
        self.get_logger().info("YOLOv8 ONNX Node has been initialized and is running.")

    def _camera_loop(self):
        """This function runs in a separate thread to continuously poll the camera."""
        while self.is_running:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame_data = np.asanyarray(color_frame.get_data())
                    with self.frame_lock:
                        self.latest_frame = frame_data
            except Exception as e:
                self.get_logger().error(f"Error in camera loop: {e}", throttle_duration_sec=2)
                time.sleep(1)

    def _preprocess(self, img):
        """Prepares the image for ONNX model inference."""
        # The model expects a square image, so we use the same padding logic
        height, width = img.shape[:2]
        required_padding = max(0, width - height)
        self.padding_amount = required_padding // 2
        padded_image = cv2.copyMakeBorder(img, self.padding_amount, self.padding_amount, 0, 0, cv2.BORDER_CONSTANT, value=[114, 114, 114])

        # Resize to the model's expected input size (e.g., 640x640)
        input_img = cv2.resize(padded_image, (self.imgsz, self.imgsz))

        # BGR to RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        
        # HWC to CHW (Height, Width, Channels to Channels, Height, Width)
        input_img = input_img.transpose(2, 0, 1)
        
        # Normalize to [0, 1]
        input_img = input_img / 255.0
        
        # Add batch dimension and convert to float32 or float16
        input_tensor = np.expand_dims(input_img, axis=0).astype(np.float16 if self.is_half else np.float32)
        
        return input_tensor, padded_image
        
    def _postprocess(self, outputs, original_padded_image):
        """Processes raw ONNX output to get bounding boxes, scores, and class IDs."""
        # The output of YOLOv8 is [batch, 84, 8400] where 84=4(xywh)+80(classes)
        # We need to transpose it to [batch, 8400, 84]
        predictions = np.squeeze(outputs[0]).T

        boxes = []
        scores = []
        class_ids = []

        # Calculate scaling factors to map model output back to the padded image size
        y_scale = original_padded_image.shape[0] / self.imgsz
        x_scale = original_padded_image.shape[1] / self.imgsz

        for pred in predictions:
            # Get the class with the highest score
            class_score = np.max(pred[4:])
            class_id = np.argmax(pred[4:])

            if class_score > self.conf_threshold:
                # Extract box coordinates (center_x, center_y, width, height)
                cx, cy, w, h = pred[:4]
                
                # Scale coordinates back to the original padded image size
                box_x = int((cx - w/2) * x_scale)
                box_y = int((cy - h/2) * y_scale)
                box_w = int(w * x_scale)
                box_h = int(h * y_scale)
                
                boxes.append([box_x, box_y, box_w, box_h])
                scores.append(class_score)
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression to clean up overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                final_boxes.append([x, y, x + w, y + h])

        return final_boxes


    def process_frame_callback(self):
        """Main loop to process frames and publish results."""
        with self.frame_lock:
            if self.latest_frame is None:
                self.get_logger().warn("No frame available from camera thread yet.", throttle_duration_sec=5)
                return
            frame = self.latest_frame.copy()

        # --- Pre-process the image for ONNX ---
        input_tensor, padded_image = self._preprocess(frame)

        # --- Run Inference ---
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # --- Post-process the output ---
        bboxes = self._postprocess(outputs, padded_image)

        valid_detections = []
        for bbox in bboxes:
            # Adjust bbox coordinates to be relative to the padded_image
            roi_bbox = padded_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            is_enemy, _ = self.check_enemy(roi_bbox)
            if is_enemy:
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                valid_detections.append({'box': bbox, 'area': area})

        if valid_detections:
            biggest_detection = max(valid_detections, key=lambda d: d['area'])
            biggest_bbox = biggest_detection['box']
            self.bbox_msg.data = biggest_bbox
            self.override.data = True
            if self.show:
                cv2.rectangle(padded_image, (biggest_bbox[0], biggest_bbox[1]), (biggest_bbox[2], biggest_bbox[3]), BBOX_COLOR, BBOX_LINE_SIZE)
        else:
            self.bbox_msg.data = [320, 320, 320, 320]
            self.override.data = False

        self.override_publisher.publish(self.override)
        self.bbox_publisher.publish(self.bbox_msg)
        
        if self.show:
            cv2.imshow("YOLOv8 ONNX RealSense Detection", padded_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("'q' pressed, shutting down.")
                self.context.try_shutdown()

    # --- Unchanged Helper Functions ---
    def id_listener_callback(self, msg):
        self.target_class = BLUE_CLASS if msg.data in RED_TEAM else RED_CLASS
        self.get_logger().info(f"Set target class to {'BLUE' if self.target_class == BLUE_CLASS else 'RED'}.")

    def check_enemy(self, roi_image):
        if roi_image.size == 0: return False, 0
        color_threshold = roi_image.shape[0] * roi_image.shape[1] * 0.01
        target_ranges = BLUE_RANGES if self.target_class == BLUE_CLASS else RED_RANGES
        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, target_ranges[0][0], target_ranges[0][1])
        if len(target_ranges) > 1:
            for low, high in target_ranges[1:]:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv_roi, low, high))
        color_area = np.count_nonzero(mask)
        return color_area > color_threshold, color_area

    def destroy_node(self):
        self.get_logger().info("Shutting down...")
        self.is_running = False
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join(timeout=1)
        if hasattr(self, 'pipeline'):
            self.get_logger().info("Stopping RealSense pipeline.")
            self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = yolov8_ONNX()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()