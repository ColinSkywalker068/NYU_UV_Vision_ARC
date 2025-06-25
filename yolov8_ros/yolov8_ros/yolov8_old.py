import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Int32MultiArray, Bool
from ultralytics import YOLO
import cv2
import numpy as np
import os
import pyrealsense2 as rs

# Ensure the display is available for cv2.imshow if used
os.environ["GDK_SYNCHRONIZE"] = "1"

# --- Constants and Configuration ---
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


class YOLOv8Node(Node):
    def __init__(self):
        super().__init__("yolov8_realsense_node")
        self.get_logger().info("Initializing YOLOv8 RealSense Node...")

        # --- ROS Parameter Declaration ---
        self.declare_parameter("model_path", "/path/to/your/model.engine")
        self.declare_parameter("task", "detect")  # Can be 'detect' or 'obb'
        self.declare_parameter("half", True)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("show_video", False)
        self.declare_parameter("conf", 0.5)

        # --- Get ROS Parameters ---
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.task = self.get_parameter("task").get_parameter_value().string_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.imgsz = self.get_parameter("imgsz").get_parameter_value().integer_value
        self.show = self.get_parameter("show_video").get_parameter_value().bool_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.target_class = BLUE_CLASS  # Default target

        # --- Model Initialization ---
        self.get_logger().info(f"Loading model: {model_path}")
        try:
            self.model = YOLO(model_path, task=self.task)
            self.get_logger().info(f"Model loaded successfully for task '{self.task}'.")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise

        # --- ROS Publishers and Subscribers ---
        self.int_subscription = self.create_subscription(Int32, "robot_id", self.id_listener_callback, 10)
        self.bbox_publisher = self.create_publisher(Int32MultiArray, "bounding_box", 10)
        self.override_publisher = self.create_publisher(Bool, "override", 10)
        
        self.override = Bool()
        self.bbox_msg = Int32MultiArray()
        self.bbox_msg.data = [320, 320, 320, 320] # Default center

        # --- RealSense Camera Initialization ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, FPS)
        
        try:
            self.pipeline.start(config)
            self.get_logger().info(f"RealSense camera started at {IMG_WIDTH}x{IMG_HEIGHT} @ {FPS} FPS.")
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense camera: {e}")
            return

        # --- ROS Timer for Main Loop ---
        self.timer_period = 1.0 / FPS
        self.timer = self.create_timer(self.timer_period, self.process_frame_callback)
        self.last_frame_time = None
        self.get_logger().info("YOLOv8 RealSense Node has been initialized and is running.")


    def process_frame_callback(self):
        """
        Main loop to grab frames from RealSense, process them, and publish results.
        """
        # --- FPS Calculation ---
        current_time = self.get_clock().now()
        if self.last_frame_time is not None:
            time_diff = current_time - self.last_frame_time
            time_diff_s = time_diff.nanoseconds / 1e9
            if time_diff_s > 0:
                fps = 1.0 / time_diff_s
                self.get_logger().info(f"Actual FPS: {fps:.2f}")
        self.last_frame_time = current_time

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            self.get_logger().warn("No color frame received from RealSense camera.")
            return

        frame = np.asanyarray(color_frame.get_data())
        height, width = frame.shape[:2]

        required_padding = max(0, width - height)
        padding = required_padding // 2
        padded_image = cv2.copyMakeBorder(
            frame, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        # --- Model Inference (FIXED) ---
        # Use the model directly as a callable, which works for all tasks
        results = self.model(
            padded_image,
            half=self.half,
            imgsz=self.imgsz,
            show=self.show,
            conf=self.conf,
            verbose=False
        )

        valid_detections = []
        for r in results:
            # --- Bounding Box Extraction (Handles OBB and Detect) ---
            if self.task == 'obb':
                if not hasattr(r, "obb") or r.obb is None: continue
                # .xyxy provides the upright bounding box for the oriented box
                bboxes = r.obb.xyxy.cpu().int().tolist()
            else: # Default to standard detection
                if not hasattr(r, "boxes") or r.boxes is None: continue
                bboxes = r.boxes.xyxy.cpu().int().tolist()
            
            for bbox in bboxes:
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
                cv2.rectangle(
                    padded_image,
                    (biggest_bbox[0], biggest_bbox[1]),
                    (biggest_bbox[2], biggest_bbox[3]),
                    BBOX_COLOR,
                    BBOX_LINE_SIZE,
                )
        else:
            self.bbox_msg.data = [320, 320, 320, 320]
            self.override.data = False

        self.override_publisher.publish(self.override)
        self.bbox_publisher.publish(self.bbox_msg)

        if self.show:
            cv2.imshow("YOLOv8 RealSense Detection", padded_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("'q' pressed, shutting down.")
                # This will trigger the main loop to exit and call destroy_node
                self.context.try_shutdown()

    def id_listener_callback(self, msg):
        self.target_class = BLUE_CLASS if msg.data in RED_TEAM else RED_CLASS
        self.get_logger().info(
            f"Set target class to {'BLUE' if self.target_class == BLUE_CLASS else 'RED'}."
        )

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
        self.get_logger().info("Stopping RealSense pipeline.")
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()