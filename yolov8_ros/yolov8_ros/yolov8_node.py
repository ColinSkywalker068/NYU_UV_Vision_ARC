import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Int32MultiArray, Bool
from cv_bridge import CvBridge
from ultralytics import YOLO, RTDETR, NAS
import cv2
import numpy as np
import os

os.environ["GDK_SYNCHRONIZE"] = "1"

RED_TEAM = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
BLUE_TEAM = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111)

RED_CLASS = 1
BLUE_CLASS = 0

BLUE_RANGES = [
    # Old range: [np.array((90, 50, 70), np.uint8), np.array((128, 255, 255), np.uint8)]
    [np.array((100, 150, 150), np.uint8), np.array((128, 255, 255), np.uint8)]
]
RED_RANGES = [
    # Old range: [np.array((0, 50, 70), np.uint8), np.array((9, 255, 255), np.uint8)]
    # Old range: [np.array((159, 50, 70), np.uint8), np.array((180, 255, 255), np.uint8)]
    [np.array((0, 150, 150), np.uint8), np.array((10, 255, 255), np.uint8)],
    [np.array((170, 150, 150), np.uint8), np.array((180, 255, 255), np.uint8)],
]


BBOX_COLOR = (0, 255, 0)
BBOX_LINE_SIZE = 2

COLOR_THRESHOLD = 200


class YOLOv8Node(Node):
    def __init__(self):
        super().__init__("yolov8_node")
        print("Initializing YOLOv8Node...")

        # Parameter declaration and retrieval with debug statements
        self.declare_parameter(
            "model_path",
            "/home/ultraviolet/CV_Detection/yolov8_ros/yolov8_ros/best.engine",
        )
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        print(f"Model path: {model_path}")

        # Other parameters
        self.declare_parameter("task", "detect")
        self.declare_parameter("type", "yolo")
        self.declare_parameter("half", True)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("show", False)
        self.declare_parameter("conf", 0.5)

        # Retrieve parameter values with debug outputs
        model_type = self.get_parameter("type").get_parameter_value().string_value
        print(f"Model type: {model_type}")
        task = self.get_parameter("task").get_parameter_value().string_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.imgsz = self.get_parameter("imgsz").get_parameter_value().integer_value
        self.show = self.get_parameter("show").get_parameter_value().bool_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.target_class = BLUE_CLASS

        # Setting up model based on type with debug output
        if model_type == "yolo":
            self.model = YOLO(model_path, task=task)
        elif model_type == "rtdetr":
            self.model = RTDETR(model_path)
        elif model_type == "nas":
            self.model = NAS(model_path)
        else:
            raise NotImplementedError("Unsupported model type specified.")
        print(f"Model initialized as {model_type} with task {task}.")

        # Subscription and publishing setup with debug statements
        self.subscription = self.create_subscription(
            Image, "video_frames", self.listener_callback, 10
        )
        print("Image subscriber created.")
        self.bridge = CvBridge()

        self.int_subscription = self.create_subscription(
            Int32, "robot_id", self.id_listener_callback, 10
        )
        print("Robot ID subscriber created.")

        self.bbox_publisher = self.create_publisher(Int32MultiArray, "bounding_box", 10)
        self.override_publisher = self.create_publisher(Bool, "override", 10)
        self.override = Bool()
        self.bbox_msg = Int32MultiArray()
        self.bbox_msg.data = [320, 320, 320, 320]

    def listener_callback(self, msg):
        print("Received image message.")
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        height, width = frame.shape[:2]
        print(f"Frame shape: {frame.shape}")

        # Padding debug output
        required_padding = width - height
        padding = required_padding // 2
        padded_image = cv2.copyMakeBorder(
            frame, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        print("Padded image.")

        # Running model and checking bounding boxes
        results = self.model.track(
            padded_image,
            half=self.half,
            imgsz=self.imgsz,
            show=self.show,
            conf=self.conf,
        )
        self.override.data = False

        for r in results:
            if task = "detect":
                if not hasattr(r, "boxes") or r.boxes is None:
                    continue
                bboxs = r.boxes.xyxy.cpu().int().tolist()
            else:
                if not hasattr(r, "obb") and r.obb is not None:
                    continue
                bboxs = r.obb.xyxy.cpu().int().tolist()

            print(f"Bounding boxes: {bboxs}")

            for i, bbox in enumerate(bboxs):
                roi_bbox = padded_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                ret, col = self.check_enemy(roi_bbox)
                if ret:
                    if self.show:
                        cv2.rectangle(
                            padded_image,
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            BBOX_COLOR,
                            BBOX_LINE_SIZE,
                        )
                    self.bbox_msg.data = bbox
                    self.override.data = True
                    self.override_publisher.publish(self.override)
                    self.bbox_publisher.publish(self.bbox_msg)
            if len(bboxs) == 0:
                self.bbox_msg.data = [320, 320, 320, 320]
                self.override.data = True
                self.override_publisher.publish(self.override)
                self.bbox_publisher.publish(self.bbox_msg)

    def id_listener_callback(self, msg):
        self.target_class = BLUE_CLASS if msg.data in RED_TEAM else RED_CLASS
        print(
            f"Set target class to {'BLUE' if self.target_class == BLUE_CLASS else 'RED'}."
        )

    def check_enemy(self, roi_image):
        if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
            print("Invalid ROI shape; returning False.")
            return False, None

        # Calculate threshold based on ROI size
        color_threshold = roi_image.shape[0] * roi_image.shape[1] * 0.01
        target_class_str = "BLUE" if self.target_class == BLUE_CLASS else "RED"
        ranges = BLUE_RANGES if target_class_str == "BLUE" else RED_RANGES
        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        mask = sum(
            cv2.inRange(hsv_roi, hsv_range[0], hsv_range[1]) for hsv_range in ranges
        )
        color_area = np.count_nonzero(mask)

        color_percentage = (
            color_area / (roi_image.shape[0] * roi_image.shape[1])
        ) * 100
        print(
            f"Color area: {color_area}, Threshold: {color_threshold:.1f}, Percentage: {color_percentage:.1f}%"
        )

        return color_area > color_threshold, color_area


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
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
