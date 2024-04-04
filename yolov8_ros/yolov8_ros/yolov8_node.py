import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from ultralytics import YOLO, RTDETR, NAS
import cv2
import numpy as np

RED_TEAM = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
BLUE_TEAM = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111)

RED_CLASS = [1]
BLUE_CLASS = [0]


class YOLOv8Node(Node):
    def __init__(self):
        super().__init__("yolov8_node")
        self.declare_parameter(
            "model_path", "/home/ultraviolet/Downloads/yolo/best_yolov8_shubham.engine"
        )
        self.declare_parameter("task", "detect")
        self.declare_parameter("type", "yolo")
        self.declare_parameter("half", True)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("show", False)
        self.declare_parameter("conf", 0.5)
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        # print(f'model_name = {model_path}')
        model_type = self.get_parameter("type").get_parameter_value().string_value
        task = self.get_parameter("task").get_parameter_value().string_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.imgsz = self.get_parameter("imgsz").get_parameter_value().integer_value
        self.show = self.get_parameter("show").get_parameter_value().bool_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.target_class = []
        self.subscription = self.create_subscription(
            Image, "video_frames", self.listener_callback, 10
        )
        self.bridge = CvBridge()

        self.int_subscription = self.create_subscription(
            Int32, "robot_id", self.id_listener_callback, 10
        )

        if model_type == "yolo":
            self.model = YOLO(model_path, task=task)
        elif model_type == "rtdetr":
            self.model = RTDETR(model_path)
        elif model_type == "nas":
            self.model = NAS(model_path)
        else:
            raise NotImplementedError

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        height, width = frame.shape[:2]
        required_padding = width - height
        padding = required_padding // 2
        padded_image = cv2.copyMakeBorder(
            frame, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        result = self.model.predict(
            padded_image,
            half=self.half,
            imgsz=self.imgsz,
            show=self.show,
            conf=self.conf,
            classes=self.target_class,
        )
        if result:
            if hasattr(result, "obb"):
                pass
            elif result.boxes and hasattr(result.boxes, "xyxy"):
                for box in result.xyxy:
                    pass

        # Process results here

    def id_listener_callback(self, msg):
        self.target_class = BLUE_CLASS if msg in RED_TEAM else RED_CLASS

    # def normalize_frame(self, frame):
    #     # print(frame)
    #     frame = frame.astype(np.float32) / 255.0
    #     mean_bgr = [0.26162062, 0.26129663, 0.24841645][::-1]
    #     stddev_bgr = [0.25822758, 0.25649119, 0.24888551][::-1]
    #     for i in range(3):
    #         frame[:, :, i] = (frame[:, :, i] - mean_bgr[i]) / stddev_bgr[i]
    #     return frame


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
