import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')
        self.declare_parameter('model_path', '/home/ultraviolet/Downloads/yolo/best_yolov8_shubham.engine')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.model = YOLO(model_path, task='detect')

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model.predict(frame, half=True, imgsz=[480,640], show=True)
        # Process results here

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

if __name__ == '__main__':
    main()
