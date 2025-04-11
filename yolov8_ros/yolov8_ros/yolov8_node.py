from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

# Replace print statements with ROS logging for better integration and debugging
class YOLOv8Node(Node):
    def __init__(self):
        super().__init__("yolov8_node")
        self.get_logger().info("Initializing YOLOv8Node...")

        # Parameter declaration and retrieval with debug statements
        self.declare_parameter(
            "model_path",
            "/home/ultraviolet/CV_Detection/yolov8_ros/yolov8_ros/best.engine",
        )
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.get_logger().info(f"Model path: {model_path}")

        # Other parameters
        self.declare_parameter("task", "detect")
        self.declare_parameter("type", "yolo")
        self.declare_parameter("half", True)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("show", False)
        self.declare_parameter("conf", 0.5)

        # Retrieve parameter values with debug outputs
        model_type = self.get_parameter("type").get_parameter_value().string_value
        self.get_logger().info(f"Model type: {model_type}")
        self.task = self.get_parameter("task").get_parameter_value().string_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.imgsz = self.get_parameter("imgsz").get_parameter_value().integer_value
        self.show = self.get_parameter("show").get_parameter_value().bool_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.target_class = BLUE_CLASS

        # Setting up model based on type with debug output
        try:
            if model_type == "yolo":
                self.model = YOLO(model_path, task=self.task)
            elif model_type == "rtdetr":
                self.model = RTDETR(model_path)
            elif model_type == "nas":
                self.model = NAS(model_path)
            else:
                raise NotImplementedError("Unsupported model type specified.")
            self.get_logger().info(
                f"Model initialized as {model_type} with task {self.task}."
            )
        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")

        # Subscription and publishing setup with debug statements
        self.subscription = self.create_subscription(
            Image, "video_frames", self.listener_callback, 10
        )
        self.get_logger().info("Image subscriber created.")
        self.bridge = CvBridge()

        self.int_subscription = self.create_subscription(
            Int32, "robot_id", self.id_listener_callback, 10
        )
        self.get_logger().info("Robot ID subscriber created.")

        self.bbox_publisher = self.create_publisher(Int32MultiArray, "bounding_box", 10)
        self.override_publisher = self.create_publisher(Bool, "override", 10)
        self.override = Bool()
        self.bbox_msg = Int32MultiArray()
        self.bbox_msg.data = [320, 320, 320, 320]

    def listener_callback(self, msg):
        self.get_logger().info("Received image message.")
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            height, width = frame.shape[:2]
            self.get_logger().info(f"Frame shape: {frame.shape}")

            # Padding debug output
            required_padding = width - height
            padding = required_padding // 2
            padded_image = cv2.copyMakeBorder(
                frame, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            self.get_logger().info("Padded image.")

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
                if self.task == "detect":
                    if not hasattr(r, "boxes") or r.boxes is None:
                        continue
                    bboxs = r.boxes.xyxy.cpu().int().tolist()
                else:
                    if not hasattr(r, "obb") and r.obb is not None:
                        continue
                    bboxs = r.obb.xyxy.cpu().int().tolist()

                self.get_logger().info(f"Bounding boxes: {bboxs}")

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
        except Exception as e:
            self.get_logger().error(f"Error in listener_callback: {e}")

    def id_listener_callback(self, msg):
        self.target_class = BLUE_CLASS if msg.data in RED_TEAM else RED_CLASS
        self.get_logger().info(
            f"Set target class to {'BLUE' if self.target_class == BLUE_CLASS else 'RED'}."
        )

    def check_enemy(self, roi_image):
        if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
            self.get_logger().warn("Invalid ROI shape; returning False.")
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
        self.get_logger().info(
            f"Color area: {color_area}, Threshold: {color_threshold:.1f}, Percentage: {color_percentage:.1f}%"
        )

        return color_area > color_threshold, color_area
