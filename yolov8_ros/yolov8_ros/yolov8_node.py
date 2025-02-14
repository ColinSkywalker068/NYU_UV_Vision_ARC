#!/usr/bin/env python3

"""
ROS2 Node for object detection and tracking using YOLOv8.
This node processes video frames to detect and track objects, with specific focus on
team-based color detection (red vs blue teams).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Int32MultiArray, Bool
from cv_bridge import CvBridge
from ultralytics import YOLO, RTDETR, NAS
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Enable GTK synchronization for display
os.environ["GDK_SYNCHRONIZE"] = "1"


@dataclass
class ColorRange:
    """Represents HSV color range for detection"""

    lower: np.ndarray
    upper: np.ndarray


# Constants
class TeamIDs:
    """Team identification constants"""

    RED_TEAM = frozenset(range(1, 12))  # IDs 1-11
    BLUE_TEAM = frozenset(range(101, 112))  # IDs 101-111


class TeamClass:
    """Team class identifiers"""

    RED = 1
    BLUE = 0


class ColorRanges:
    """HSV color ranges for team detection"""

    BLUE = [
        ColorRange(
            np.array((90, 50, 70), np.uint8), np.array((128, 255, 255), np.uint8)
        )
    ]
    RED = [
        ColorRange(np.array((0, 50, 70), np.uint8), np.array((9, 255, 255), np.uint8)),
        ColorRange(
            np.array((159, 50, 70), np.uint8), np.array((180, 255, 255), np.uint8)
        ),
    ]


class DetectionConfig:
    """Detection configuration constants"""

    BBOX_COLOR = (0, 255, 0)
    BBOX_LINE_SIZE = 2
    COLOR_THRESHOLD = 200
    DEFAULT_BBOX = [320, 320, 320, 320]


class YOLOv8Node(Node):
    """
    ROS2 Node for YOLOv8-based object detection and tracking.

    This node subscribes to video frames and robot ID topics, processes frames using
    YOLOv8 (or similar models), and publishes bounding box and override information.
    """

    def __init__(self):
        """Initialize the YOLOv8 node with parameters and publishers/subscribers."""
        super().__init__("yolov8_node")
        self.get_logger().info("Initializing YOLOv8Node...")

        self._init_parameters()
        self._setup_model()
        self._init_ros_interfaces()

    def _init_parameters(self):
        """Initialize and declare all ROS parameters."""
        # Define parameter defaults
        param_defaults = {
            "model_path": "/home/ultraviolet/Downloads/yolo/best.pt",
            "task": "detect",
            "type": "yolo",
            "half": True,
            "imgsz": 640,
            "show": True,
            "conf": 0.5,
        }

        # Declare and get parameters
        for name, default in param_defaults.items():
            self.declare_parameter(name, default)

        # Store parameters as instance variables
        self.model_path = self.get_parameter("model_path").value
        self.model_type = self.get_parameter("type").value
        self.task = self.get_parameter("task").value
        self.half = self.get_parameter("half").value
        self.imgsz = self.get_parameter("imgsz").value
        self.show = self.get_parameter("show").value
        self.conf = self.get_parameter("conf").value

        # Initialize target class
        self.target_class = TeamClass.RED

    def _setup_model(self):
        """Initialize the AI model based on specified type."""
        model_classes = {"yolo": YOLO, "rtdetr": RTDETR, "nas": NAS}

        if self.model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        ModelClass = model_classes[self.model_type]
        self.model = ModelClass(self.model_path, task=self.task)
        self.get_logger().info(f"Initialized {self.model_type} model")

    def _init_ros_interfaces(self):
        """Initialize ROS publishers and subscribers."""
        # Create subscribers
        self.create_subscription(Image, "video_frames", self._process_frame, 10)
        self.create_subscription(Int32, "robot_id", self._handle_robot_id, 10)

        # Create publishers
        self.bbox_publisher = self.create_publisher(Int32MultiArray, "bounding_box", 10)
        self.override_publisher = self.create_publisher(Bool, "override", 10)

        # Initialize message objects
        self.override_msg = Bool()
        self.bbox_msg = Int32MultiArray()
        self.bbox_msg.data = DetectionConfig.DEFAULT_BBOX

        # Initialize CV bridge
        self.bridge = CvBridge()

    def _process_frame(self, msg: Image):
        """
        Process incoming video frames for object detection.

        Args:
            msg (Image): ROS Image message containing the video frame
        """
        try:
            # Convert ROS Image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            padded_image = self._pad_image(frame)

            # Perform detection
            results = self.model.track(
                padded_image,
                half=self.half,
                imgsz=self.imgsz,
                show=self.show,
                conf=self.conf,
            )

            self._process_results(results, padded_image)

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {str(e)}")

    def _pad_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Pad the input image to make it square.

        Args:
            frame (np.ndarray): Input image

        Returns:
            np.ndarray: Padded square image
        """
        height, width = frame.shape[:2]
        padding = (width - height) // 2
        return cv2.copyMakeBorder(
            frame, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    def _process_results(self, results, padded_image: np.ndarray):
        """
        Process detection results and publish findings.

        Args:
            results: YOLOv8 detection results
            padded_image (np.ndarray): Processed image
        """
        self.override_msg.data = False

        for r in results:
            if not hasattr(r, "obb") or r.obb is None:
                continue

            bboxes = r.obb.xyxy.cpu().int().tolist()
            for bbox in bboxes:
                roi = padded_image[bbox[1] : bbox[3], bbox[0] : bbox[2]]

                if self._check_enemy(roi):
                    self._handle_detection(bbox, padded_image)
                else:
                    self._publish_default_values()

    def _handle_detection(self, bbox: list[int], image: np.ndarray):
        """Handle positive detection by updating visualization and publishing data."""
        if self.show:
            cv2.rectangle(
                image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                DetectionConfig.BBOX_COLOR,
                DetectionConfig.BBOX_LINE_SIZE,
            )

        self.bbox_msg.data = bbox
        self.override_msg.data = True
        self._publish_messages()

    def _publish_default_values(self):
        """Publish default values when no detection is made."""
        self.bbox_msg.data = DetectionConfig.DEFAULT_BBOX
        self.override_msg.data = False
        self._publish_messages()

    def _publish_messages(self):
        """Publish current bbox and override messages."""
        self.override_publisher.publish(self.override_msg)
        self.bbox_publisher.publish(self.bbox_msg)

    def _handle_robot_id(self, msg: Int32):
        """
        Handle robot ID messages to determine target team.

        Args:
            msg (Int32): Robot ID message
        """
        self.target_class = (
            TeamClass.BLUE if msg.data in TeamIDs.RED_TEAM else TeamClass.RED
        )
        self.get_logger().info(
            f"Target class set to: {'BLUE' if self.target_class == TeamClass.BLUE else 'RED'}"
        )

    def _check_enemy(self, roi_image: np.ndarray) -> bool:
        """
        Check if ROI contains enemy team colors.

        Args:
            roi_image (np.ndarray): Region of interest from the image

        Returns:
            bool: True if enemy detected, False otherwise
        """
        if roi_image.size == 0:
            return False

        ranges = (
            ColorRanges.BLUE if self.target_class == TeamClass.BLUE else ColorRanges.RED
        )

        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        color_area = sum(
            np.count_nonzero(cv2.inRange(hsv_roi, r.lower, r.upper)) for r in ranges
        )

        return color_area > DetectionConfig.COLOR_THRESHOLD


def main(args=None):
    """Main function to initialize and run the node."""
    rclpy.init(args=args)

    try:
        node = YOLOv8Node()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error running node: {str(e)}")
    finally:
        if "node" in locals():
            node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
