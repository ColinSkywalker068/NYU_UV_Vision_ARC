#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import qos_profile_sensor_data
import cv2
import os
import time


class FrameGrabberNode(Node):
    """
    A simple ROS2 node to grab and save frames from an image topic.
    This version's architecture mirrors a typical vision node where processing
    happens directly in the callback.
    """

    def __init__(self):
        super().__init__("frame_grabber_node")

        # --- Parameters ---
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )

        # --- Node State ---
        self.bridge = CvBridge()
        self.save_count = 0
        self.save_dir = "saved_frames"

        # --- ROS Subscriptions ---
        self.subscription = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )

        # Create directory for saving frames if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.get_logger().info(f"Subscribed to '{image_topic}'")
        self.get_logger().info(
            f"Frames will be saved in the '{self.save_dir}/' directory."
        )
        self.get_logger().info("Press 's' in the display window to save a frame.")
        self.get_logger().info("Press 'q' in the display window to quit.")

        # Create the window here. It will be updated in the callback.
        cv2.namedWindow("Frame Grabber")

    def image_callback(self, msg: Image):
        """
        Callback to process incoming image messages, display them, and handle input.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # Display the frame
        cv2.imshow("Frame Grabber", cv_image)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            self.save_frame(cv_image)
        elif key == ord("q"):
            self.get_logger().info("'q' pressed, shutting down node.")
            self.destroy_node()  # This will cause rclpy.spin() to exit

    def save_frame(self, frame):
        """
        Saves the given frame to a file with a unique name.
        """
        if frame is None:
            self.get_logger().warn("No frame available to save.")
            return

        timestamp = int(time.time())
        filename = os.path.join(
            self.save_dir, f"frame_{timestamp}_{self.save_count:04d}.png"
        )
        try:
            cv2.imwrite(filename, frame)
            self.get_logger().info(f"Saved frame to {filename}")
            self.save_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to save frame: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FrameGrabberNode()
    try:
        # spin() will block until the node is shut down (e.g., by destroy_node()).
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Allow Ctrl+C to exit gracefully
    finally:
        # This block executes regardless of how spin() exited.
        node.get_logger().info("Final cleanup.")
        cv2.destroyAllWindows()
        # The node may or may not be already destroyed.
        if node.context.ok():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
