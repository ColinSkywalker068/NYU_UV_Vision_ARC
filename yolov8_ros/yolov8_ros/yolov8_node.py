import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Int32MultiArray, Bool, Float32MultiArray
import cv2
import numpy as np
import math
import traceback
import time

# Import components
try:
    from .camera import BaslerCamera
    from .detector import YOLODetector
    from .tracker import ArmorPlateTracker, TrackerWrapper
    from .utils import (
        RED_TEAM,
        BLUE_TEAM,
        RED_CLASS,
        BLUE_CLASS,
        letterbox_frame,
        calculate_angles,
        check_enemy,
    )
except (ImportError, ValueError):
    from camera import BaslerCamera
    from detector import YOLODetector
    from tracker import ArmorPlateTracker, TrackerWrapper
    from utils import (
        RED_TEAM,
        BLUE_TEAM,
        RED_CLASS,
        BLUE_CLASS,
        letterbox_frame,
        calculate_angles,
        check_enemy,
    )


class YOLOv8Node(Node):
    def __init__(self):
        super().__init__("yolov8_modular_node")
        self.get_logger().info("Initializing Modular YOLOv8 Node with Basler Camera...")

        # --- Parameters ---
        self.declare_parameter(
            "model_path", "/home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine"
        )
        self.declare_parameter("task", "detect")
        self.declare_parameter("type", "yolo")
        self.declare_parameter("half", True)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("show", False)
        self.declare_parameter("conf", 0.05)
        self.declare_parameter("tracker_timeout", 15)
        self.declare_parameter("prediction_lookahead_steps", 0.5)

        # Basler specific
        self.declare_parameter("basler_exposure_us", 5000.0)
        self.declare_parameter("basler_gain_db", 5.0)
        self.declare_parameter("basler_fps", 0)  # 0 for max

        # Kalman / Tracking
        self.declare_parameter("enable_tracking", True)
        self.declare_parameter("persist_tracking", True)
        self.declare_parameter("use_kalman", False)

        # Angle calc
        self.declare_parameter(
            "hfov_degrees", 79.2
        )  # Keeping D455 defaults or adjust for Basler
        self.declare_parameter("vfov_degrees", 63.78)

        # Retrieve params
        self.model_path = self.get_parameter("model_path").value
        self.task = self.get_parameter("task").value
        self.model_type = self.get_parameter("type").value
        self.half = self.get_parameter("half").value
        self.imgsz_width = self.get_parameter("imgsz_width").value
        self.imgsz_height = self.get_parameter("imgsz_height").value
        self.show = self.get_parameter("show").value
        self.conf = self.get_parameter("conf").value
        self.tracker_timeout = self.get_parameter("tracker_timeout").value
        self.lookahead_steps = self.get_parameter("prediction_lookahead_steps").value

        self.basler_exposure = self.get_parameter("basler_exposure_us").value
        self.basler_gain = self.get_parameter("basler_gain_db").value
        self.basler_fps = self.get_parameter("basler_fps").value

        self.enable_tracking = self.get_parameter("enable_tracking").value
        self.persist_tracking = self.get_parameter("persist_tracking").value
        self.use_kalman = self.get_parameter("use_kalman").value

        self.hfov_rad = math.radians(self.get_parameter("hfov_degrees").value)
        self.vfov_rad = math.radians(self.get_parameter("vfov_degrees").value)

        # State
        self.target_class = BLUE_CLASS
        self.active_trackers = {}
        self.last_frame_time = None

        # --- Components Init ---

        # 1. Camera
        try:
            self.camera = BaslerCamera(
                width=self.imgsz_width,
                height=self.imgsz_height,
                fps=self.basler_fps,
                exposure_us=self.basler_exposure,
                gain_db=self.basler_gain,
            )
            self.camera.initialize()
            self.get_logger().info("Basler Camera Initialized.")
        except Exception as e:
            self.get_logger().error(f"Failed to init camera: {e}")
            raise e

        # 2. Detector
        self.detector = YOLODetector(self.model_path, self.task, self.model_type)
        self.get_logger().info("Detector Initialized.")

        # --- Publishers/Subscribers ---
        self.int_subscription = self.create_subscription(
            Int32, "robot_id", self.id_listener_callback, 10
        )
        self.bbox_publisher = self.create_publisher(Int32MultiArray, "bounding_box", 10)
        self.override_publisher = self.create_publisher(Bool, "override", 10)
        self.angle_publisher = self.create_publisher(
            Float32MultiArray, "bounding_box_angles", 10
        )

        self.bbox_msg = Int32MultiArray()
        self.override_msg = Bool()

        # Timer
        timer_fps = self.basler_fps if self.basler_fps > 0 else 60
        self.timer = self.create_timer(1.0 / timer_fps, self.timer_callback)

    def id_listener_callback(self, msg: Int32):
        robot_id = msg.data
        if robot_id in RED_TEAM:
            self.target_class = BLUE_CLASS
            self.get_logger().info(f"ID {robot_id} is RED. Targeting BLUE.")
        elif robot_id in BLUE_TEAM:
            self.target_class = RED_CLASS
            self.get_logger().info(f"ID {robot_id} is BLUE. Targeting RED.")

    def publish_bbox_and_angles(self, x1, y1, x2, y2, override_bool):
        self.bbox_msg.data = [int(x1), int(y1), int(x2), int(y2)]
        self.override_msg.data = override_bool
        self.bbox_publisher.publish(self.bbox_msg)
        self.override_publisher.publish(self.override_msg)

        yaw, pitch = calculate_angles(
            x1,
            y1,
            x2,
            y2,
            self.imgsz_width,
            self.imgsz_height,
            self.hfov_rad,
            self.vfov_rad,
        )
        angle_msg = Float32MultiArray()
        angle_msg.data = [float(yaw), float(pitch)]
        self.angle_publisher.publish(angle_msg)

    def timer_callback(self):
        try:
            # 1. Get Frame
            frame = self.camera.get_frame()
            if frame is None:
                return

            # dt calc
            current_time = self.get_clock().now()
            dt = 1.0 / 60.0  # Default
            if self.last_frame_time is not None:
                dt = (current_time - self.last_frame_time).nanoseconds / 1e9
                if dt <= 0:
                    dt = 0.001
            self.last_frame_time = current_time

            # 2. Preprocess
            padded_image = letterbox_frame(frame, self.imgsz_width, self.imgsz_height)

            # 3. Inference
            if self.imgsz_width == self.imgsz_height:
                imgsz_val = self.imgsz_width
            else:
                imgsz_val = (self.imgsz_height, self.imgsz_width)

            if self.enable_tracking:
                results = self.detector.track(
                    padded_image,
                    half=self.half,
                    imgsz=imgsz_val,
                    show=False,
                    conf=self.conf,
                    persist=self.persist_tracking,
                    tracker="bytetrack.yaml",
                )
            else:
                results = self.detector.predict(
                    padded_image,
                    half=self.half,
                    imgsz=imgsz_val,
                    show=False,
                    conf=self.conf,
                    verbose=False,
                )

            # 4. Process Results
            vis_frame = padded_image.copy() if self.show else None
            published_this_frame = False
            current_processed_ids = set()
            best_direct_bbox = None  # Fallback if Kalman disabled

            if results and len(results) > 0:
                res_obj = results[0]
                boxes = None
                if hasattr(res_obj, "boxes") and res_obj.boxes is not None:
                    boxes = res_obj.boxes
                elif (
                    hasattr(res_obj, "obb")
                    and res_obj.obb is not None
                    and self.task == "obb"
                ):
                    boxes = res_obj.obb

                if boxes is not None and hasattr(boxes, "xyxy"):
                    bbox_list = boxes.xyxy.cpu().int().tolist()
                    id_list = (
                        boxes.id.cpu().int().tolist()
                        if hasattr(boxes, "id") and boxes.id is not None
                        else [None] * len(bbox_list)
                    )

                    for i, bbox in enumerate(bbox_list):
                        x1, y1, x2, y2 = bbox
                        track_id = id_list[i]

                        # Visualization
                        if self.show and vis_frame is not None:
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            if track_id is not None:
                                cv2.putText(
                                    vis_frame,
                                    f"ID:{track_id}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                )

                        # Check Enemy
                        roi = padded_image[
                            max(0, y1) : min(y2, self.imgsz_height),
                            max(0, x1) : min(x2, self.imgsz_width),
                        ]
                        is_enemy, _ = check_enemy(roi, self.target_class)

                        if is_enemy:
                            if self.use_kalman:
                                if track_id is not None:
                                    current_processed_ids.add(track_id)
                                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                                    w, h = float(abs(x2 - x1)), float(abs(y2 - y1))

                                    if track_id not in self.active_trackers:
                                        self.active_trackers[track_id] = TrackerWrapper(
                                            ArmorPlateTracker(cx, cy, w, h)
                                        )
                                    else:
                                        self.active_trackers[track_id].tracker.update(
                                            cx, cy, w, h
                                        )
                                    self.active_trackers[track_id].lost_frames = 0
                            else:
                                if best_direct_bbox is None:
                                    best_direct_bbox = (x1, y1, x2, y2)

            # 5. Tracking / Publishing Logic
            if self.use_kalman:
                ids_to_remove = []
                for tid, wrapper in self.active_trackers.items():
                    if tid not in current_processed_ids:
                        wrapper.lost_frames += 1

                    if wrapper.lost_frames >= self.tracker_timeout:
                        ids_to_remove.append(tid)
                        continue

                    if not published_this_frame and wrapper.lost_frames == 0:
                        # Predict
                        px, py = wrapper.tracker.predict(dt)

                        # Lookahead
                        state = wrapper.tracker.kf.statePost.flatten()
                        vx, vy = state[2], state[3]
                        ax, ay = state[4], state[5]
                        t_look = self.lookahead_steps * dt

                        xf = px + vx * t_look + 0.5 * ax * t_look**2
                        yf = py + vy * t_look + 0.5 * ay * t_look**2

                        w, h = wrapper.tracker.width, wrapper.tracker.height

                        bx1 = int(xf - w / 2)
                        by1 = int(yf - h / 2)
                        bx2 = int(xf + w / 2)
                        by2 = int(yf + h / 2)

                        # Clamp
                        bx1 = max(0, min(bx1, self.imgsz_width - 1))
                        by1 = max(0, min(by1, self.imgsz_height - 1))
                        bx2 = max(0, min(bx2, self.imgsz_width - 1))
                        by2 = max(0, min(by2, self.imgsz_height - 1))

                        if bx1 < bx2 and by1 < by2:
                            self.publish_bbox_and_angles(bx1, by1, bx2, by2, True)
                            published_this_frame = True

                            if self.show and vis_frame is not None:
                                cv2.rectangle(
                                    vis_frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2
                                )

                for tid in ids_to_remove:
                    del self.active_trackers[tid]

            else:
                # Direct publish if not using Kalman
                if best_direct_bbox:
                    x1, y1, x2, y2 = best_direct_bbox
                    self.publish_bbox_and_angles(x1, y1, x2, y2, True)
                    published_this_frame = True

            # Default publish if nothing found
            if not published_this_frame:
                c = self.imgsz_width // 2
                self.publish_bbox_and_angles(c, c, c, c, False)

            if self.show and vis_frame is not None:
                cv2.imshow("YOLOv8 Basler Node", vis_frame)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def destroy_node(self):
        if hasattr(self, "camera") and self.camera:
            self.camera.release()
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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
