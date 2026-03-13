import rclpy
import sys
import argparse
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter
from rclpy.node import Node
from std_msgs.msg import Int32, Int32MultiArray, Bool, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import traceback
import time

# Import components
try:
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
    def __init__(self, parameter_overrides=None):
        super().__init__("yolov8_modular_node", parameter_overrides=parameter_overrides)
        self.get_logger().info("Initializing Modular YOLOv8 Node...")
        package_share_dir = Path(get_package_share_directory("yolov8_ros"))
        default_model_path = package_share_dir / "models" / "best_jithendra.pt"

        # --- Parameters ---
        self.declare_parameter("model_path", str(default_model_path))
        self.declare_parameter("task", "obb")
        self.declare_parameter("type", "yolo")
        self.declare_parameter("half", True)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("show", False)
        self.declare_parameter("conf", 0.05)
        self.declare_parameter("tracker_timeout", 15)
        self.declare_parameter("prediction_lookahead_steps", 0.5)

        # Kalman / Tracking
        self.declare_parameter("enable_tracking", True)
        self.declare_parameter("persist_tracking", True)
        self.declare_parameter("use_kalman", False)
        self.declare_parameter("enable_color_verification", False)

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

        self.enable_tracking = self.get_parameter("enable_tracking").value
        self.persist_tracking = self.get_parameter("persist_tracking").value
        self.use_kalman = self.get_parameter("use_kalman").value
        self.enable_color_verification = self.get_parameter(
            "enable_color_verification"
        ).value

        self.hfov_rad = math.radians(self.get_parameter("hfov_degrees").value)
        self.vfov_rad = math.radians(self.get_parameter("vfov_degrees").value)

        # State
        self.target_class = BLUE_CLASS
        self.active_trackers = {}
        self.last_frame_time = None

        # --- Components Init ---
        # 1. Detector
        self.detector = YOLODetector(self.model_path, self.task, self.model_type)
        self.get_logger().info("Detector Initialized.")

        # 2. CV Bridge
        self.bridge = CvBridge()

        # --- Publishers/Subscribers ---
        self.image_subscription = self.create_subscription(
            Image, "image_raw", self.image_callback, 10
        )
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

    def _get_inference_size(self):
        if self.imgsz_width == self.imgsz_height:
            return self.imgsz_width
        return (self.imgsz_height, self.imgsz_width)

    def _run_inference(self, image):
        imgsz_value = self._get_inference_size()
        if self.use_kalman and self.enable_tracking:
            return self.detector.track(
                image,
                half=self.half,
                imgsz=imgsz_value,
                show=False,
                conf=self.conf,
                verbose=False,
                persist=self.persist_tracking,
                tracker="bytetrack.yaml",
            )
        return self.detector.predict(
            image,
            half=self.half,
            imgsz=imgsz_value,
            show=False,
            conf=self.conf,
            verbose=False,
        )

    def _resolve_class_name(self, result_obj, class_id):
        if class_id is None:
            return "target"
        names = getattr(result_obj, "names", None)
        if names is None:
            names = getattr(self.detector.model, "names", {})
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _extract_detections(self, result_obj):
        boxes = None
        if hasattr(result_obj, "boxes") and result_obj.boxes is not None:
            boxes = result_obj.boxes
        elif (
            hasattr(result_obj, "obb")
            and result_obj.obb is not None
            and self.task == "obb"
        ):
            boxes = result_obj.obb
        if boxes is None or not hasattr(boxes, "xyxy") or boxes.xyxy is None:
            return []
        bbox_list = boxes.xyxy.cpu().int().tolist()
        detection_count = len(bbox_list)
        id_list = (
            boxes.id.cpu().int().tolist()
            if hasattr(boxes, "id") and boxes.id is not None
            else [None] * detection_count
        )
        class_list = (
            boxes.cls.cpu().int().tolist()
            if hasattr(boxes, "cls") and boxes.cls is not None
            else [None] * detection_count
        )
        confidence_list = (
            boxes.conf.cpu().tolist()
            if hasattr(boxes, "conf") and boxes.conf is not None
            else [0.0] * detection_count
        )
        detections = []
        for index, bbox in enumerate(bbox_list):
            class_id = class_list[index]
            detections.append(
                {
                    "bbox": tuple(bbox),
                    "track_id": id_list[index],
                    "class_id": class_id,
                    "confidence": float(confidence_list[index]),
                    "class_name": self._resolve_class_name(result_obj, class_id),
                }
            )
        return detections

    def _draw_detection(self, frame, detection, is_target):
        x1, y1, x2, y2 = detection["bbox"]
        color = (0, 255, 255) if is_target else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_parts = [detection["class_name"], f"{detection['confidence']:.2f}"]
        if detection["track_id"] is not None:
            label_parts.append(f"ID:{detection['track_id']}")
        if is_target:
            label_parts.append("target")
        cv2.putText(
            frame,
            " ".join(label_parts),
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    def image_callback(self, msg: Image):
        try:
            # 1. Get Frame
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if frame is None:
                self.get_logger().warn("Failed to convert image")
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
            results = self._run_inference(padded_image)

            # 4. Process Results
            vis_frame = padded_image.copy() if self.show else None
            published_this_frame = False
            current_processed_ids = set()
            best_direct_detection = None

            if results and len(results) > 0:
                res_obj = results[0]
                detections = self._extract_detections(res_obj)
                for detection in detections:
                    x1, y1, x2, y2 = detection["bbox"]
                    track_id = detection["track_id"]
                    class_id = detection["class_id"]
                    matches_target_class = (
                        class_id is None or class_id == self.target_class
                    )
                    matches_target_color = True
                    roi = padded_image[
                        max(0, y1) : min(y2, self.imgsz_height),
                        max(0, x1) : min(x2, self.imgsz_width),
                    ]
                    if self.enable_color_verification:
                        matches_target_color, _ = check_enemy(roi, self.target_class)
                    is_target_detection = matches_target_class and matches_target_color
                    if self.show and vis_frame is not None:
                        self._draw_detection(vis_frame, detection, is_target_detection)
                    if not is_target_detection:
                        continue
                    if (
                        best_direct_detection is None
                        or detection["confidence"] > best_direct_detection["confidence"]
                    ):
                        best_direct_detection = detection
                    if (
                        self.use_kalman
                        and self.enable_tracking
                        and track_id is not None
                    ):
                        current_processed_ids.add(track_id)
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        w, h = float(abs(x2 - x1)), float(abs(y2 - y1))
                        if track_id not in self.active_trackers:
                            self.active_trackers[track_id] = TrackerWrapper(
                                ArmorPlateTracker(cx, cy, w, h)
                            )
                        else:
                            self.active_trackers[track_id].tracker.update(cx, cy, w, h)
                        self.active_trackers[track_id].lost_frames = 0

            # 5. Tracking / Publishing Logic
            if self.use_kalman and self.enable_tracking:
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
                                cv2.putText(
                                    vis_frame,
                                    f"KF ID:{tid}",
                                    (bx1, max(20, by1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 0, 0),
                                    1,
                                )

                for tid in ids_to_remove:
                    del self.active_trackers[tid]

            if not published_this_frame and best_direct_detection is not None:
                x1, y1, x2, y2 = best_direct_detection["bbox"]
                self.publish_bbox_and_angles(x1, y1, x2, y2, True)
                published_this_frame = True

            # Default publish if nothing found
            if not published_this_frame:
                center_x = self.imgsz_width // 2
                center_y = self.imgsz_height // 2
                self.publish_bbox_and_angles(
                    center_x, center_y, center_x, center_y, False
                )

            if self.show and vis_frame is not None:
                # The window name should not depend on camera_type anymore
                cv2.imshow("YOLOv8 Modular Node", vis_frame)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in image_callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def destroy_node(self):
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    # Parse custom arguments if running as script
    parser = argparse.ArgumentParser(description="YOLOv8 Node")
    parser.add_argument(
        "--show", type=str, default=None, help="Enable visualization (True/False)"
    )

    source_args = args if args is not None else sys.argv[1:]
    parsed_args, remaining_args = parser.parse_known_args(source_args)

    rclpy.init(args=remaining_args)

    param_overrides = []
    if parsed_args.show is not None:
        val = parsed_args.show.lower() in ("true", "1", "yes")
        param_overrides.append(Parameter("show", Parameter.Type.BOOL, val))

    node = YOLOv8Node(parameter_overrides=param_overrides)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(f"Unexpected error: {exc}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
