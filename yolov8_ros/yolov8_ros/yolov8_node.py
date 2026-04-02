import rclpy
import sys
import argparse
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

try:
    from .detector import YOLODetector
    from .tracker import ArmorPlateTracker, UKFArmorPlateTracker, TrackerWrapper
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
    from tracker import ArmorPlateTracker, UKFArmorPlateTracker, TrackerWrapper
    from utils import (
        RED_TEAM,
        BLUE_TEAM,
        RED_CLASS,
        BLUE_CLASS,
        letterbox_frame,
        calculate_angles,
        check_enemy,
    )

PREVIEW_MAX_W = 960
PREVIEW_MAX_H = 540


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


_COLOR_TABLE = [
    tuple(
        cv2.cvtColor(np.uint8([[[(c * 40) % 180, 220, 255]]]), cv2.COLOR_HSV2BGR)[
            0, 0
        ].tolist()
    )
    for c in range(80)
]


def _class_color(cid: int) -> tuple:
    return _COLOR_TABLE[cid % len(_COLOR_TABLE)]


def _class_name(names, cid: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cid, cid))
    if isinstance(names, (list, tuple)) and 0 <= cid < len(names):
        return str(names[cid])
    return str(cid)


def tick_fps_ema(prev: float, dt: float) -> float:
    inst = 1.0 / max(dt, 1e-9)
    if prev <= 0.0:
        return inst
    return 0.9 * prev + 0.1 * inst


def preview_scale(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def draw_detections(frame: np.ndarray, result, task: str) -> None:
    """Lightweight replacement for result.plot() — avoids full annotator overhead."""
    names = getattr(result, "names", None)
    if task == "obb":
        obb = getattr(result, "obb", None)
        if obb is None:
            return
        xyxyxyxy = getattr(obb, "xyxyxyxy", None)
        if xyxyxyxy is None or len(xyxyxyxy) == 0:
            return
        polys = _to_numpy(xyxyxyxy).astype(np.int32)
        confs = _to_numpy(obb.conf)
        clss = _to_numpy(obb.cls).astype(int)
        for i, poly in enumerate(polys):
            cid = int(clss[i])
            color = _class_color(cid)
            cv2.polylines(frame, [poly.reshape(-1, 1, 2)], True, color, 2)
            xi = int(poly[:, 0].min())
            yi = max(16, int(poly[:, 1].min()) - 4)
            label = f"{_class_name(names, cid)} {float(confs[i]):.2f}"
            cv2.putText(
                frame,
                label,
                (xi, yi),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    else:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return
        bboxes = _to_numpy(boxes.xyxy).astype(int)
        confs = _to_numpy(boxes.conf)
        clss = _to_numpy(boxes.cls).astype(int)
        for i, box in enumerate(bboxes):
            cid = int(clss[i])
            color = _class_color(cid)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{_class_name(names, cid)} {float(confs[i]):.2f}"
            cv2.putText(
                frame,
                label,
                (box[0], max(16, box[1] - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )


class YOLOv8Node(Node):
    def __init__(self, parameter_overrides=None):
        super().__init__("yolov8_modular_node", parameter_overrides=parameter_overrides)
        self.get_logger().info("Initializing Modular YOLOv8 Node...")
        default_model_path = "/home/ultraviolet/CV_Detection/model_test/best.pt"

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

        self.declare_parameter("enable_tracking", True)
        self.declare_parameter("persist_tracking", True)
        self.declare_parameter("use_kalman", False)
        self.declare_parameter("tracker_type", "lkf")
        self.declare_parameter("enable_color_verification", False)

        self.declare_parameter("hfov_degrees", 79.2)
        self.declare_parameter("vfov_degrees", 63.78)
        self.declare_parameter("vis_window_name", "YOLOv8 Modular Node")

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
        self.tracker_type = self.get_parameter("tracker_type").value.lower()
        self.enable_color_verification = self.get_parameter(
            "enable_color_verification"
        ).value

        self.hfov_rad = math.radians(self.get_parameter("hfov_degrees").value)
        self.vfov_rad = math.radians(self.get_parameter("vfov_degrees").value)
        self.vis_window_name = self.get_parameter("vis_window_name").value

        self.target_class = BLUE_CLASS
        self.active_trackers = {}
        self.last_frame_time = None
        self._fps_ema = 0.0
        self._fps_t_prev = None
        self._window_created = False

        self.detector = YOLODetector(self.model_path, self.task, self.model_type)
        self.get_logger().info("Detector Initialized.")

        self.bridge = CvBridge()

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
        self._angle_msg = Float32MultiArray()

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
        self._angle_msg.data = [float(yaw), float(pitch)]
        self.angle_publisher.publish(self._angle_msg)

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if frame is None:
                self.get_logger().warn("Failed to convert image")
                return

            current_time = self.get_clock().now()
            dt = 1.0 / 60.0
            if self.last_frame_time is not None:
                dt = (current_time - self.last_frame_time).nanoseconds / 1e9
                if dt <= 0:
                    dt = 0.001
            self.last_frame_time = current_time

            padded = letterbox_frame(frame, self.imgsz_width, self.imgsz_height)

            imgsz_val = (
                self.imgsz_width
                if self.imgsz_width == self.imgsz_height
                else (self.imgsz_height, self.imgsz_width)
            )

            if self.enable_tracking:
                results = self.detector.track(
                    padded,
                    half=self.half,
                    imgsz=imgsz_val,
                    show=False,
                    conf=self.conf,
                    verbose=False,
                    persist=self.persist_tracking,
                    tracker="bytetrack.yaml",
                )
            else:
                results = self.detector.predict(
                    padded,
                    half=self.half,
                    imgsz=imgsz_val,
                    show=False,
                    conf=self.conf,
                    verbose=False,
                )

            vis_frame = padded.copy() if self.show else None

            published_this_frame = False
            current_processed_ids = set()
            best_direct_bbox = None

            if results and len(results) > 0:
                res_obj = results[0]

                if self.show and vis_frame is not None:
                    draw_detections(vis_frame, res_obj, self.task)

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
                    bbox_arr = boxes.xyxy.cpu().numpy().astype(np.int32)
                    id_arr = (
                        boxes.id.cpu().numpy().astype(np.int32)
                        if hasattr(boxes, "id") and boxes.id is not None
                        else None
                    )

                    for i in range(len(bbox_arr)):
                        x1, y1, x2, y2 = bbox_arr[i]
                        track_id = int(id_arr[i]) if id_arr is not None else None

                        if (
                            self.show
                            and vis_frame is not None
                            and self.use_kalman
                            and track_id is not None
                        ):
                            cv2.putText(
                                vis_frame,
                                f"ID:{track_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )

                        roi = padded[
                            max(0, y1) : min(y2, self.imgsz_height),
                            max(0, x1) : min(x2, self.imgsz_width),
                        ]
                        is_enemy = True
                        if self.enable_color_verification:
                            is_enemy, _ = check_enemy(roi, self.target_class)

                        if is_enemy:
                            if self.use_kalman:
                                if track_id is not None:
                                    current_processed_ids.add(track_id)
                                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                                    w, h = float(abs(x2 - x1)), float(abs(y2 - y1))
                                    if track_id not in self.active_trackers:
                                        if self.tracker_type == "ukf":
                                            trk = UKFArmorPlateTracker(cx, cy, w, h)
                                        else:
                                            trk = ArmorPlateTracker(cx, cy, w, h)
                                        self.active_trackers[track_id] = TrackerWrapper(
                                            trk
                                        )
                                    else:
                                        self.active_trackers[track_id].tracker.update(
                                            cx, cy, w, h
                                        )
                                    self.active_trackers[track_id].lost_frames = 0
                            else:
                                if best_direct_bbox is None:
                                    best_direct_bbox = (x1, y1, x2, y2)

            if self.use_kalman:
                ids_to_remove = []
                for tid, wrapper in self.active_trackers.items():
                    if tid not in current_processed_ids:
                        wrapper.lost_frames += 1
                    if wrapper.lost_frames >= self.tracker_timeout:
                        ids_to_remove.append(tid)
                        continue
                    if not published_this_frame and wrapper.lost_frames == 0:
                        px, py = wrapper.tracker.predict(dt)
                        state = wrapper.tracker.kf.statePost.flatten()
                        vx, vy = state[2], state[3]
                        ax, ay = state[4], state[5]
                        t_look = self.lookahead_steps * dt
                        xf = px + vx * t_look + 0.5 * ax * t_look**2
                        yf = py + vy * t_look + 0.5 * ay * t_look**2
                        w, h = wrapper.tracker.width, wrapper.tracker.height
                        bx1 = max(0, min(int(xf - w / 2), self.imgsz_width - 1))
                        by1 = max(0, min(int(yf - h / 2), self.imgsz_height - 1))
                        bx2 = max(0, min(int(xf + w / 2), self.imgsz_width - 1))
                        by2 = max(0, min(int(yf + h / 2), self.imgsz_height - 1))
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
                if best_direct_bbox:
                    x1, y1, x2, y2 = best_direct_bbox
                    self.publish_bbox_and_angles(x1, y1, x2, y2, True)
                    published_this_frame = True

            if not published_this_frame:
                c = self.imgsz_width // 2
                self.publish_bbox_and_angles(c, c, c, c, False)

            if self.show and vis_frame is not None:
                t_now = time.monotonic()
                if self._fps_t_prev is not None:
                    self._fps_ema = tick_fps_ema(
                        self._fps_ema, t_now - self._fps_t_prev
                    )
                self._fps_t_prev = t_now
                cv2.putText(
                    vis_frame,
                    f"FPS: {self._fps_ema:.1f}",
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if not self._window_created:
                    cv2.namedWindow(self.vis_window_name, cv2.WINDOW_NORMAL)
                    self._window_created = True
                view = preview_scale(vis_frame, PREVIEW_MAX_W, PREVIEW_MAX_H)
                cv2.imshow(self.vis_window_name, view)
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
