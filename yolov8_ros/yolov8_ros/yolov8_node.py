"""YOLOv8 modular ROS 2 detection node.

Subscribes to ``image_raw``, runs YOLO inference, selects a target using
an explicit ranking policy, and publishes bounding-box, angle, override,
and (for OBB models) orientation messages.

Refactored following docs/yolov8_node/NEXT_STEPS.md items 1–16.
"""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime
import math
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, Int32MultiArray

try:
    from .detector import YOLODetector
    from .tracker import ArmorPlateTracker, TrackerWrapper
    from .utils import (
        BLUE_CLASS,
        BLUE_TEAM,
        RED_CLASS,
        RED_TEAM,
        LetterboxResult,
        calculate_angles,
        check_enemy,
        letterbox_frame,
        unletterbox_coords,
    )
except (ImportError, ValueError):
    from detector import YOLODetector
    from tracker import ArmorPlateTracker, TrackerWrapper
    from utils import (
        BLUE_CLASS,
        BLUE_TEAM,
        RED_CLASS,
        RED_TEAM,
        LetterboxResult,
        calculate_angles,
        check_enemy,
        letterbox_frame,
        unletterbox_coords,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PREVIEW_MAX_W = 960
PREVIEW_MAX_H = 540

_DEFAULT_MAX_HSV_CANDIDATES = 3
_DEFAULT_MAX_COAST_FRAMES = 3
_STAGE_LOG_INTERVAL = 500  # frames between timing summaries
_VISION_LOG_DIRNAME = "vision_logs"


def _resolve_runtime_log_dir(explicit_dir: str) -> str:
    """Resolve runtime log directory for headless mode."""
    if explicit_dir:
        return explicit_dir
    env_dir = os.getenv("VISION_LOG_DIR", "").strip()
    if env_dir:
        return env_dir
    # Keep logs with the process working directory (deployment/runtime root).
    return os.path.join(os.getcwd(), _VISION_LOG_DIRNAME)

# ---------------------------------------------------------------------------
# Candidate dataclass  (Item 15)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Candidate:
    """A single detection candidate with both model-space and source-space data."""

    bbox_model: tuple[int, int, int, int]
    bbox_source: tuple[int, int, int, int]
    center_source: tuple[float, float]
    confidence: float
    class_id: int
    track_id: int | None
    is_enemy: bool = True
    obb_angle: float | None = None  # radians, from OBB xywhr


@dataclass(slots=True)
class AngleRequest:
    bbox: tuple[int, int, int, int]
    src_size: tuple[int, int]


@dataclass(slots=True)
class KalmanPublishRequest:
    center_model: tuple[float, float]
    size_model: tuple[float, float]
    letterbox: LetterboxResult
    obb_angle: float | None


@dataclass(slots=True)
class VisionLogRecord:
    timestamp: str
    fps: float
    opponent_bbox: tuple[int, int, int, int]
    confidence: float
    color_pass: bool
    latency_ms: float


class VisionLogSink(Protocol):
    def write_line(self, line: str) -> None:
        ...

    def close(self) -> None:
        ...


class ConsoleVisionLogSink:
    def __init__(self, node_logger) -> None:
        self._node_logger = node_logger

    def write_line(self, line: str) -> None:
        self._node_logger.info(line)

    def close(self) -> None:
        return


class FileVisionLogSink:
    def __init__(self, log_dir: str) -> None:
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"vision_runtime_{stamp}.log")
        self._handle = open(self._path, "a", encoding="utf-8")

    def write_line(self, line: str) -> None:
        self._handle.write(f"{line}\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.close()

    @property
    def path(self) -> str:
        return self._path


class VisionRuntimeLogger:
    def __init__(self, sink: VisionLogSink) -> None:
        self._sink = sink
        self._last_fps_log_at: float = 0.0

    def log_fps_once_per_second(self, now_monotonic: float, fps: float) -> None:
        if now_monotonic - self._last_fps_log_at < 1.0:
            return
        self._last_fps_log_at = now_monotonic
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        self._sink.write_line(f"ts={timestamp} fps={fps:.2f}")

    def log_opponent_event(self, record: VisionLogRecord) -> None:
        x1, y1, x2, y2 = record.opponent_bbox
        self._sink.write_line(
            " ".join(
                (
                    f"ts={record.timestamp}",
                    f"fps={record.fps:.2f}",
                    f"bbox={x1},{y1},{x2},{y2}",
                    f"conf={record.confidence:.3f}",
                    f"color_pass={int(record.color_pass)}",
                    f"latency_ms={record.latency_ms:.2f}",
                )
            )
        )

    def log_note(self, level: str, message: str) -> None:
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        self._sink.write_line(f"ts={timestamp} level={level} msg={message}")

    def close(self) -> None:
        self._sink.close()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    """Convert a tensor (or array-like) to a numpy ndarray."""
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


def _resolve_default_model_path() -> str:
    """Return the package-relative default model path (models/bestmerge.engine).

    Walks up from this file to find the ``yolov8_ros`` package root, then
    appends ``models/best.engine``.  Falls back to the legacy absolute
    path if the package-relative file does not exist.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(pkg_dir, "model_test", "bestmerge.engine")
    if os.path.isfile(candidate):
        return candidate
    # Legacy fallback — kept for backward compat, emits a warning at startup
    return "/home/ultraviolet/CV_Detection/model_test/bestmerge.engine"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class YOLOv8Node(Node):
    def __init__(self, parameter_overrides=None):
        super().__init__("yolov8_modular_node", parameter_overrides=parameter_overrides)

        # --- Parameters -------------------------------------------------
        default_model = _resolve_default_model_path()

        self.declare_parameter("model_path", default_model)
        self.declare_parameter("task", "obb")
        self.declare_parameter("type", "yolo")
        self.declare_parameter("half", True)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("show", False)
        self.declare_parameter("conf", 0.45)
        self.declare_parameter("tracker_timeout", 15)
        self.declare_parameter("prediction_lookahead_steps", 0.5)
        self.declare_parameter("inference_mode", "predict")

        self.declare_parameter("persist_tracking", True)
        self.declare_parameter("use_kalman", False)
        self.declare_parameter("enable_color_verification", True)
        self.declare_parameter("target_color", "blue")
        self.declare_parameter("use_robot_id_targeting", False)
        self.declare_parameter("max_hsv_candidates", _DEFAULT_MAX_HSV_CANDIDATES)
        self.declare_parameter("max_coast_frames", _DEFAULT_MAX_COAST_FRAMES)
        self.declare_parameter("runtime_log_dir", "")

        self.declare_parameter("hfov_degrees", 79.2)
        self.declare_parameter("vfov_degrees", 63.78)
        self.declare_parameter("vis_window_name", "YOLOv8 Modular Node")

        # Read parameters
        self.model_path: str = self.get_parameter("model_path").value
        self.task: str = self.get_parameter("task").value
        self.model_type: str = self.get_parameter("type").value
        self.half: bool = self.get_parameter("half").value
        self.imgsz_width: int = self.get_parameter("imgsz_width").value
        self.imgsz_height: int = self.get_parameter("imgsz_height").value
        self.show: bool = self.get_parameter("show").value
        self.conf: float = self.get_parameter("conf").value
        self.tracker_timeout: int = self.get_parameter("tracker_timeout").value
        self.lookahead_steps: float = self.get_parameter(
            "prediction_lookahead_steps"
        ).value
        self.inference_mode: str = str(
            self.get_parameter("inference_mode").value
        ).lower()

        self.persist_tracking: bool = self.get_parameter("persist_tracking").value
        self.use_kalman: bool = self.get_parameter("use_kalman").value
        self.enable_color_verification: bool = self.get_parameter(
            "enable_color_verification"
        ).value
        self.target_color: str = str(self.get_parameter("target_color").value).lower()
        self.use_robot_id_targeting: bool = bool(
            self.get_parameter("use_robot_id_targeting").value
        )
        self.max_hsv_candidates: int = self.get_parameter("max_hsv_candidates").value
        self.max_coast_frames: int = self.get_parameter("max_coast_frames").value
        self.runtime_log_dir: str = str(self.get_parameter("runtime_log_dir").value)

        # --- Validate inference_mode ------------------------------------
        if self.inference_mode not in ("predict", "track"):
            raise ValueError(
                f"Unsupported inference_mode: {self.inference_mode}. "
                "Use 'predict' or 'track'."
            )

        # Item 12: warn when Kalman is used without tracked IDs
        if self.use_kalman and self.inference_mode != "track":
            self.get_logger().warning(
                "use_kalman=True with inference_mode='predict': "
                "Kalman will operate in single-target mode (track_id=0 assigned "
                "to the first valid detection each frame)."
            )

        # --- Validate model path ----------------------------------------
        if not os.path.isfile(self.model_path):
            self.get_logger().warning(
                f"Model file not found: {self.model_path}. "
                "Inference will fail at first frame."
            )

        # --- Derived state -----------------------------------------------
        self.hfov_rad: float = math.radians(self.get_parameter("hfov_degrees").value)
        self.vfov_rad: float = math.radians(self.get_parameter("vfov_degrees").value)
        self.vis_window_name: str = self.get_parameter("vis_window_name").value

        self.target_class: int = BLUE_CLASS if self.target_color != "red" else RED_CLASS
        self.active_trackers: dict[int, TrackerWrapper] = {}
        self.last_frame_time = None
        self._fps_ema = 0.0
        self._fps_t_prev = None
        self._window_created = False
        self._proc_fps_ema = 0.0
        self._stream_fps_ema = 0.0

        # Stage timing accumulators (Item 4)
        self._frame_count = 0
        self._stage_totals: dict[str, float] = {
            "decode": 0.0,
            "letterbox": 0.0,
            "inference": 0.0,
            "extract": 0.0,
            "select": 0.0,
            "publish": 0.0,
            "render": 0.0,
        }

        # Build runtime logger before heavy initialization so headless logs include startup.
        self._runtime_logger = self._create_runtime_logger()
        self._log_info("Initializing Modular YOLOv8 Node...")

        # --- Detector ----------------------------------------------------
        self.detector = YOLODetector(self.model_path, self.task, self.model_type)
        self._log_info("Detector Initialized.")
        self._log_info(
            f"inference_mode={self.inference_mode}, use_kalman={self.use_kalman}, "
            f"persist_tracking={self.persist_tracking}, target_color={self.target_color}, "
            f"use_robot_id_targeting={self.use_robot_id_targeting}, "
            f"max_hsv_candidates={self.max_hsv_candidates}, "
            f"max_coast_frames={self.max_coast_frames}"
        )

        # --- ROS transport -----------------------------------------------
        self.bridge = CvBridge()
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.image_subscription = self.create_subscription(
            Image, "image_raw", self.image_callback, image_qos
        )
        self.int_subscription = self.create_subscription(
            Int32, "robot_id", self.id_listener_callback, 10
        )
        self.bbox_publisher = self.create_publisher(Int32MultiArray, "bounding_box", 10)
        self.override_publisher = self.create_publisher(Bool, "override", 10)
        self.angle_publisher = self.create_publisher(
            Float32MultiArray, "bounding_box_angles", 10
        )
        # Item 9: OBB orientation publisher
        self.obb_angle_publisher = self.create_publisher(Float32, "obb_angle", 10)

        # Pre-allocated messages
        self.bbox_msg = Int32MultiArray()
        self.override_msg = Bool()
        self._angle_msg = Float32MultiArray()
        self._obb_angle_msg = Float32()

    def _create_runtime_logger(self) -> VisionRuntimeLogger:
        if self.show:
            return VisionRuntimeLogger(ConsoleVisionLogSink(self.get_logger()))
        log_dir = _resolve_runtime_log_dir(self.runtime_log_dir)
        sink = FileVisionLogSink(log_dir)
        self.get_logger().info(f"Headless runtime logs: {sink.path}")
        return VisionRuntimeLogger(sink)

    def _log_info(self, message: str) -> None:
        self.get_logger().info(message)
        if self.show:
            return
        self._runtime_logger.log_note("INFO", message)

    def _log_error(self, message: str) -> None:
        self.get_logger().error(message)
        if self.show:
            return
        self._runtime_logger.log_note("ERROR", message)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def id_listener_callback(self, msg: Int32):
        if not self.use_robot_id_targeting:
            return
        robot_id = msg.data
        if robot_id in RED_TEAM:
            self.target_class = BLUE_CLASS
            self._log_info(f"ID {robot_id} is RED. Targeting BLUE.")
        elif robot_id in BLUE_TEAM:
            self.target_class = RED_CLASS
            self._log_info(f"ID {robot_id} is BLUE. Targeting RED.")

    # ------------------------------------------------------------------
    # Pipeline stages  (Items 14, 16)
    # ------------------------------------------------------------------

    def decode_frame(self, msg: Image) -> np.ndarray | None:
        """Decode a ROS Image into a BGR numpy array."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if frame is None:
            self.get_logger().warn("Failed to convert image")
        return frame

    def prepare_inference_frame(self, frame: np.ndarray) -> LetterboxResult:
        """Letterbox-resize *frame* to the model's expected input size."""
        return letterbox_frame(frame, self.imgsz_width, self.imgsz_height)

    def run_inference(self, padded_frame: np.ndarray):
        """Run the detector on *padded_frame* and return raw results."""
        imgsz_val = (
            self.imgsz_width
            if self.imgsz_width == self.imgsz_height
            else (self.imgsz_height, self.imgsz_width)
        )

        if self.inference_mode == "track":
            return self.detector.track(
                padded_frame,
                half=self.half,
                imgsz=imgsz_val,
                show=False,
                conf=self.conf,
                verbose=False,
                persist=self.persist_tracking,
                tracker="bytetrack.yaml",
            )
        return self.detector.predict(
            padded_frame,
            half=self.half,
            imgsz=imgsz_val,
            show=False,
            conf=self.conf,
            verbose=False,
        )

    def extract_candidates(self, result, lb: LetterboxResult) -> list[Candidate]:
        """Extract detection candidates from a single result object.

        Converts coordinates to source-image space and performs a single
        compact CPU tensor transfer (Item 5).
        """
        candidates: list[Candidate] = []
        boxes = None

        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes
        elif hasattr(result, "obb") and result.obb is not None and self.task == "obb":
            boxes = result.obb

        if boxes is None or not hasattr(boxes, "xyxy"):
            return candidates

        # Item 5: single compact CPU transfer
        bbox_arr = _to_numpy(boxes.xyxy).astype(np.int32)
        conf_arr = _to_numpy(boxes.conf)
        cls_arr = _to_numpy(boxes.cls).astype(int)

        id_arr = None
        if (
            self.inference_mode == "track"
            and hasattr(boxes, "id")
            and boxes.id is not None
        ):
            id_arr = _to_numpy(boxes.id).astype(np.int32)

        # Item 9: extract OBB angles if available
        obb_angles = None
        if self.task == "obb" and hasattr(boxes, "xywhr") and boxes.xywhr is not None:
            xywhr = _to_numpy(boxes.xywhr)
            if xywhr.ndim == 2 and xywhr.shape[1] >= 5:
                obb_angles = xywhr[:, 4]  # angle column

        for i in range(len(bbox_arr)):
            x1, y1, x2, y2 = (
                int(bbox_arr[i][0]),
                int(bbox_arr[i][1]),
                int(bbox_arr[i][2]),
                int(bbox_arr[i][3]),
            )
            track_id = int(id_arr[i]) if id_arr is not None else None

            # Item 12: Kalman without track IDs → single-target mode
            if self.use_kalman and track_id is None:
                # Limit no-ID mode to one candidate so trackers do not alias across detections.
                if candidates:
                    continue
                track_id = 0

            # Convert to source-image space (Item 2)
            sx1, sy1, sx2, sy2 = unletterbox_coords(x1, y1, x2, y2, lb)
            scx = (sx1 + sx2) / 2.0
            scy = (sy1 + sy2) / 2.0

            angle = float(obb_angles[i]) if obb_angles is not None else None

            candidates.append(
                Candidate(
                    bbox_model=(x1, y1, x2, y2),
                    bbox_source=(sx1, sy1, sx2, sy2),
                    center_source=(scx, scy),
                    confidence=float(conf_arr[i]),
                    class_id=int(cls_arr[i]),
                    track_id=track_id,
                    is_enemy=True,  # default; overridden by colour check below
                    obb_angle=angle,
                )
            )

        # Item 6: HSV colour verification on top-N candidates only
        if self.enable_color_verification and candidates:
            # Sort by confidence (descending) so HSV runs on best candidates
            candidates.sort(key=lambda c: c.confidence, reverse=True)
            check_count = min(self.max_hsv_candidates, len(candidates))

            for c in candidates[:check_count]:
                mx1, my1, mx2, my2 = c.bbox_model
                roi = lb.frame[
                    max(0, my1) : min(my2, lb.frame.shape[0]),
                    max(0, mx1) : min(mx2, lb.frame.shape[1]),
                ]
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    c.is_enemy = False
                    continue
                c.is_enemy, _ = check_enemy(roi, self.target_class)

            # Mark unchecked candidates as non-enemy (conservative)
            for c in candidates[check_count:]:
                c.is_enemy = False

        return candidates

    def rank_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        """Rank candidates by explicit policy (Item 10, 16).

        Sort order: enemy status (True first), confidence (descending),
        distance from source-image center (ascending).
        """
        if not candidates:
            return candidates

        img_cx = self.imgsz_width / 2.0
        img_cy = self.imgsz_height / 2.0

        def _sort_key(c: Candidate):
            dist = math.hypot(c.center_source[0] - img_cx, c.center_source[1] - img_cy)
            return (not c.is_enemy, -c.confidence, dist)

        candidates.sort(key=_sort_key)
        return candidates

    def select_target(self, candidates: list[Candidate]) -> Candidate | None:
        """Select the best target from ranked candidates (Item 16)."""
        for c in candidates:
            if c.is_enemy:
                return c
        return None

    def _calculate_angles_for_request(
        self, request: AngleRequest
    ) -> tuple[float, float]:
        sx1, sy1, sx2, sy2 = request.bbox
        src_w, src_h = request.src_size
        return calculate_angles(
            sx1,
            sy1,
            sx2,
            sy2,
            src_w,
            src_h,
            self.hfov_rad,
            self.vfov_rad,
        )

    # ------------------------------------------------------------------
    # Publishing helpers  (Items 1, 16)
    # ------------------------------------------------------------------

    def publish_target(self, target: Candidate, lb: LetterboxResult):
        """Publish a valid target's bbox, angles, override, and OBB angle."""
        sx1, sy1, sx2, sy2 = target.bbox_source
        self.bbox_msg.data = [sx1, sy1, sx2, sy2]
        self.override_msg.data = True
        self.bbox_publisher.publish(self.bbox_msg)
        self.override_publisher.publish(self.override_msg)

        angle_request = AngleRequest(
            bbox=(sx1, sy1, sx2, sy2),
            src_size=(lb.src_w, lb.src_h),
        )
        yaw, pitch = self._calculate_angles_for_request(angle_request)
        self._angle_msg.data = [float(yaw), float(pitch)]
        self.angle_publisher.publish(self._angle_msg)

        if target.obb_angle is not None:
            self._obb_angle_msg.data = float(target.obb_angle)
        else:
            self._obb_angle_msg.data = 0.0
        self.obb_angle_publisher.publish(self._obb_angle_msg)

    def publish_no_target(self):
        """Publish centered zero-area bbox and zero angles (Item 1).

        Uses (src_w/2, src_h/2)—not imgsz—because downstream expects
        source-image-space coordinates after Item 2.  For the first frame
        before any letterbox is computed, we fall back to imgsz.
        """
        cx = self.imgsz_width // 2
        cy = self.imgsz_height // 2
        self.bbox_msg.data = [cx, cy, cx, cy]
        self.override_msg.data = False
        self.bbox_publisher.publish(self.bbox_msg)
        self.override_publisher.publish(self.override_msg)

        self._angle_msg.data = [0.0, 0.0]
        self.angle_publisher.publish(self._angle_msg)

        self._obb_angle_msg.data = 0.0
        self.obb_angle_publisher.publish(self._obb_angle_msg)

    def publish_kalman_target(self, request: KalmanPublishRequest):
        """Publish a Kalman-predicted target position."""
        xf, yf = request.center_model
        w, h = request.size_model
        lb = request.letterbox

        # 1. Clamp to model space.
        bx1 = max(0, min(int(xf - w / 2), self.imgsz_width - 1))
        by1 = max(0, min(int(yf - h / 2), self.imgsz_height - 1))
        bx2 = max(0, min(int(xf + w / 2), self.imgsz_width - 1))
        by2 = max(0, min(int(yf + h / 2), self.imgsz_height - 1))

        if bx1 >= bx2 or by1 >= by2:
            return False

        # 2. Convert to source-image space and publish.
        sx1, sy1, sx2, sy2 = unletterbox_coords(bx1, by1, bx2, by2, lb)
        self.bbox_msg.data = [sx1, sy1, sx2, sy2]
        self.override_msg.data = True
        self.bbox_publisher.publish(self.bbox_msg)
        self.override_publisher.publish(self.override_msg)

        angle_request = AngleRequest(
            bbox=(sx1, sy1, sx2, sy2),
            src_size=(lb.src_w, lb.src_h),
        )
        yaw, pitch = self._calculate_angles_for_request(angle_request)
        self._angle_msg.data = [float(yaw), float(pitch)]
        self.angle_publisher.publish(self._angle_msg)

        self._obb_angle_msg.data = (
            float(request.obb_angle) if request.obb_angle is not None else 0.0
        )
        self.obb_angle_publisher.publish(self._obb_angle_msg)
        return True

    # ------------------------------------------------------------------
    # Main callback  (Item 14 — split into stages)
    # ------------------------------------------------------------------

    def image_callback(self, msg: Image):
        try:
            t0 = time.monotonic()

            # 1. Decode image.
            frame = self.decode_frame(msg)
            if frame is None:
                return
            t1 = time.monotonic()

            # 2. Compute frame delta.
            current_time = self.get_clock().now()
            dt = 1.0 / 60.0
            if self.last_frame_time is not None:
                dt = (current_time - self.last_frame_time).nanoseconds / 1e9
                if dt <= 0:
                    # Guard against clock glitches; zero dt breaks velocity/acceleration math.
                    dt = 0.001
            self._stream_fps_ema = tick_fps_ema(self._stream_fps_ema, dt)
            self.last_frame_time = current_time

            # 3. Prepare model input.
            lb = self.prepare_inference_frame(frame)
            t2 = time.monotonic()

            # 4. Run inference.
            results = self.run_inference(lb.frame)
            t3 = time.monotonic()

            # 5. Extract candidates.
            published_this_frame = False
            res_obj = results[0] if results and len(results) > 0 else None

            vis_frame = None
            if self.show and res_obj is not None:
                vis_frame = lb.frame.copy()
                draw_detections(vis_frame, res_obj, self.task)

            candidates = self.extract_candidates(res_obj, lb) if res_obj else []
            t4 = time.monotonic()
            # 6. Rank and publish target.
            candidates = self.rank_candidates(candidates)
            detected_opponent = self.select_target(candidates)

            if self.use_kalman:
                published_this_frame = self._kalman_pipeline(
                    candidates, dt, lb, vis_frame
                )
            else:
                target = self.select_target(candidates)
                if target is not None:
                    self.publish_target(target, lb)
                    published_this_frame = True
            t5 = time.monotonic()

            # 7. Publish no-target fallback.
            if not published_this_frame:
                self.publish_no_target()
            t6 = time.monotonic()

            # 8. Render debug view.
            if self.show:
                if vis_frame is None:
                    vis_frame = lb.frame.copy()
                self._render_debug(vis_frame, t0)
            t7 = time.monotonic()
            self._proc_fps_ema = tick_fps_ema(self._proc_fps_ema, max(t7 - t0, 1e-9))

            self._runtime_logger.log_fps_once_per_second(t7, self._proc_fps_ema)
            if (
                self.enable_color_verification
                and detected_opponent is not None
                and detected_opponent.is_enemy
            ):
                event = VisionLogRecord(
                    timestamp=datetime.now().isoformat(timespec="milliseconds"),
                    fps=self._proc_fps_ema,
                    opponent_bbox=detected_opponent.bbox_source,
                    confidence=detected_opponent.confidence,
                    color_pass=True,
                    latency_ms=(t6 - t0) * 1000.0,
                )
                self._runtime_logger.log_opponent_event(event)

            # 9. Update stage timings.
            self._frame_count += 1
            self._stage_totals["decode"] += t1 - t0
            self._stage_totals["letterbox"] += t2 - t1
            self._stage_totals["inference"] += t3 - t2
            self._stage_totals["extract"] += t4 - t3
            self._stage_totals["select"] += t5 - t4
            self._stage_totals["publish"] += t6 - t5
            self._stage_totals["render"] += t7 - t6

            if self._frame_count % _STAGE_LOG_INTERVAL == 0:
                self._log_stage_timing()

        except Exception as e:
            self._log_error(f"Error in image_callback: {e}")
            self._log_error(traceback.format_exc())

    # ------------------------------------------------------------------
    # Kalman tracking pipeline  (Items 3, 11)
    # ------------------------------------------------------------------

    def _kalman_pipeline(
        self,
        candidates: list[Candidate],
        dt: float,
        lb: LetterboxResult,
        vis_frame: np.ndarray | None,
    ) -> bool:
        """Process candidates through the Kalman tracking pipeline.

        Returns True if a target was published this frame.
        """
        current_ids: set[int] = set()
        published = False

        # Update trackers with current detections
        for c in candidates:
            if not c.is_enemy or c.track_id is None:
                continue
            current_ids.add(c.track_id)
            mx1, my1, mx2, my2 = c.bbox_model
            cx = (mx1 + mx2) / 2.0
            cy = (my1 + my2) / 2.0
            w = float(abs(mx2 - mx1))
            h = float(abs(my2 - my1))

            if c.track_id not in self.active_trackers:
                self.active_trackers[c.track_id] = TrackerWrapper(
                    ArmorPlateTracker(cx, cy, w, h)
                )
            else:
                tracker = self.active_trackers[c.track_id].tracker
                # Item 3: predict then update — no double advance
                tracker.predict(dt)
                tracker.update(cx, cy, w, h)

            self.active_trackers[c.track_id].lost_frames = 0
            self.active_trackers[c.track_id].obb_angle = getattr(c, "obb_angle", None)

            if self.show and vis_frame is not None:
                cv2.putText(
                    vis_frame,
                    f"ID:{c.track_id}",
                    (mx1, my1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # Publish from trackers + expire stale ones
        ids_to_remove: list[int] = []
        for tid, wrapper in self.active_trackers.items():
            if tid not in current_ids:
                wrapper.lost_frames += 1

            if wrapper.lost_frames >= self.tracker_timeout:
                ids_to_remove.append(tid)
                continue

            # Item 11: publish from trackers with lost_frames <= max_coast_frames
            if not published and wrapper.lost_frames <= self.max_coast_frames:
                state = wrapper.tracker.state
                px, py = state[0], state[1]
                vx, vy = state[2], state[3]
                ax, ay = state[4], state[5]
                t_look = self.lookahead_steps * dt
                xf = px + vx * t_look + 0.5 * ax * t_look**2
                yf = py + vy * t_look + 0.5 * ay * t_look**2
                w_box = wrapper.tracker.width
                h_box = wrapper.tracker.height

                kalman_request = KalmanPublishRequest(
                    center_model=(xf, yf),
                    size_model=(w_box, h_box),
                    letterbox=lb,
                    obb_angle=getattr(wrapper, "obb_angle", None),
                )
                if self.publish_kalman_target(kalman_request):
                    published = True
                    if self.show and vis_frame is not None:
                        bx1 = max(0, int(xf - w_box / 2))
                        by1 = max(0, int(yf - h_box / 2))
                        bx2 = min(self.imgsz_width - 1, int(xf + w_box / 2))
                        by2 = min(self.imgsz_height - 1, int(yf + h_box / 2))
                        cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), (255, 0, 0), 2)

        for tid in ids_to_remove:
            del self.active_trackers[tid]

        return published

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _render_debug(self, vis_frame: np.ndarray, _t_cb_start: float):
        """Render FPS overlay and display the debug window."""
        cv2.putText(
            vis_frame,
            f"Proc FPS: {self._proc_fps_ema:.1f}",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis_frame,
            f"Stream FPS: {self._stream_fps_ema:.1f}",
            (8, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        if not self._window_created:
            cv2.namedWindow(self.vis_window_name, cv2.WINDOW_NORMAL)
            self._window_created = True
        view = preview_scale(vis_frame, PREVIEW_MAX_W, PREVIEW_MAX_H)
        cv2.imshow(self.vis_window_name, view)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Stage timing  (Item 4)
    # ------------------------------------------------------------------

    def _log_stage_timing(self):
        """Log per-stage mean timing over the last N frames."""
        n = _STAGE_LOG_INTERVAL
        parts = ", ".join(
            f"{name}={1000 * total / n:.2f}ms"
            for name, total in self._stage_totals.items()
        )
        total_ms = 1000 * sum(self._stage_totals.values()) / n
        self.get_logger().info(
            f"Stage timing (mean over {n} frames): {parts} | total={total_ms:.2f}ms"
        )
        for k in self._stage_totals:
            self._stage_totals[k] = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy_node(self):
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        self.get_logger().info("Shutting down YOLOv8 node...")
        if hasattr(self, "_runtime_logger"):
            self._runtime_logger.close()

        # 1. Destroy OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # 2. Release TensorRT engine / YOLO model to free GPU memory.
        #    Without this, repeated Ctrl+C restarts leak CUDA contexts and
        #    pinned GPU memory, eventually freezing the Jetson.
        if hasattr(self, "detector") and self.detector is not None:
            try:
                # Drop the Ultralytics model — triggers C++ TensorRT destructor
                if hasattr(self.detector, "model"):
                    del self.detector.model
                self.detector = None
            except Exception as exc:
                self.get_logger().warning(f"Detector cleanup error: {exc}")

        # 3. Flush CUDA cache so pinned/cached GPU memory is truly returned
        #    to the OS before the next process tries to allocate.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _shutdown(node, executor=None):
    """Shared shutdown sequence — called from signals, atexit, and finally."""
    if node is None:
        return
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        if executor is not None:
            executor.shutdown()
    except Exception:
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass


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

    # Register atexit so the GPU is released even if Python exits
    # without hitting our finally block (e.g. os._exit, fatal signal).
    atexit.register(lambda: _shutdown(node))

    # Install signal handlers so Ctrl+C / kill always triggers a clean
    # shutdown — releases TensorRT engine and CUDA memory instead of
    # relying on rclpy's internal signal handling.
    def _signal_handler(signum, _frame):
        sig_name = signal.Signals(signum).name
        print(f"\n[yolov8_node] Received {sig_name}, shutting down cleanly...")
        _shutdown(node)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as exc:
        node.get_logger().error(f"Unexpected error: {exc}")
    finally:
        _shutdown(node)


if __name__ == "__main__":
    main()
