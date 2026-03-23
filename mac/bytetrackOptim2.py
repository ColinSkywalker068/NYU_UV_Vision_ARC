import argparse
import os
import platform
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.engine.results import Results

# make yolov8_ros importable regardless of invocation directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolov8_ros.yolov8_ros.tracker import ArmorPlateTracker, TrackerWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = Path("model_test/best.mlpackage")

_HAS_CUDA = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
_IS_MAC = platform.system() == "Darwin"

COREML_IMGSZ = 640


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO detect + custom Kalman tracker")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", type=str, default="obb")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=COREML_IMGSZ,
                        help=f"must match export imgsz (default: {COREML_IMGSZ})")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--fps-window", type=int, default=30)
    parser.add_argument("--mps-memory", type=float, default=0.0,
                        help="MPS per-process memory fraction (0.0 = unlimited)")
    parser.add_argument("--tracker-timeout", type=int, default=15,
                        help="frames before a lost track is deleted")
    parser.add_argument("--prediction-lookahead", type=float, default=0.5,
                        help="lookahead multiplier on dt for future position")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="min IoU to match a detection to an existing track")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def configure_mps(memory_fraction: float) -> None:
    """Must be called before the model is loaded."""
    if not _IS_MAC:
        return
    if memory_fraction > 0.0:
        try:
            import torch
            torch.mps.set_per_process_memory_fraction(memory_fraction)
        except (ImportError, AttributeError):
            pass
    else:
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def resolve_model_path(model_path: Path) -> Path:
    resolved = model_path.expanduser().resolve()
    if not (resolved.is_file() or resolved.is_dir()):
        raise FileNotFoundError(f"Model not found: {resolved}")
    return resolved


def resolve_source(source: str) -> int | str:
    try:
        return int(source)
    except ValueError:
        return source


def resolve_device(device: Optional[str], model_path: Path) -> str:
    if device is not None:
        return device
    if model_path.suffix == ".engine":
        return "0"
    if _IS_MAC:
        return "mps"
    return "cpu"


def warn_imgsz_mismatch(args: argparse.Namespace, model_path: Path) -> None:
    if model_path.suffix != ".mlpackage":
        return
    if args.imgsz != COREML_IMGSZ:
        print(
            f"[warn] --imgsz {args.imgsz} differs from compiled COREML_IMGSZ={COREML_IMGSZ}. "
            "This forces a CPU resize and may break ANE routing. "
            "Update COREML_IMGSZ at the top of the file if you re-exported."
        )


def warmup(model: YOLO, device: str, imgsz: int) -> None:
    """Trigger ANE shader compile before the detection loop."""
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy, device=device, imgsz=imgsz, verbose=False)


# ---------------------------------------------------------------------------
# IoU + Hungarian matching
# ---------------------------------------------------------------------------

def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def match_detections(
    predicted: list[tuple[int, tuple[int, int, int, int]]],
    detections: list[tuple[int, int, int, int]],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Hungarian IoU matching.
    Returns (matched, unmatched_track_indices, unmatched_det_indices).
    """
    if not predicted or not detections:
        return [], list(range(len(predicted))), list(range(len(detections)))

    cost = np.array(
        [[1.0 - _iou(pbbox, dbbox) for dbbox in detections] for _, pbbox in predicted],
        dtype=np.float32,
    )
    row_ind, col_ind = linear_sum_assignment(cost)

    matched, matched_t, matched_d = [], set(), set()
    for ti, di in zip(row_ind, col_ind):
        if cost[ti, di] <= 1.0 - iou_threshold:
            matched.append((ti, di))
            matched_t.add(ti)
            matched_d.add(di)

    unmatched_tracks = [i for i in range(len(predicted)) if i not in matched_t]
    unmatched_dets = [i for i in range(len(detections)) if i not in matched_d]
    return matched, unmatched_tracks, unmatched_dets


# ---------------------------------------------------------------------------
# Bounding box helpers
# ---------------------------------------------------------------------------

def clamp_bbox(
    bbox: tuple[int, int, int, int], w: int, h: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(x1, w - 1)),
        max(0, min(y1, h - 1)),
        max(0, min(x2, w - 1)),
        max(0, min(y2, h - 1)),
    )


def extract_detections(result: Results, task: str) -> list[tuple[int, int, int, int]]:
    boxes = (
        result.obb
        if task == "obb" and getattr(result, "obb", None) is not None
        else result.boxes
    )
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return []
    return [tuple(b) for b in boxes.xyxy.cpu().int().tolist()]


# ---------------------------------------------------------------------------
# Kalman tracker manager
# ---------------------------------------------------------------------------

class KalmanTracker:
    """
    Predict → match (Hungarian/IoU) → update/spawn/reap.
    No dependency on Ultralytics tracker internals.
    """

    def __init__(
        self,
        tracker_timeout: int,
        prediction_lookahead: float,
        iou_threshold: float,
    ) -> None:
        self._tracks: dict[int, TrackerWrapper] = {}
        self._next_id = 0
        self._tracker_timeout = tracker_timeout
        self._lookahead = prediction_lookahead
        self._iou_threshold = iou_threshold
        self._last_ts: Optional[float] = None

    def _dt(self) -> float:
        now = time.perf_counter()
        if self._last_ts is None:
            self._last_ts = now
            return 1.0 / 60.0
        dt = now - self._last_ts
        self._last_ts = now
        return max(dt, 0.001)

    def _predict_bbox(
        self, wrapper: TrackerWrapper, dt: float, fw: int, fh: int
    ) -> tuple[int, int, int, int]:
        px, py = wrapper.tracker.predict(dt)
        s = wrapper.tracker.state
        vx, vy, ax, ay = s[2], s[3], s[4], s[5]
        la = self._lookahead * dt
        fx = px + vx * la + 0.5 * ax * la ** 2
        fy = py + vy * la + 0.5 * ay * la ** 2
        w, h = wrapper.tracker.width, wrapper.tracker.height
        return clamp_bbox(
            (int(fx - w / 2), int(fy - h / 2), int(fx + w / 2), int(fy + h / 2)),
            fw, fh,
        )

    def update(
        self,
        detections: list[tuple[int, int, int, int]],
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[int, tuple[int, int, int, int], bool]]:
        dt = self._dt()
        track_ids = list(self._tracks.keys())

        # 1. predict
        predicted = [
            (tid, self._predict_bbox(self._tracks[tid], dt, frame_w, frame_h))
            for tid in track_ids
        ]

        # 2. match
        matched, unmatched_t, unmatched_d = match_detections(
            predicted, detections, self._iou_threshold
        )

        # 3. update matched tracks
        for ti, di in matched:
            tid = track_ids[ti]
            x1, y1, x2, y2 = detections[di]
            self._tracks[tid].tracker.update(
                (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                float(abs(x2 - x1)), float(abs(y2 - y1)),
            )
            self._tracks[tid].lost_frames = 0

        # 4. age unmatched tracks
        for ti in unmatched_t:
            self._tracks[track_ids[ti]].lost_frames += 1

        # 5. spawn new tracks for unmatched detections
        for di in unmatched_d:
            x1, y1, x2, y2 = detections[di]
            self._tracks[self._next_id] = TrackerWrapper(
                ArmorPlateTracker(
                    (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                    float(abs(x2 - x1)), float(abs(y2 - y1)),
                )
            )
            self._next_id += 1

        # 6. reap stale tracks
        for tid in [t for t, w in self._tracks.items() if w.lost_frames >= self._tracker_timeout]:
            del self._tracks[tid]

        # 7. output: predicted bbox per surviving track
        pred_map = dict(predicted)
        return [
            (tid, pred_map[tid], w.lost_frames == 0)
            for tid, w in self._tracks.items()
            if tid in pred_map and pred_map[tid][0] < pred_map[tid][2] and pred_map[tid][1] < pred_map[tid][3]
        ]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_tracks(
    frame: np.ndarray,
    tracks: list[tuple[int, tuple[int, int, int, int], bool]],
) -> np.ndarray:
    for tid, (x1, y1, x2, y2), is_live in tracks:
        color = (255, 80, 80) if is_live else (0, 165, 255)
        label = f"KF {tid}" if is_live else f"LOST {tid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def display_frame(frame: np.ndarray) -> np.ndarray:
    if not _HAS_CUDA:
        return frame
    gpu = cv2.cuda.GpuMat()
    gpu.upload(frame)
    return gpu.download()


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._timestamps.append(time.perf_counter())
        if len(self._timestamps) < 2:
            return 0.0
        return (len(self._timestamps) - 1) / (self._timestamps[-1] - self._timestamps[0])

    def draw(self, frame: np.ndarray) -> np.ndarray:
        label = f"FPS: {self.tick():.1f}"
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 220, 80), 2, cv2.LINE_AA)
        return frame


# ---------------------------------------------------------------------------
# Inference thread — predict only, no built-in tracker
# ---------------------------------------------------------------------------

class InferenceThread(threading.Thread):
    def __init__(self, model: YOLO, predict_args: dict) -> None:
        super().__init__(daemon=True)
        self._model = model
        self._predict_args = predict_args
        self._q: queue.Queue[Results | None] = queue.Queue(maxsize=1)

    def run(self) -> None:
        for result in self._model.predict(**self._predict_args):
            try:
                self._q.put_nowait(result)
            except queue.Full:
                pass
        self._q.put(None)

    def next_result(self) -> Results | None | type(...):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return ...


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tracking(args: argparse.Namespace) -> None:
    configure_mps(args.mps_memory)

    model_path = resolve_model_path(args.model)
    warn_imgsz_mismatch(args, model_path)
    source = resolve_source(args.source)
    device = resolve_device(args.device, model_path)

    model = YOLO(str(model_path), task=args.task)

    print(f"[info] warming up ANE/CoreML ({args.imgsz}px dummy forward)...")
    warmup(model, device, args.imgsz)
    print("[info] warm-up done")

    fps_counter = FPSCounter(window=args.fps_window)
    kf_tracker = KalmanTracker(
        tracker_timeout=args.tracker_timeout,
        prediction_lookahead=args.prediction_lookahead,
        iou_threshold=args.iou_threshold,
    )

    predict_args = dict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        stream=True,
        verbose=False,
        show=False,
        save=args.save,
        device=device,
        half=False,
    )

    inference = InferenceThread(model, predict_args)
    inference.start()

    # cv2 GUI must stay on the main thread — macOS Cocoa requirement
    cv2.namedWindow("YOLO + KF", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO + KF", 1280, 720)

    while True:
        result = inference.next_result()

        if result is None:
            break
        if result is ...:
            cv2.waitKey(1)   # yield to Cocoa event loop
            continue

        frame_h, frame_w = result.orig_shape
        detections = extract_detections(result, args.task)
        tracks = kf_tracker.update(detections, frame_w, frame_h)

        frame = result.orig_img.copy()   # raw frame, no YOLO overlays
        frame = draw_tracks(frame, tracks)
        frame = display_frame(frame)
        frame = fps_counter.draw(frame)
        cv2.imshow("YOLO + KF", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    inference.join()


if __name__ == "__main__":
    run_tracking(parse_args())
