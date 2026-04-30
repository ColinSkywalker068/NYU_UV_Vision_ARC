import argparse
import os
import platform
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional
from collections.abc import Iterator

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

DEFAULT_MODEL_PATH = Path("model_test/best.mlpackage")

_HAS_CUDA = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
_IS_MAC = platform.system() == "Darwin"

COREML_IMGSZ = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO + BoT-SORT — CoreML INT8")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", type=str, default="obb")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=COREML_IMGSZ,
        help=f"must match export imgsz (default: {COREML_IMGSZ})",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--fps-window", type=int, default=30)
    parser.add_argument(
        "--mps-memory",
        type=float,
        default=0.0,
        help="MPS per-process memory fraction (0.0 = unlimited)",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="Webcam exposure value. Lower is usually darker if the camera supports it.",
    )
    return parser.parse_args()


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


def configure_camera_exposure(source: int | str, exposure: Optional[float]) -> None:
    if exposure is None or not isinstance(source, int):
        return
    backend = cv2.CAP_AVFOUNDATION if _IS_MAC else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        print(f"[warn] could not open camera {source} to set exposure")
        return
    try:
        capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        applied_auto_exposure = capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
        applied_exposure = capture.get(cv2.CAP_PROP_EXPOSURE)
        print(
            "[info] requested exposure "
            f"{exposure}; camera reports auto_exposure={applied_auto_exposure} "
            f"exposure={applied_exposure}"
        )
    finally:
        capture.release()


def warmup(model: YOLO, device: str, imgsz: int) -> None:
    """Trigger ANE shader compile before the tracking loop to hide the first-frame spike."""
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy, device=device, imgsz=imgsz, verbose=False)


def display_frame(frame: np.ndarray) -> np.ndarray:
    """GPU round-trip for accelerated post-processing. No-op on non-CUDA builds."""
    if not _HAS_CUDA:
        return frame
    gpu = cv2.cuda.GpuMat()
    gpu.upload(frame)
    # add cv2.cuda ops here
    return gpu.download()


class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._timestamps.append(time.perf_counter())
        if len(self._timestamps) < 2:
            return 0.0
        return (len(self._timestamps) - 1) / (
            self._timestamps[-1] - self._timestamps[0]
        )

    def draw(self, frame: np.ndarray) -> np.ndarray:
        label = f"FPS: {self.tick():.1f}"
        cv2.putText(
            frame,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (50, 220, 80),
            2,
            cv2.LINE_AA,
        )
        return frame


class InferenceThread(threading.Thread):
    def __init__(self, model: YOLO, track_args: dict) -> None:
        super().__init__(daemon=True)
        self._model = model
        self._track_args = track_args
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)

    def run(self) -> None:
        for result in self._model.track(**self._track_args):
            frame = result.plot(img=None)
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                pass  # display busy — drop frame, never block inference
        self._q.put(None)  # signal done

    def next_frame(self) -> np.ndarray | None | type(...):
        """
        Returns:
          np.ndarray  — new frame ready
          None        — inference finished
          ...         — still running, nothing ready yet
        """
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return ...


def run_tracking(args: argparse.Namespace) -> None:
    configure_mps(args.mps_memory)

    model_path = resolve_model_path(args.model)
    warn_imgsz_mismatch(args, model_path)
    source = resolve_source(args.source)
    configure_camera_exposure(source, args.exposure)

    device = resolve_device(args.device, model_path)
    model = YOLO(str(model_path), task=args.task)

    print(f"[info] warming up ANE/CoreML ({args.imgsz}px dummy forward)...")
    warmup(model, device, args.imgsz)
    print("[info] warm-up done")

    fps_counter = FPSCounter(window=args.fps_window)

    track_args = dict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        tracker="botsort.yaml",
        persist=True,
        stream=True,
        verbose=False,
        show=False,
        save=args.save,
        device=device,
        half=False,
    )

    inference = InferenceThread(model, track_args)
    inference.start()

    cv2.namedWindow("YOLO BoT-SORT", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO BoT-SORT", 1280, 720)

    while True:
        frame = inference.next_frame()

        if frame is None:
            break  # inference finished
        if frame is ...:
            cv2.waitKey(1)  # nothing ready — yield to Cocoa event loop
            continue

        frame = display_frame(frame)
        frame = fps_counter.draw(frame)
        cv2.imshow("YOLO BoT-SORT", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    inference.join()


if __name__ == "__main__":
    run_tracking(parse_args())
