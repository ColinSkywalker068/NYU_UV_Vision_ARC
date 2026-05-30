#!/usr/bin/env python3
"""
Basler (pypylon) + YOLO + Kalman tracker — same pipeline as bytetrackOptim2, camera via PyPylon.

Run from repo root or legacy/prototypes/mac/:
  cd legacy/prototypes/mac && python baslertrack.py --model ../../models/best.mlpackage --task obb
  # Second Basler only: --camera-index 1
"""
from __future__ import annotations

import argparse
import platform
import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from pypylon import pylon
from ultralytics import YOLO
from ultralytics.engine.results import Results

# yolov8_ros (tracker imports inside bytetrackOptim2)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bytetrackOptim2 import (
    COREML_IMGSZ,
    FPSCounter,
    KalmanTracker,
    configure_mps,
    display_frame,
    draw_tracks,
    extract_detections,
    resolve_device,
    resolve_model_path,
    warn_imgsz_mismatch,
    warmup,
)

_CAMERA_SCRIPTS = (
    Path(__file__).resolve().parents[1] / "yolov8_ros" / "scripts" / "Camera"
)
sys.path.insert(0, str(_CAMERA_SCRIPTS))
from basler_test import (  # noqa: E402
    DEFAULT_EXPOSURE_US,
    DEFAULT_FPS,
    DEFAULT_GAIN_DB,
    build_converter,
    configure_camera,
    get_actual_fps,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best.mlpackage"
_IS_MAC = platform.system() == "Darwin"
_LOADING_WINDOW = "loading..."
_MAC_GUI_PUMP_MS = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basler camera + YOLO + custom Kalman tracker (bytetrackOptim2-style)"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index in pypylon EnumerateDevices() (0 = first camera; use 1 only if two devices are listed)",
    )
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
    parser.add_argument("--fps-window", type=int, default=30)
    parser.add_argument(
        "--mps-memory",
        type=float,
        default=0.0,
        help="MPS per-process memory fraction (0.0 = unlimited)",
    )
    parser.add_argument("--tracker-timeout", type=int, default=15)
    parser.add_argument("--prediction-lookahead", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument(
        "--img-width",
        type=int,
        default=640,
        help="Ignored when using full sensor; passed to configure_camera",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=480,
        help="Ignored when using full sensor; passed to configure_camera",
    )
    parser.add_argument(
        "--use-full-sensor",
        action="store_true",
        default=True,
        help="Use full sensor resolution (default)",
    )
    parser.add_argument(
        "--no-use-full-sensor",
        action="store_false",
        dest="use_full_sensor",
    )
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=DEFAULT_EXPOSURE_US,
    )
    parser.add_argument("--binning", type=int, default=1)
    parser.add_argument("--decimation", type=int, default=1)
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip dummy CoreML forward (faster start; first live frames pay compile cost)",
    )
    return parser.parse_args()


class BaslerInferenceThread(threading.Thread):
    """Grab BGR frames from Basler, run YOLO per frame, queue latest Results (size 1)."""

    def __init__(
        self,
        model: YOLO,
        camera: pylon.InstantCamera,
        device: str,
        conf: float,
        imgsz: int,
    ) -> None:
        super().__init__(daemon=True)
        self._model = model
        self._camera = camera
        self._device = device
        self._conf = conf
        self._imgsz = imgsz
        self._q: queue.Queue[tuple[Results, np.ndarray] | None] = queue.Queue(maxsize=1)

    def run(self) -> None:
        converter = build_converter()
        while self._camera.IsGrabbing():
            grab = self._camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
            if grab is None:
                continue
            if not grab.GrabSucceeded():
                grab.Release()
                continue
            image = converter.Convert(grab)
            frame = image.GetArray()
            grab.Release()
            if frame is None or frame.size == 0:
                continue
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            try:
                results = self._model(
                    frame,
                    device=self._device,
                    conf=self._conf,
                    imgsz=self._imgsz,
                    verbose=False,
                    half=False,
                )
            except Exception as e:
                print(f"[inference error] {type(e).__name__}: {e}", flush=True)
                continue
            if not results:
                continue
            try:
                self._q.put_nowait((results[0], frame))
            except queue.Full:
                pass
        self._q.put(None)

    def next_result(self) -> tuple[Results, np.ndarray] | None | type(...):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return ...


def _run_coreml_warmup(model: YOLO, device: str, imgsz: int, skip: bool) -> None:
    if skip:
        print(
            "[info] skipping warm-up (--skip-warmup); first frames may stall while Metal/CoreML compiles.",
            flush=True,
        )
        return
    print(
        "[info] CoreML first run can take 30s–2min while Apple Neural Engine / Metal compiles; "
        "lines like [mutex.cc ... Lock blocking] are normal during that compile.",
        flush=True,
    )
    print(f"[info] warming up ({imgsz}px dummy forward)...", flush=True)
    warmup(model, device, imgsz)
    print("[info] warm-up done", flush=True)


def _load_model_threaded(model_path: Path, task: str) -> YOLO:
    """Load YOLO on a worker thread; main thread must pump cv2 (macOS WindowServer watchdog)."""
    holder: list[YOLO | None] = [None]
    err: list[BaseException | None] = [None]

    def _load() -> None:
        try:
            holder[0] = YOLO(str(model_path), task=task)
        except BaseException as e:
            err[0] = e

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    print(
        "[info] Loading model (main thread pumps GUI for macOS watchdog)...", flush=True
    )
    while t.is_alive():
        cv2.waitKey(_MAC_GUI_PUMP_MS)
    t.join()
    if err[0] is not None:
        raise err[0]
    model = holder[0]
    if model is None:
        raise RuntimeError("YOLO load returned None")
    return model


def _warmup_threaded_with_gui_pump(model: YOLO, device: str, imgsz: int) -> None:
    err: list[BaseException | None] = [None]

    def _wu() -> None:
        try:
            warmup(model, device, imgsz)
        except BaseException as e:
            err[0] = e

    print(
        "[info] CoreML first run can take 30s–2min; main thread pumps GUI (macOS)...",
        flush=True,
    )
    print(f"[info] warming up ({imgsz}px dummy forward)...", flush=True)
    wu_t = threading.Thread(target=_wu, daemon=True)
    wu_t.start()
    while wu_t.is_alive():
        cv2.waitKey(_MAC_GUI_PUMP_MS)
    wu_t.join()
    if err[0] is not None:
        raise err[0]
    print("[info] warm-up done", flush=True)


def _disable_frame_rate_limit(camera: pylon.InstantCamera) -> None:
    try:
        node = camera.AcquisitionFrameRateEnable
        if node.IsWritable():
            node.SetValue(False)
    except Exception:
        pass


def run_tracking(args: argparse.Namespace) -> None:
    configure_mps(args.mps_memory)

    model_path = resolve_model_path(args.model)
    warn_imgsz_mismatch(args, model_path)
    device = resolve_device(args.device, model_path)

    # CoreML/Metal before any pypylon. On macOS, load/warmup must not block the main thread
    # without pumping the OpenCV/GUI runloop or WindowServer may SIGTERM the process.
    if _IS_MAC:
        cv2.namedWindow(_LOADING_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_LOADING_WINDOW, 400, 100)
        try:
            model = _load_model_threaded(model_path, args.task)
            if args.skip_warmup:
                print(
                    "[info] skipping warm-up (--skip-warmup); first live frames may stall while Metal compiles.",
                    flush=True,
                )
            else:
                _warmup_threaded_with_gui_pump(model, device, args.imgsz)
        finally:
            try:
                cv2.destroyWindow(_LOADING_WINDOW)
            except Exception:
                pass
    else:
        model = YOLO(str(model_path), task=args.task)
        _run_coreml_warmup(model, device, args.imgsz, args.skip_warmup)

    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera found (pypylon EnumerateDevices empty)")
    if args.camera_index < 0 or args.camera_index >= len(devices):
        raise IndexError(
            f"camera-index {args.camera_index} invalid: found {len(devices)} device(s); "
            f"use --camera-index in 0..{len(devices) - 1} (single camera => 0)"
        )

    fps_counter = FPSCounter(window=args.fps_window)
    kf_tracker = KalmanTracker(
        tracker_timeout=args.tracker_timeout,
        prediction_lookahead=args.prediction_lookahead,
        iou_threshold=args.iou_threshold,
    )

    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[args.camera_index]))
    inference: BaslerInferenceThread | None = None
    camera.Open()
    try:
        _disable_frame_rate_limit(camera)
        applied: dict[str, str | float] = {}
        settings = configure_camera(
            camera,
            width=args.img_width,
            height=args.img_height,
            fps=DEFAULT_FPS,
            use_full_sensor=args.use_full_sensor,
            auto_exposure=args.auto_exposure,
            exposure_us=args.exposure_us,
            auto_gain=False,
            gain_db=DEFAULT_GAIN_DB,
            centered_roi=True,
            binning=max(1, int(args.binning)),
            decimation=max(1, int(args.decimation)),
            applied_settings=applied,
            start_grabbing=False,
        )
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        fps_now = get_actual_fps(camera)
        print(
            f"[info] Basler[{args.camera_index}]: {int(settings['width'])}x{int(settings['height'])} "
            f"@ {fps_now:.2f} FPS (resulting)"
        )

        inference = BaslerInferenceThread(model, camera, device, args.conf, args.imgsz)
        inference.start()

        win = "YOLO + KF (Basler)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)

        while True:
            result = inference.next_result()

            if result is None:
                break
            if result is ...:
                if _IS_MAC:
                    cv2.waitKey(1)
                if not inference.is_alive():
                    print("[main] inference thread died!", flush=True)
                    break
                continue

            yolo_result, frame_bgr = result
            frame_h, frame_w = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
            detections = extract_detections(yolo_result, args.task)
            tracks = kf_tracker.update(detections, frame_w, frame_h)

            frame = frame_bgr.copy()
            frame = draw_tracks(frame, tracks)
            frame = display_frame(frame)
            frame = fps_counter.draw(frame)
            cv2.imshow(win, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if inference is not None:
            inference.join()
        if camera.IsOpen():
            camera.Close()


if __name__ == "__main__":
    run_tracking(parse_args())
