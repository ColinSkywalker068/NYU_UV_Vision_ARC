#!/usr/bin/env python3
"""Basler or RealSense + YOLO OBB on Jetson / NVIDIA GPU using PyTorch weights only (no CoreML)."""

import argparse
import os
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
from pypylon import pylon
from ultralytics import YOLO

WINDOW_BASLER = "YOLO OBB — Basler"
WINDOW_REALSENSE = "YOLO OBB — RealSense"
MODEL_PATH = Path("/home/ultraviolet/CV_Detection/model_test/best.pt")
PREVIEW_MAX_WIDTH = 960
PREVIEW_MAX_HEIGHT = 540
HEADLESS_FPS_LOG_SEC = 1.0
BASELER_QUEUE_GET_TIMEOUT_SEC = 1.0

# FIX: Precomputed HSV→BGR color palette — eliminates per-detection cv2.cvtColor call in hot loop.
_COLOR_PALETTE: list[tuple[int, int, int]] = [
    tuple(
        int(c)
        for c in cv2.cvtColor(  # type: ignore[misc]
            np.uint8([[[(40 * i) % 180, 220, 255]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
    )
    for i in range(32)
]

# Warmup frames to skip before resetting FPS baseline in headless mode.
_WARMUP_FRAMES = 5


class LatestFrameQueue:
    """Single-slot buffer: consumer always sees the most recent frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._event = threading.Event()

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
        self._event.set()

    def get(self, timeout: float) -> Optional[np.ndarray]:
        if not self._event.wait(timeout):
            return None
        # FIX: clear the event inside the lock so the flag and the stored frame
        # stay consistent — previously a second put() between wait() returning
        # and clear() could cause the next get() to block for a full timeout
        # even though a fresh frame was already available.
        with self._lock:
            frame = self._frame
            self._event.clear()
        return frame


def basler_grab_thread_fn(
    camera: pylon.InstantCamera,
    converter: Optional[pylon.ImageFormatConverter],
    uses_native_bgr8: bool,
    fq: LatestFrameQueue,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        try:
            grab = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
        except pylon.TimeoutException:
            continue
        except Exception:
            if stop.is_set():
                break
            raise
        if not grab.GrabSucceeded():
            grab.Release()
            continue
        if uses_native_bgr8:
            raw = grab.GetArray()
        else:
            assert converter is not None
            raw = converter.Convert(grab).GetArray()
        payload = raw.copy()
        grab.Release()
        fq.put(payload)


def diag_timing_enabled() -> bool:
    return os.environ.get("OBB_DIAG_TIMING", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


PLATE_BIAS_ADD = int(os.environ.get("PLATE_BIAS_ADD", "8"))
PLATE_BIAS_SUB = int(os.environ.get("PLATE_BIAS_SUB", "6"))


def parse_plate_target() -> str:
    raw = os.environ.get("PLATE_TARGET", "").strip().lower()
    if raw in ("", "none", "off", "0"):
        return ""
    if raw in ("red", "blue"):
        return raw
    return ""


def apply_plate_hue_bias_bgr(frame: np.ndarray, target: str) -> None:
    if target == "blue":
        frame[:, :, 2] = cv2.add(frame[:, :, 2], PLATE_BIAS_ADD)
        frame[:, :, 0] = cv2.subtract(frame[:, :, 0], PLATE_BIAS_SUB)
    elif target == "red":
        frame[:, :, 0] = cv2.add(frame[:, :, 0], PLATE_BIAS_ADD)
        frame[:, :, 2] = cv2.subtract(frame[:, :, 2], PLATE_BIAS_SUB)


def normalize_torch_device(spec: str) -> str:
    """YOLO-style '0' -> 'cuda:0' for torch.Tensor.to()."""
    s = spec.strip()
    low = s.lower()
    if not s or low == "cpu":
        return "cpu"
    if low == "mps":
        return "mps"
    if low.startswith("cuda"):
        return s
    if s.isdigit():
        return f"cuda:{s}"
    return s


def resolve_device() -> str:
    override = os.environ.get("YOLO_DEVICE", "").strip()
    if override:
        return normalize_torch_device(override)
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path), task="obb")


def open_default_camera(
    cap_fps: Optional[float],
) -> tuple[pylon.InstantCamera, bool]:
    tl = pylon.TlFactory.GetInstance()
    devices = tl.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera detected.")
    camera = pylon.InstantCamera(tl.CreateDevice(devices[0]))
    camera.Open()
    native_bgr8 = False
    try:
        camera.PixelFormat.SetValue("BGR8")
        native_bgr8 = True
    except Exception:
        pass
    try:
        if cap_fps is not None and cap_fps > 0:
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(float(cap_fps))
        else:
            camera.AcquisitionFrameRateEnable.SetValue(False)
    except Exception:
        pass
    return camera, native_bgr8


def _realsense_best_bgr8_stream(
    rs: Any, width: int, height: int
) -> Optional[tuple[int, int, int]]:
    devs = list(rs.context().query_devices())
    if not devs:
        return None
    best: Optional[tuple[int, int, int]] = None
    best_fps = -1
    for sensor in devs[0].query_sensors():
        if not sensor.is_color_sensor():
            continue
        for p in sensor.get_stream_profiles():
            vp = p.as_video_stream_profile()
            if vp.format() != rs.format.bgr8:
                continue
            if vp.width() != width or vp.height() != height:
                continue
            f = int(vp.fps())
            if f > best_fps:
                best_fps = f
                best = (vp.width(), vp.height(), f)
    return best


def realsense_color_frames(
    cap_fps: Optional[float],
) -> Generator[np.ndarray, None, None]:
    import pyrealsense2 as rs  # type: ignore[import-untyped]

    pipeline = rs.pipeline()
    config = rs.config()
    if cap_fps is not None and cap_fps > 0:
        fps_i = max(1, int(round(cap_fps)))
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps_i)
    else:
        triplet = _realsense_best_bgr8_stream(rs, 640, 480)
        if triplet is not None:
            w_i, h_i, fps_i = triplet
            config.enable_stream(rs.stream.color, w_i, h_i, rs.format.bgr8, fps_i)
        else:
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            yield np.asanyarray(color.get_data())
    finally:
        pipeline.stop()


def create_frame_converter() -> pylon.ImageFormatConverter:
    conv = pylon.ImageFormatConverter()
    conv.OutputPixelFormat = pylon.PixelType_BGR8packed
    conv.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
    return conv


def frame_for_preview(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return frame
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def tick_fps_ema(prev_ema: float, dt: float) -> float:
    inst = 1.0 / max(dt, 1e-9)
    if prev_ema <= 0.0:
        return inst
    return 0.9 * prev_ema + 0.1 * inst


def resolve_inference_imgsz(model: YOLO) -> int:
    raw = os.environ.get("YOLO_IMGSZ", "").strip()
    if raw:
        return max(32, int(raw))
    overrides = getattr(model, "overrides", None)
    if isinstance(overrides, dict) and "imgsz" in overrides:
        return int(overrides["imgsz"])
    return 640


class ModelInputBuffer:
    """Reusable GPU tensors for BGR uint8 → NCHW RGB model input (reduces Jetson UMA alloc churn)."""

    def __init__(self, device: str, use_half: bool, imgsz: int) -> None:
        self.device = device
        self.imgsz = imgsz
        self.dtype = torch.float16 if use_half else torch.float32
        self._use_half = use_half
        self._stage: Optional[torch.Tensor] = None
        self._float_hwc: Optional[torch.Tensor] = None
        self._out: Optional[torch.Tensor] = None

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        h, w, c = frame.shape
        if c != 3:
            raise ValueError(f"Expected 3-channel BGR, got shape {frame.shape}")
        if (
            self._stage is None
            or int(self._stage.shape[0]) != h
            or int(self._stage.shape[1]) != w
        ):
            self._stage = torch.empty((h, w, c), dtype=torch.uint8, device=self.device)
            self._float_hwc = torch.empty(
                (h, w, c), dtype=torch.float32, device=self.device
            )
            self._out = torch.empty(
                (1, 3, self.imgsz, self.imgsz),
                dtype=self.dtype,
                device=self.device,
            )
        frame_c = frame if frame.flags.c_contiguous else np.ascontiguousarray(frame)
        self._stage.copy_(torch.from_numpy(frame_c))
        self._float_hwc.copy_(self._stage)
        self._float_hwc.mul_(1.0 / 255.0)
        t = self._float_hwc.permute(2, 0, 1).unsqueeze(0)
        t = t[:, [2, 1, 0], :, :]
        if t.shape[2] != self.imgsz or t.shape[3] != self.imgsz:
            t = torch.nn.functional.interpolate(
                t,
                size=(self.imgsz, self.imgsz),
                mode="bilinear",
                align_corners=False,
            )
        if self._use_half:
            self._out.copy_(t.to(dtype=torch.float16))
        else:
            self._out.copy_(t)
        return self._out


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def class_label_from_names(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def draw_obb_on_bgr(frame_bgr: np.ndarray, result: Any) -> None:
    obb = getattr(result, "obb", None)
    if obb is None:
        return
    xyxyxyxy = getattr(obb, "xyxyxyxy", None)
    if xyxyxyxy is None or len(xyxyxyxy) == 0:
        return

    # FIX: Batch all three D2H transfers into a single sync point before the
    # loop.  Previously each _to_numpy() on a CUDA tensor triggered its own
    # full stream synchronization; with N detections that was 3 N stalls.
    # Now it is exactly 3 transfers total regardless of detection count.
    polys = _to_numpy(xyxyxyxy).astype(np.int32)
    confs = _to_numpy(obb.conf)
    clss = _to_numpy(obb.cls).astype(int)

    names = getattr(result, "names", None)
    for i, poly in enumerate(polys):
        pts = poly.reshape(-1, 1, 2)
        cid = int(clss[i])
        # FIX: Use precomputed palette — eliminates cv2.cvtColor + np.uint8
        # allocation on every iteration of this hot loop.
        color = _COLOR_PALETTE[cid % len(_COLOR_PALETTE)]
        cv2.polylines(frame_bgr, [pts], isClosed=True, color=color, thickness=2)
        xi = int(poly[:, 0].min())
        yi = max(16, int(poly[:, 1].min()) - 4)
        label = f"{class_label_from_names(names, cid)} {float(confs[i]):.2f}"
        cv2.putText(
            frame_bgr,
            label,
            (xi, yi),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def run_inference_on_bgr_frames(
    model: YOLO,
    window_name: str,
    frame_source: Callable[[], Optional[np.ndarray]],
    device: str,
    use_half: bool,
    plate_target: str,
    headless: bool,
) -> None:
    imgsz = resolve_inference_imgsz(model)
    model_input = ModelInputBuffer(device, use_half, imgsz)
    fps_ema = 0.0
    fps_log_prev = time.monotonic()
    t_start = fps_log_prev
    frame_count = 0
    diag = diag_timing_enabled()

    # FIX: CUDA stream for async inference overlap — allows GPU to pipeline
    # the next submit while the CPU processes/draws the current result.
    cuda_stream: Optional[torch.cuda.Stream] = (
        torch.cuda.Stream() if device != "cpu" else None
    )

    try:
        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            t_grab0 = time.monotonic()
            frame = frame_source()
            t_grab1 = time.monotonic()
            if frame is None:
                continue
            t0 = time.monotonic()
            if plate_target:
                apply_plate_hue_bias_bgr(frame, plate_target)
            inp = model_input(frame)

            t_pre_predict = time.monotonic()

            # FIX: Wrap predict in inference_mode to suppress gradient tracking
            # overhead inside YOLO's AutoBackend on every call (~2–5 ms saved).
            # Also route through the CUDA stream so the driver can overlap
            # kernel launches with CPU-side drawing work.
            with torch.inference_mode():
                if cuda_stream is not None:
                    with torch.cuda.stream(cuda_stream):
                        results = model.predict(
                            inp,
                            device=device,
                            half=use_half,
                            verbose=False,
                        )
                    # Sync only the stream we used, not the entire device.
                    torch.cuda.current_stream().wait_stream(cuda_stream)
                else:
                    results = model.predict(
                        inp,
                        device=device,
                        half=use_half,
                        verbose=False,
                    )

            t_post_predict = time.monotonic()

            if not results:
                continue
            draw_obb_on_bgr(frame, results[0])
            t1 = time.monotonic()
            fps_ema = tick_fps_ema(fps_ema, t1 - t0)

            if diag:
                prep_ms = 1000.0 * (t_pre_predict - t0)
                draw_ms = 1000.0 * (t1 - t_post_predict)
                print(
                    f"grab={1000.0 * (t_grab1 - t_grab0):.1f}ms  "
                    f"predict={1000.0 * (t_post_predict - t_pre_predict):.1f}ms  "
                    f"prep={prep_ms:.1f}ms  draw={draw_ms:.1f}ms",
                    flush=True,
                )

            if headless:
                frame_count += 1

                # FIX: Reset the FPS baseline after warmup frames so CUDA JIT
                # compilation and first-run overhead don't drag down the
                # reported actual FPS for the entire session.
                if frame_count == _WARMUP_FRAMES:
                    t_start = time.monotonic()
                    fps_log_prev = t_start
                    frame_count = 0
                    continue

                # FIX: Yield to the OS scheduler so the CUDA driver can retire
                # completed kernels and pipeline the next inference.  imshow/
                # waitKey provided this implicitly in non-headless mode (~1 ms
                # window); without it the tight spin loop serializes submits.
                time.sleep(0)

                if t1 - fps_log_prev >= HEADLESS_FPS_LOG_SEC:
                    elapsed = t1 - t_start
                    actual = frame_count / elapsed if elapsed > 0 else 0.0
                    print(
                        f"FPS ema={fps_ema:.1f}  actual={actual:.1f}  "
                        f"frames={frame_count}",
                        flush=True,
                    )
                    fps_log_prev = t1
            else:
                cv2.putText(
                    frame,
                    f"FPS: {fps_ema:.1f}",
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                view = frame_for_preview(frame, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)
                cv2.imshow(window_name, view)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
    finally:
        if not headless:
            cv2.destroyAllWindows()


def run_basler_loop(
    model: YOLO,
    camera: pylon.InstantCamera,
    device: str,
    use_half: bool,
    plate_target: str,
    window_name: str,
    uses_native_bgr8: bool,
    headless: bool,
) -> None:
    converter = None if uses_native_bgr8 else create_frame_converter()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    fq = LatestFrameQueue()
    stop = threading.Event()
    grab_thread = threading.Thread(
        target=basler_grab_thread_fn,
        args=(camera, converter, uses_native_bgr8, fq, stop),
        name="basler-grab",
        daemon=True,
    )
    grab_thread.start()
    try:

        def grab_bgr() -> Optional[np.ndarray]:
            return fq.get(BASELER_QUEUE_GET_TIMEOUT_SEC)

        run_inference_on_bgr_frames(
            model,
            window_name,
            grab_bgr,
            device,
            use_half,
            plate_target,
            headless,
        )
    finally:
        stop.set()
        if camera.IsGrabbing():
            camera.StopGrabbing()
        grab_thread.join(timeout=3.0)
        if camera.IsOpen():
            camera.Close()


def run_realsense_loop(
    model: YOLO,
    cap_fps: Optional[float],
    device: str,
    use_half: bool,
    plate_target: str,
    window_name: str,
    headless: bool,
) -> None:
    frames = realsense_color_frames(cap_fps)
    try:

        def next_bgr() -> Optional[np.ndarray]:
            return next(frames)

        run_inference_on_bgr_frames(
            model,
            window_name,
            next_bgr,
            device,
            use_half,
            plate_target,
            headless,
        )
    finally:
        frames.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO OBB from Basler or RealSense.")
    p.add_argument(
        "--source",
        choices=("basler", "realsense"),
        default="basler",
        help="Camera backend (default: basler).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        metavar="N",
        help="Cap capture at N FPS (Basler + RealSense). Default: uncapped / sensor max.",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="No GUI; log FPS to stdout (~1 Hz). Use for throughput without imshow.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    plate_target = parse_plate_target()
    device = resolve_device()
    use_half = os.environ.get("YOLO_HALF", "1") != "0" and device != "cpu"
    try:
        model = load_model(MODEL_PATH)
    except Exception as err:
        print(f"Failed to load model: {err}")
        return 1
    print(
        f"Using device={device}, half={use_half} (set YOLO_DEVICE / YOLO_HALF=0 to override)"
    )
    if plate_target:
        print(
            f"Plate hue bias: target={plate_target} (+{PLATE_BIAS_ADD}/-{PLATE_BIAS_SUB} BGR)"
        )
    cap_fps = args.fps if args.fps is not None and args.fps > 0 else None
    print(
        f"Source={args.source}, capture_fps_cap={'uncapped' if cap_fps is None else cap_fps}, "
        f"headless={args.headless}"
    )
    if diag_timing_enabled():
        print(
            "OBB_DIAG_TIMING: per frame — grab, predict (YOLO), prep (tensor/plate), draw.",
            flush=True,
        )
    if args.headless:
        print(
            f"Headless: EMA + actual FPS ~1 Hz (Ctrl+C to stop). "
            f"First {_WARMUP_FRAMES} frames skipped from FPS baseline. "
            "Set OBB_DIAG_TIMING=1 for timing lines.",
            flush=True,
        )
    else:
        print(
            f"Preview capped to max {PREVIEW_MAX_WIDTH}x{PREVIEW_MAX_HEIGHT} (inference uses full frame)."
        )
    try:
        if args.source == "basler":
            cam, native = open_default_camera(cap_fps)
            print(
                "Basler: native BGR8."
                if native
                else "Basler: ImageFormatConverter (LSB BGR8packed)."
            )
            run_basler_loop(
                model,
                cam,
                device,
                use_half,
                plate_target,
                WINDOW_BASLER,
                native,
                args.headless,
            )
        else:
            run_realsense_loop(
                model,
                cap_fps,
                device,
                use_half,
                plate_target,
                WINDOW_REALSENSE,
                args.headless,
            )
    except KeyboardInterrupt:
        pass
    except Exception as err:
        print(f"Runtime error: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
