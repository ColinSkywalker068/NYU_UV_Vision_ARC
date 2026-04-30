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
MODEL_PATH = Path(
    "/home/ultraviolet/CV_Detection/model_test/best-roboflow-nitish-obb.pt"
)
# Preview window never exceeds this (full-res frame is still used for inference + overlays).
PREVIEW_MAX_WIDTH = 960
PREVIEW_MAX_HEIGHT = 540
HEADLESS_FPS_LOG_SEC = 1.0
BASLER_QUEUE_GET_TIMEOUT_SEC = (
    0.05  # 5 missed frames at 100 fps — was 1.0 (catastrophic stall)
)

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
        # Swap frame reference and clear flag in one lock window so put()
        # is never blocked by a lingering reference held by the consumer.
        with self._lock:
            frame = self._frame
            self._frame = None
            self._event.clear()
        return frame


def basler_grab_thread_fn(
    camera: pylon.InstantCamera,
    converter: Optional[pylon.ImageFormatConverter],
    uses_native_bgr8: bool,
    fq: LatestFrameQueue,
    stop: threading.Event,
) -> None:
    # Pre-allocated destination buffer — reused every frame so the OS never
    # needs to page in new memory during the copy while the grab is held.
    buf: Optional[np.ndarray] = None

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
            # GetArray() is a zero-copy view into Basler's DMA buffer.
            # We must copy before Release(), but copying into a pre-allocated
            # buffer avoids repeated OS allocation and minimises the window
            # the DMA buffer is held locked.
            raw = grab.GetArray()
            if buf is None or buf.shape != raw.shape or buf.dtype != raw.dtype:
                buf = np.empty(raw.shape, dtype=raw.dtype)
            np.copyto(buf, raw)
            grab.Release()
            fq.put(buf.copy())  # hand off an independent copy; buf is reused next frame
        else:
            assert converter is not None
            # Convert() internally allocates its own buffer — it does not hold
            # the DMA memory — so we can release the grab immediately after
            # conversion and before the array copy, freeing the camera buffer
            # slot as early as possible.
            converted = converter.Convert(grab)
            grab.Release()
            payload = converted.GetArray().copy()
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


def _try_set(label: str, fn: Callable[[], None]) -> None:
    """Apply a camera parameter, logging a warning on failure (not all models support all features)."""
    try:
        fn()
    except Exception as e:
        print(f"  [basler] {label}: {e}")


def open_default_camera(
    cap_fps: Optional[float],
) -> tuple[pylon.InstantCamera, bool]:
    tl = pylon.TlFactory.GetInstance()
    devices = tl.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera detected.")
    camera = pylon.InstantCamera(tl.CreateDevice(devices[0]))
    camera.Open()

    print(f"[basler] Model: {camera.GetDeviceInfo().GetModelName()}")

    # ── Pixel format ────────────────────────────────────────────────────────
    native_bgr8 = False
    _try_set("PixelFormat=BGR8", lambda: camera.PixelFormat.SetValue("BGR8"))
    try:
        native_bgr8 = camera.PixelFormat.GetValue() == "BGR8"
    except Exception:
        pass

    # ── Disable auto functions ───────────────────────────────────────────────
    # ExposureAuto=Continuous is the factory default. When an object appears,
    # the camera increases exposure to compensate for reduced average brightness
    # directly cutting achievable FPS. Must be Off for predictable throughput.
    _try_set("ExposureAuto=Off", lambda: camera.ExposureAuto.SetValue("Off"))
    _try_set("GainAuto=Off", lambda: camera.GainAuto.SetValue("Off"))

    # ── Exposure time ────────────────────────────────────────────────────────
    # FPS hard ceiling = 1 / ExposureTime. At 33ms you cannot exceed 30 fps.
    # Read from env so the user can tune without touching code.
    # Default 4000 us -> ~250 fps ceiling, well above any YOLO bottleneck.
    exposure_us = float(os.environ.get("BASLER_EXPOSURE_US", "4000"))

    def _set_exposure() -> None:
        lo = camera.ExposureTime.GetMin()
        hi = camera.ExposureTime.GetMax()
        camera.ExposureTime.SetValue(max(lo, min(hi, exposure_us)))

    _try_set(f"ExposureTime={exposure_us}us", _set_exposure)

    # ── AOI (Area of Interest) ───────────────────────────────────────────────
    # Reducing height directly increases the sensor's achievable frame rate:
    # fewer scanlines = shorter frame period at the hardware level.
    # Width/Height are env-configurable; 0 means use full sensor dimension.
    aoi_w = int(os.environ.get("BASLER_AOI_W", "0"))
    aoi_h = int(os.environ.get("BASLER_AOI_H", "0"))
    try:
        max_w = camera.Width.GetMax()
        max_h = camera.Height.GetMax()
        target_w = aoi_w if aoi_w > 0 else max_w
        target_h = aoi_h if aoi_h > 0 else max_h
        # Width/Height must be multiples of the camera's increment (usually 4 or 8).
        w_inc = camera.Width.GetInc()
        h_inc = camera.Height.GetInc()
        target_w = max(int(camera.Width.GetMin()), (target_w // w_inc) * w_inc)
        target_h = max(int(camera.Height.GetMin()), (target_h // h_inc) * h_inc)
        # Centre the AOI on the sensor.
        off_x = ((max_w - target_w) // 2 // w_inc) * w_inc
        off_y = ((max_h - target_h) // 2 // h_inc) * h_inc
        # Must reset offsets before shrinking Width/Height to avoid range errors.
        camera.OffsetX.SetValue(0)
        camera.OffsetY.SetValue(0)
        camera.Width.SetValue(target_w)
        camera.Height.SetValue(target_h)
        camera.OffsetX.SetValue(off_x)
        camera.OffsetY.SetValue(off_y)
        print(
            f"[basler] AOI: {target_w}x{target_h} @ offset ({off_x},{off_y}) "
            f"[sensor max {max_w}x{max_h}]"
        )
    except Exception as e:
        print(f"  [basler] AOI: could not set ({e}), using full sensor.")

    # ── Acquisition frame rate cap ───────────────────────────────────────────
    try:
        if cap_fps is not None and cap_fps > 0:
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(float(cap_fps))
        else:
            camera.AcquisitionFrameRateEnable.SetValue(False)
    except Exception as e:
        print(f"  [basler] FPS cap: {e}. Camera runs uncapped.")

    # ── Stream grabber (USB transport) tuning ────────────────────────────────
    # Applied after Open() but before StartGrabbing().
    # MaxTransferSize: larger = fewer USB transactions = lower CPU overhead.
    # NumMaxQueuedUrbs: more queued URBs = smoother delivery, less jitter.
    # MaxNumBuffer: more internal buffers = grab thread survives Jetson stalls.
    try:
        sg = camera.GetStreamGrabberParams()
        _try_set("MaxTransferSize=65536", lambda: sg.MaxTransferSize.SetValue(65536))
        _try_set("NumMaxQueuedUrbs=32", lambda: sg.NumMaxQueuedUrbs.SetValue(32))
    except Exception as e:
        print(f"  [basler] StreamGrabberParams unavailable before grab start: {e}")
    _try_set("MaxNumBuffer=32", lambda: camera.MaxNumBuffer.SetValue(32))

    # ── Report camera-side ceiling ───────────────────────────────────────────
    try:
        resulting = camera.ResultingFrameRate.GetValue()
        print(
            f"[basler] ResultingFrameRate (camera-reported ceiling): {resulting:.1f} fps"
        )
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
    proc_fps_ema = 0.0
    stream_fps_ema = 0.0
    fps_log_prev = time.monotonic()
    frame_count = 0
    diag = diag_timing_enabled()
    prev_grab_end = None

    cuda_stream: Optional[torch.cuda.Stream] = (
        torch.cuda.Stream() if device != "cpu" else None
    )

    # Warm up: runs one full predict to initialise the YOLO predictor, CUDA
    # JIT kernels, and letterbox state before the main loop starts.
    # This also sets model.predictor which the headless path uses directly.
    warmup_frame = frame_source()
    if warmup_frame is None:
        warmup_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with torch.inference_mode():
        model.predict(
            warmup_frame, imgsz=imgsz, device=device, half=use_half, verbose=False
        )
    predictor = model.predictor
    print("Predictor warmed up.", flush=True)

    try:
        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            t_grab0 = time.monotonic()
            frame = frame_source()
            t_grab1 = time.monotonic()
            if frame is None:
                continue
            if prev_grab_end is not None:
                stream_fps_ema = tick_fps_ema(
                    stream_fps_ema, max(t_grab1 - prev_grab_end, 1e-9)
                )
            prev_grab_end = t_grab1
            t0 = time.monotonic()
            if plate_target:
                apply_plate_hue_bias_bgr(frame, plate_target)

            t_pre_predict = time.monotonic()

            if headless:
                # Headless: run preprocess + forward pass only.
                # Skips YOLO's postprocess entirely (NMS, D2H transfer of the
                # raw prediction tensor, coordinate remapping) — all of which
                # are the source of the FPS drop when objects are on screen.
                # Results are unused in headless so this is safe.
                with torch.inference_mode():
                    imgs = predictor.preprocess([frame])
                    if cuda_stream is not None:
                        with torch.cuda.stream(cuda_stream):
                            predictor.model(imgs)
                        torch.cuda.current_stream().wait_stream(cuda_stream)
                    else:
                        predictor.model(imgs)
            else:
                # Non-headless: full predict so we get results for drawing.
                with torch.inference_mode():
                    if cuda_stream is not None:
                        with torch.cuda.stream(cuda_stream):
                            results = model.predict(
                                frame,
                                imgsz=imgsz,
                                device=device,
                                half=use_half,
                                verbose=False,
                            )
                        torch.cuda.current_stream().wait_stream(cuda_stream)
                    else:
                        results = model.predict(
                            frame,
                            imgsz=imgsz,
                            device=device,
                            half=use_half,
                            verbose=False,
                        )

            t_post_predict = time.monotonic()

            if not headless:
                if results:
                    draw_obb_on_bgr(frame, results[0])

            t1 = time.monotonic()
            proc_fps_ema = tick_fps_ema(proc_fps_ema, t1 - t0)

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
                time.sleep(0)
                if t1 - fps_log_prev >= HEADLESS_FPS_LOG_SEC:
                    print(
                        f"Proc FPS {proc_fps_ema:.1f}  Stream FPS {stream_fps_ema:.1f}",
                        flush=True,
                    )
                    fps_log_prev = t1
            else:
                cv2.putText(
                    frame,
                    f"Proc FPS: {proc_fps_ema:.1f}",
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Stream FPS: {stream_fps_ema:.1f}",
                    (8, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 200, 255),
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
            return fq.get(BASLER_QUEUE_GET_TIMEOUT_SEC)

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
