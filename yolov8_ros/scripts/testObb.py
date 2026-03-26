#!/usr/bin/env python3
"""Basler or RealSense + YOLO OBB on Jetson / NVIDIA GPU using PyTorch weights only (no CoreML)."""

import argparse
import os
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


def bgr_frame_to_model_input(
    frame: np.ndarray,
    device: str,
    use_half: bool,
    imgsz: int,
) -> torch.Tensor:
    dtype = torch.float16 if use_half else torch.float32
    t = torch.as_tensor(frame, dtype=dtype, device=device)
    t = t / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    if t.shape[2] != imgsz or t.shape[3] != imgsz:
        t = torch.nn.functional.interpolate(
            t,
            size=(imgsz, imgsz),
            mode="bilinear",
            align_corners=False,
        )
    t = t[:, [2, 1, 0], :, :]
    return t


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
    polys = _to_numpy(xyxyxyxy).astype(np.int32)
    confs = _to_numpy(obb.conf)
    clss = _to_numpy(obb.cls).astype(int)
    names = getattr(result, "names", None)
    for i, poly in enumerate(polys):
        pts = poly.reshape(-1, 1, 2)
        cid = int(clss[i])
        hue = (40 * cid) % 180
        bgr = cv2.cvtColor(
            np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR
        )[0, 0]
        color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))
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
) -> None:
    imgsz = resolve_inference_imgsz(model)
    fps_ema = 0.0
    t_prev = time.monotonic()
    try:
        while True:
            t_now = time.monotonic()
            fps_ema = tick_fps_ema(fps_ema, t_now - t_prev)
            t_prev = t_now
            frame = frame_source()
            if frame is None:
                continue
            if plate_target:
                apply_plate_hue_bias_bgr(frame, plate_target)
            results = model.predict(
                bgr_frame_to_model_input(frame, device, use_half, imgsz),
                device=device,
                half=use_half,
                verbose=False,
            )
            if not results:
                continue
            draw_obb_on_bgr(frame, results[0])
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
            cv2.imshow(window_name, frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


def run_basler_loop(
    model: YOLO,
    camera: pylon.InstantCamera,
    device: str,
    use_half: bool,
    plate_target: str,
    window_name: str,
    uses_native_bgr8: bool,
) -> None:
    converter = None if uses_native_bgr8 else create_frame_converter()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    try:

        def grab_bgr() -> Optional[np.ndarray]:
            grab = camera.RetrieveResult(
                200, pylon.TimeoutHandling_Return
            )
            if grab is None or not grab.GrabSucceeded():
                if grab is not None:
                    grab.Release()
                return None
            if uses_native_bgr8:
                out = grab.GetArray()
            else:
                assert converter is not None
                out = converter.Convert(grab).GetArray()
            grab.Release()
            return out

        run_inference_on_bgr_frames(
            model, window_name, grab_bgr, device, use_half, plate_target
        )
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()


def run_realsense_loop(
    model: YOLO,
    cap_fps: Optional[float],
    device: str,
    use_half: bool,
    plate_target: str,
    window_name: str,
) -> None:
    frames = realsense_color_frames(cap_fps)
    try:

        def next_bgr() -> Optional[np.ndarray]:
            return next(frames)

        run_inference_on_bgr_frames(
            model, window_name, next_bgr, device, use_half, plate_target
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
        f"Source={args.source}, capture_fps_cap={'uncapped' if cap_fps is None else cap_fps}"
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
                model, cam, device, use_half, plate_target, WINDOW_BASLER, native
            )
        else:
            run_realsense_loop(
                model, cap_fps, device, use_half, plate_target, WINDOW_REALSENSE
            )
    except KeyboardInterrupt:
        pass
    except Exception as err:
        print(f"Runtime error: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
