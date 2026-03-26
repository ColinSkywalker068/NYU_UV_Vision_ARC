#!/usr/bin/env python3
"""Basler or RealSense + YOLO OBB on Jetson / NVIDIA GPU using PyTorch weights only (no CoreML)."""

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional
from collections.abc import Generator

import cv2
import numpy as np
import torch
from pypylon import pylon
from ultralytics import YOLO


WINDOW_BASLER: str = "YOLO OBB — Basler"
WINDOW_REALSENSE: str = "YOLO OBB — RealSense"
MODEL_PATH: Path = Path("/home/ultraviolet/CV_Detection/model_test/best.pt")

PLATE_BIAS_ADD: int = int(os.environ.get("PLATE_BIAS_ADD", "8"))
PLATE_BIAS_SUB: int = int(os.environ.get("PLATE_BIAS_SUB", "6"))


def parse_plate_target() -> str:
    raw: str = os.environ.get("PLATE_TARGET", "").strip().lower()
    if raw in ("", "none", "off", "0"):
        return ""
    if raw in ("red", "blue"):
        return raw
    return ""


def apply_plate_hue_bias_bgr(frame: np.ndarray, target: str) -> None:
    """In-place BGR nudge — saturating ops, no int16 temp buffers."""
    if target == "blue":
        frame[:, :, 2] = cv2.add(frame[:, :, 2], PLATE_BIAS_ADD)
        frame[:, :, 0] = cv2.subtract(frame[:, :, 0], PLATE_BIAS_SUB)
    elif target == "red":
        frame[:, :, 0] = cv2.add(frame[:, :, 0], PLATE_BIAS_ADD)
        frame[:, :, 2] = cv2.subtract(frame[:, :, 2], PLATE_BIAS_SUB)


def resolve_device() -> str:
    override = os.environ.get("YOLO_DEVICE", "").strip()
    if override:
        return override
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path), task="obb")


def open_default_camera(
    cap_fps: Optional[float],
) -> tuple[pylon.InstantCamera, bool]:
    """Open first Basler device; try native BGR8 to skip CPU ImageFormatConverter when supported."""
    transport_layer_factory: pylon.TlFactory = pylon.TlFactory.GetInstance()
    devices = transport_layer_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera detected.")
    camera: pylon.InstantCamera = pylon.InstantCamera(
        transport_layer_factory.CreateDevice(devices[0])
    )
    camera.Open()
    uses_native_bgr8: bool = False
    try:
        camera.PixelFormat.SetValue("BGR8")
        uses_native_bgr8 = True
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
    return camera, uses_native_bgr8


def _realsense_best_bgr8_stream(
    rs: Any, width: int, height: int
) -> Optional[tuple[int, int, int]]:
    """Return (width, height, fps) for highest-FPS BGR8 profile at given resolution."""
    ctx = rs.context()
    devs = list(ctx.query_devices())
    if not devs:
        return None
    best: Optional[tuple[int, int, int]] = None
    best_fps: int = -1
    for sensor in devs[0].query_sensors():
        if not sensor.is_color_sensor():
            continue
        for p in sensor.get_stream_profiles():
            vp = p.as_video_stream_profile()
            if vp.format() != rs.format.bgr8:
                continue
            if vp.width() != width or vp.height() != height:
                continue
            f: int = int(vp.fps())
            if f > best_fps:
                best_fps = f
                best = (vp.width(), vp.height(), f)
    return best


def realsense_color_frames(
    cap_fps: Optional[float],
) -> Generator[np.ndarray, None, None]:
    import pyrealsense2 as rs  # type: ignore[import-untyped]

    pipeline: rs.pipeline = rs.pipeline()
    config: rs.config = rs.config()
    if cap_fps is not None and cap_fps > 0:
        fps_i: int = max(1, int(round(cap_fps)))
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
    converter: pylon.ImageFormatConverter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    # MSB alignment shifts 8-bit streams incorrectly (yellow tint on Bayer→BGR paths).
    converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
    return converter


def tick_fps_ema(prev_ema: float, dt: float) -> float:
    inst: float = 1.0 / max(dt, 1e-9)
    if prev_ema <= 0.0:
        return inst
    return 0.9 * prev_ema + 0.1 * inst


def resolve_inference_imgsz(model: YOLO) -> int:
    raw = os.environ.get("YOLO_IMGSZ", "").strip()
    if raw:
        return max(32, int(raw))
    imgsz = getattr(model, "overrides", None)
    if isinstance(imgsz, dict) and "imgsz" in imgsz:
        return int(imgsz["imgsz"])
    return 640


def bgr_frame_to_model_input(
    frame: np.ndarray,
    device: str,
    use_half: bool,
    imgsz: int,
) -> torch.Tensor:
    dtype = torch.float16 if use_half else torch.float32
    t = torch.from_numpy(frame).to(device, dtype=dtype)
    t = t / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = t[:, [2, 1, 0], :, :]
    if t.shape[2] != imgsz or t.shape[3] != imgsz:
        t = torch.nn.functional.interpolate(
            t,
            size=(imgsz, imgsz),
            mode="bilinear",
            align_corners=False,
        )
    return t


def class_label_from_names(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def draw_obb_on_bgr(frame_bgr: np.ndarray, result: Any) -> None:
    """Draw oriented boxes on frame (BGR, in-place). Faster than results.plot() for OBB."""
    if not hasattr(result, "obb") or result.obb is None:
        return
    obb = result.obb
    xyxyxyxy = getattr(obb, "xyxyxyxy", None)
    if xyxyxyxy is None or len(xyxyxyxy) == 0:
        return
    polys = xyxyxyxy
    if hasattr(polys, "cpu"):
        polys = polys.cpu().numpy()
    else:
        polys = np.asarray(polys)
    polys = polys.astype(np.int32)
    confs = obb.conf
    if hasattr(confs, "cpu"):
        confs = confs.cpu().numpy()
    else:
        confs = np.asarray(confs)
    clss = obb.cls
    if hasattr(clss, "cpu"):
        clss = clss.cpu().numpy().astype(int)
    else:
        clss = np.asarray(clss, dtype=int)
    names = getattr(result, "names", None)
    for i, poly in enumerate(polys):
        pts = poly.reshape(-1, 1, 2)
        cid = int(clss[i])
        hue = (40 * cid) % 180
        color_hsv = np.uint8([[[hue, 220, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        color = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
        cv2.polylines(frame_bgr, [pts], isClosed=True, color=color, thickness=2)
        xi = int(poly[:, 0].min())
        yi = int(poly[:, 1].min()) - 4
        yi = max(yi, 16)
        name = class_label_from_names(names, cid)
        label = f"{name} {float(confs[i]):.2f}"
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
    imgsz: int = resolve_inference_imgsz(model)
    fps_ema: float = 0.0
    t_prev: float = time.monotonic()
    try:
        while True:
            frame = frame_source()
            if frame is None:
                continue
            if plate_target:
                apply_plate_hue_bias_bgr(frame, plate_target)
            input_tensor: torch.Tensor = bgr_frame_to_model_input(
                frame, device, use_half, imgsz
            )
            results = model.predict(
                input_tensor,
                device=device,
                half=use_half,
                imgsz=imgsz,
                verbose=False,
            )
            if not results:
                continue
            draw_obb_on_bgr(frame, results[0])
            dt_loop: float = time.monotonic() - t_prev
            t_prev = time.monotonic()
            fps_ema = tick_fps_ema(fps_ema, dt_loop)
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
            pressed_key: int = cv2.waitKey(1) & 0xFF
            if pressed_key == ord("q"):
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
    frame_converter: Optional[pylon.ImageFormatConverter] = (
        None if uses_native_bgr8 else create_frame_converter()
    )
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    try:

        def grab_bgr() -> Optional[np.ndarray]:
            grab_result: pylon.GrabResult = camera.RetrieveResult(
                1000, pylon.TimeoutHandling_ThrowException
            )
            if not grab_result.GrabSucceeded():
                grab_result.Release()
                return None
            if uses_native_bgr8:
                frame = grab_result.GetArray()
            else:
                assert frame_converter is not None
                converted_image = frame_converter.Convert(grab_result)
                frame = converted_image.GetArray()
            grab_result.Release()
            return frame

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
    frame_iter: Generator[np.ndarray, None, None] = realsense_color_frames(cap_fps)
    try:

        def next_bgr() -> Optional[np.ndarray]:
            return next(frame_iter)

        run_inference_on_bgr_frames(
            model, window_name, next_bgr, device, use_half, plate_target
        )
    finally:
        frame_iter.close()


def parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="YOLO OBB from Basler or RealSense."
    )
    parser.add_argument(
        "--source",
        choices=("basler", "realsense"),
        default="basler",
        help="Camera backend (default: basler).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        metavar="N",
        help="Cap capture at N FPS (Basler + RealSense). Default: uncapped / sensor max.",
    )
    return parser.parse_args()


def main() -> int:
    args: argparse.Namespace = parse_args()
    plate_target: str = parse_plate_target()
    device: str = resolve_device()
    use_half: bool = os.environ.get("YOLO_HALF", "1") != "0" and device != "cpu"
    try:
        model: YOLO = load_model(MODEL_PATH)
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
    cap_fps: Optional[float] = (
        args.fps if args.fps is not None and args.fps > 0 else None
    )
    print(
        f"Source={args.source}, capture_fps_cap={'uncapped' if cap_fps is None else cap_fps}"
    )
    try:
        if args.source == "basler":
            camera, basler_native_bgr8 = open_default_camera(cap_fps)
            if basler_native_bgr8:
                print("Basler: using native BGR8 (no ImageFormatConverter).")
            else:
                print("Basler: using ImageFormatConverter (LSB-aligned BGR8packed).")
            try:
                run_basler_loop(
                    model=model,
                    camera=camera,
                    device=device,
                    use_half=use_half,
                    plate_target=plate_target,
                    window_name=WINDOW_BASLER,
                    uses_native_bgr8=basler_native_bgr8,
                )
            except KeyboardInterrupt:
                pass
        else:
            try:
                run_realsense_loop(
                    model=model,
                    cap_fps=cap_fps,
                    device=device,
                    use_half=use_half,
                    plate_target=plate_target,
                    window_name=WINDOW_REALSENSE,
                )
            except KeyboardInterrupt:
                pass
    except Exception as err:
        print(f"Runtime error: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# --- Deferred RealSense (not modified in this pass; apply when ready) ---
# 1. After pipeline.start(config): get color stream profile and print format,
#    e.g. profile.get_stream(rs.stream.color).as_video_stream_profile().format()
#    to confirm bgr8 (avoid silent rgb8 fallback / wrong imshow colors).
# 2. If 640x480 bgr8 is unavailable, expand _realsense_best_bgr8_stream resolution
#    or handle enable_stream fallback explicitly instead of implicit defaults.
