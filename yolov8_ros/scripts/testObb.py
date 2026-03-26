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
MODEL_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent / "model_test" / "best.pt"
)

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
    """In-place BGR nudge — cheap vs HSV/blur."""
    if target == "blue":
        frame[:, :, 2] = np.clip(
            frame[:, :, 2].astype(np.int16) + PLATE_BIAS_ADD, 0, 255
        ).astype(np.uint8)
        frame[:, :, 0] = np.clip(
            frame[:, :, 0].astype(np.int16) - PLATE_BIAS_SUB, 0, 255
        ).astype(np.uint8)
    elif target == "red":
        frame[:, :, 0] = np.clip(
            frame[:, :, 0].astype(np.int16) + PLATE_BIAS_ADD, 0, 255
        ).astype(np.uint8)
        frame[:, :, 2] = np.clip(
            frame[:, :, 2].astype(np.int16) - PLATE_BIAS_SUB, 0, 255
        ).astype(np.uint8)


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


def open_default_camera(cap_fps: Optional[float]) -> pylon.InstantCamera:
    transport_layer_factory: pylon.TlFactory = pylon.TlFactory.GetInstance()
    devices = transport_layer_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera detected.")
    camera: pylon.InstantCamera = pylon.InstantCamera(
        transport_layer_factory.CreateDevice(devices[0])
    )
    camera.Open()
    try:
        if cap_fps is not None and cap_fps > 0:
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(float(cap_fps))
        else:
            camera.AcquisitionFrameRateEnable.SetValue(False)
    except Exception:
        pass
    return camera


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


def realsense_color_frames(cap_fps: Optional[float]) -> Generator[np.ndarray, None, None]:
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
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def tick_fps_ema(prev_ema: float, dt: float) -> float:
    inst: float = 1.0 / max(dt, 1e-9)
    if prev_ema <= 0.0:
        return inst
    return 0.9 * prev_ema + 0.1 * inst


def run_inference_on_bgr_frames(
    model: YOLO,
    window_name: str,
    frame_source: Callable[[], Optional[np.ndarray]],
    device: str,
    use_half: bool,
    plate_target: str,
) -> None:
    fps_ema: float = 0.0
    t_prev: float = time.monotonic()
    try:
        while True:
            frame = frame_source()
            if frame is None:
                continue
            if plate_target:
                apply_plate_hue_bias_bgr(frame, plate_target)
            results = model.predict(
                frame,
                device=device,
                half=use_half,
                verbose=False,
            )
            if not results:
                continue
            annotated_frame = results[0].plot()
            dt_loop: float = time.monotonic() - t_prev
            t_prev = time.monotonic()
            fps_ema = tick_fps_ema(fps_ema, dt_loop)
            cv2.putText(
                annotated_frame,
                f"FPS: {fps_ema:.1f}",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated_frame)
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
) -> None:
    frame_converter: pylon.ImageFormatConverter = create_frame_converter()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    try:

        def grab_bgr() -> Optional[np.ndarray]:
            grab_result: pylon.GrabResult = camera.RetrieveResult(
                1000, pylon.TimeoutHandling_ThrowException
            )
            if not grab_result.GrabSucceeded():
                grab_result.Release()
                return None
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
            camera: pylon.InstantCamera = open_default_camera(cap_fps)
            try:
                run_basler_loop(
                    model=model,
                    camera=camera,
                    device=device,
                    use_half=use_half,
                    plate_target=plate_target,
                    window_name=WINDOW_BASLER,
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
