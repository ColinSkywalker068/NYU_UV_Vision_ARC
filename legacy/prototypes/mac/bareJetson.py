#!/usr/bin/env python3
"""Basler + YOLO OBB on Jetson / NVIDIA GPU using PyTorch weights only (no CoreML)."""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pypylon import pylon
from ultralytics import YOLO


WINDOW_NAME: str = "Basler YOLO OBB (Jetson)"
MODEL_PATH: Path = Path(__file__).resolve().parents[2] / "models" / "best.pt"

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


def open_default_camera() -> pylon.InstantCamera:
    transport_layer_factory: pylon.TlFactory = pylon.TlFactory.GetInstance()
    devices = transport_layer_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera detected.")
    camera: pylon.InstantCamera = pylon.InstantCamera(
        transport_layer_factory.CreateDevice(devices[0])
    )
    camera.Open()
    try:
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(30.0)
    except Exception:
        pass
    return camera


def create_frame_converter() -> pylon.ImageFormatConverter:
    converter: pylon.ImageFormatConverter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def run_inference_loop(
    model: YOLO,
    camera: pylon.InstantCamera,
    device: str,
    use_half: bool,
    plate_target: str,
) -> None:
    frame_converter: pylon.ImageFormatConverter = create_frame_converter()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    try:
        while camera.IsGrabbing():
            grab_result: pylon.GrabResult = camera.RetrieveResult(
                1000, pylon.TimeoutHandling_ThrowException
            )
            if not grab_result.GrabSucceeded():
                grab_result.Release()
                continue
            converted_image = frame_converter.Convert(grab_result)
            frame = converted_image.GetArray()
            grab_result.Release()
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
            cv2.imshow(WINDOW_NAME, annotated_frame)
            pressed_key: int = cv2.waitKey(1) & 0xFF
            if pressed_key == ord("q"):
                break
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()
        cv2.destroyAllWindows()


def main() -> int:
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
    try:
        camera: pylon.InstantCamera = open_default_camera()
    except Exception as err:
        print(f"Failed to open Basler camera: {err}")
        return 1
    try:
        run_inference_loop(
            model=model,
            camera=camera,
            device=device,
            use_half=use_half,
            plate_target=plate_target,
        )
    except KeyboardInterrupt:
        pass
    except Exception as err:
        print(f"Runtime error: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
