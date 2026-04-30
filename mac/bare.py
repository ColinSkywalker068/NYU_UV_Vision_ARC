#!/usr/bin/env python3
"""Minimal Basler + YOLO OBB runner using CoreML package."""

import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np
from pypylon import pylon

# CoreML GPU runtime can deadlock with camera/UI threads on some macOS setups.
os.environ.setdefault("COREML_DISABLE_GPU", "1")

from ultralytics import YOLO


WINDOW_NAME: str = "Basler YOLO OBB"
COREML_MODEL_PATH: Path = (
    Path(__file__).resolve().parent.parent / "model_test" / "best.mlpackage"
)
PT_MODEL_PATH: Path = Path(__file__).resolve().parent.parent / "model_test" / "best.pt"
PROBE_TIMEOUT_SECONDS: int = int(os.environ.get("COREML_PROBE_TIMEOUT_SECONDS", "2"))

# Cheap color bias for plate experiments (BGR, in-place). No HSV / blur.
# PLATE_TARGET=blue  -> slight warm/red cast (often helps blue plates pop).
# PLATE_TARGET=red   -> slight cool/blue cast (often helps red plates pop).
# PLATE_TARGET unset/none/off -> disabled.
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
    """In-place BGR channel nudge — minimal cost vs full color space conversion."""
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


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(str(model_path), task="obb")


def run_coreml_probe_worker(model_path: str, queue: mp.Queue) -> None:
    try:
        os.environ.setdefault("COREML_DISABLE_GPU", "1")
        probe_model: YOLO = YOLO(model_path, task="obb")
        probe_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        probe_model.predict(probe_frame, verbose=False)
        queue.put("ok")
    except Exception as err:
        queue.put(f"error: {err}")


def run_coreml_probe(model_path: Path, timeout_seconds: int) -> tuple[bool, str]:
    context: mp.context.BaseContext = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    process: mp.Process = context.Process(
        target=run_coreml_probe_worker,
        args=(str(model_path), queue),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1)
        return False, "CoreML probe timed out (possible runtime deadlock)."
    if queue.empty():
        return False, f"CoreML probe exited unexpectedly with code {process.exitcode}."
    message: str = queue.get()
    if message == "ok":
        return True, "CoreML probe passed."
    return False, message


def load_runtime_model() -> YOLO:
    if COREML_MODEL_PATH.exists():
        print("Probing CoreML model in isolated process...")
        probe_passed, probe_message = run_coreml_probe(
            model_path=COREML_MODEL_PATH, timeout_seconds=PROBE_TIMEOUT_SECONDS
        )
        if probe_passed:
            print("CoreML probe passed. Loading CoreML model...")
            return load_model(COREML_MODEL_PATH)
        print(f"CoreML probe failed: {probe_message}")
        print("Falling back to PyTorch model for runtime stability...")
    if not PT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CoreML probe failed and fallback model not found: {PT_MODEL_PATH}"
        )
    return load_model(PT_MODEL_PATH)


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
        camera.AcquisitionFrameRate.SetValue(60.0)
    except Exception:
        pass
    return camera


def create_frame_converter() -> pylon.ImageFormatConverter:
    converter: pylon.ImageFormatConverter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def run_inference_loop(
    model: YOLO, camera: pylon.InstantCamera, plate_target: str
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
            results = model.predict(frame, verbose=False)
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
    try:
        model: YOLO = load_runtime_model()
    except Exception as err:
        print(f"Failed to load model: {err}")
        return 1
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
        run_inference_loop(model=model, camera=camera, plate_target=plate_target)
    except KeyboardInterrupt:
        pass
    except Exception as err:
        print(f"Runtime error: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
