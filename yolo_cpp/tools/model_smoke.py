#!/usr/bin/env python3
"""Camera-backed YOLO smoke test for `yolo_cpp/models/latest_best.pt`.

Default path:
- Basler camera via pypylon
- `yolo_cpp/models/latest_best.pt`
- OBB inference

This is intentionally small: grab camera frames, run one Ultralytics inference
pass per frame, print counts, and optionally show the annotated preview.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from pypylon import pylon
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "pypylon is required for the Basler smoke test inside the Jetson container"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "yolo_cpp" / "models" / "latest_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basler camera + YOLO smoke test.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the .pt model (defaults to yolo_cpp/models/latest_best.pt).",
    )
    parser.add_argument(
        "--task",
        choices=["detect", "obb", "segment", "pose", "classify"],
        default="obb",
        help="Ultralytics task for the model.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold."
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of frames to process before exiting.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        default=True,
        help="Display annotated frames in an OpenCV window.",
    )
    parser.add_argument(
        "--no-view",
        action="store_false",
        dest="view",
        help="Disable the OpenCV window.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the last annotated frame.",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Requested camera width."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Requested camera height.",
    )
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=5000.0,
        help="Manual exposure in microseconds.",
    )
    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        help="Use auto exposure instead of a fixed exposure.",
    )
    parser.add_argument(
        "--use-full-sensor",
        action="store_true",
        default=True,
        help="Keep the camera on the full sensor when possible.",
    )
    parser.add_argument(
        "--no-use-full-sensor",
        action="store_false",
        dest="use_full_sensor",
        help="Allow the camera to use the requested width/height as ROI.",
    )
    return parser.parse_args()


def build_converter() -> pylon.ImageFormatConverter:
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def configure_camera(
    camera: pylon.InstantCamera,
    *,
    width: int,
    height: int,
    exposure_us: float,
    auto_exposure: bool,
    use_full_sensor: bool,
) -> dict[str, float]:
    camera.Open()
    try:
        max_width = camera.WidthMax.GetValue()
        max_height = camera.HeightMax.GetValue()
        target_width = max_width if use_full_sensor else min(width, max_width)
        target_height = max_height if use_full_sensor else min(height, max_height)
        camera.OffsetX.SetValue(0)
        camera.OffsetY.SetValue(0)
        camera.Width.SetValue(target_width)
        camera.Height.SetValue(target_height)

        try:
            symbolics = list(camera.PixelFormat.GetSymbolics())
        except Exception:
            symbolics = []
        for candidate in [
            "BayerRG8",
            "BayerBG8",
            "BayerGB8",
            "BayerGR8",
            "BGR8",
            "RGB8",
            "Mono8",
        ]:
            if candidate in symbolics:
                camera.PixelFormat.SetValue(candidate)
                break

        if auto_exposure:
            camera.ExposureAuto.SetValue("Continuous")
        else:
            camera.ExposureAuto.SetValue("Off")
            camera.ExposureTime.SetValue(float(exposure_us))

        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        try:
            fps = float(camera.ResultingFrameRate.GetValue())
        except Exception:
            fps = 0.0
        return {
            "width": float(target_width),
            "height": float(target_height),
            "fps": fps,
        }
    except Exception:
        camera.Close()
        raise


def grab_bgr_frame(
    camera: pylon.InstantCamera, converter: pylon.ImageFormatConverter
) -> np.ndarray | None:
    grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
    if not grab:
        return None
    try:
        if not grab.GrabSucceeded():
            return None
        image = converter.Convert(grab)
        return image.GetArray()
    finally:
        grab.Release()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera found")

    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
    model = YOLO(str(args.model), task=args.task)
    converter = build_converter()

    window_name = "yolo_cpp smoke"
    if args.view:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        settings = configure_camera(
            camera,
            width=args.width,
            height=args.height,
            exposure_us=args.exposure_us,
            auto_exposure=args.auto_exposure,
            use_full_sensor=args.use_full_sensor,
        )
        print(
            f"camera={settings['width']:.0f}x{settings['height']:.0f} "
            f"fps={settings['fps']:.2f} model={args.model}"
        )

        first_frame_time = None
        last_annotated = None
        frame_count = 0
        while camera.IsGrabbing() and frame_count < args.frames:
            frame = grab_bgr_frame(camera, converter)
            if frame is None:
                continue
            if first_frame_time is None:
                first_frame_time = perf_counter()

            frame_count += 1
            results = model.predict(
                frame, imgsz=args.imgsz, conf=args.conf, verbose=False
            )
            result = results[0]
            annotated = result.plot()
            last_annotated = annotated

            det_count = 0
            if result.boxes is not None:
                det_count = len(result.boxes)
            elif getattr(result, "obb", None) is not None:
                det_count = len(result.obb)

            print(f"frame={frame_count} detections={det_count}")

            if args.view:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

        if last_annotated is not None and args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.output), last_annotated):
                raise RuntimeError(f"Failed to write {args.output}")
            print(f"saved={args.output}")

        if first_frame_time is not None and frame_count > 0:
            elapsed = perf_counter() - first_frame_time
            print(
                f"frames={frame_count} elapsed={elapsed:.3f}s avg_fps={frame_count / elapsed:.2f}"
            )
        return 0
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()
        if args.view:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
