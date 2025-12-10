#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from time import perf_counter
from collections.abc import Iterable
import numpy as np

import cv2
from pypylon import pylon
from ultralytics import YOLO

DURATION_SECONDS = 5.0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 60
DEFAULT_EXPOSURE_US = 5000.0
DEFAULT_GAIN_DB = 5.0
DEFAULT_MODEL_PATH = "/home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine"


def get_actual_fps(camera: pylon.InstantCamera) -> float:
    try:
        if hasattr(camera, "ResultingFrameRate"):
            return float(camera.ResultingFrameRate.GetValue())
        if hasattr(camera, "ResultingFrameRateAbs"):
            return float(camera.ResultingFrameRateAbs.GetValue())
    except Exception:
        pass

    try:
        if camera.AcquisitionFrameRateEnable.GetValue():
            return float(camera.AcquisitionFrameRate.GetValue())
    except Exception:
        pass

    return 0.0


def configure_camera(
    camera: pylon.InstantCamera,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    use_full_sensor: bool = True,
    auto_exposure: bool = False,
    exposure_us: float = DEFAULT_EXPOSURE_US,
    auto_gain: bool = False,
    gain_db: float = DEFAULT_GAIN_DB,
) -> dict[str, float]:
    max_width = camera.WidthMax.GetValue()
    max_height = camera.HeightMax.GetValue()

    target_width = max_width if use_full_sensor else min(width, max_width)
    target_height = max_height if use_full_sensor else min(height, max_height)

    try:
        camera.OffsetX.SetValue(0)
        camera.OffsetY.SetValue(0)
    except Exception:
        pass

    camera.Width.SetValue(target_width)
    camera.Height.SetValue(target_height)

    try:
        if camera.PixelFormat.IsWritable():
            camera.PixelFormat.SetValue("Mono8")
    except Exception:
        pass

    try:
        if camera.AcquisitionFrameRateEnable.GetValue():
            camera.AcquisitionFrameRate.SetValue(float(fps))
    except Exception:
        pass

    if auto_exposure:
        try:
            camera.ExposureAuto.SetValue("Continuous")
        except Exception:
            pass
    else:
        try:
            camera.ExposureAuto.SetValue("Off")
            camera.ExposureTime.SetValue(float(exposure_us))
        except Exception:
            pass

    if auto_gain:
        try:
            camera.GainAuto.SetValue("Continuous")
        except Exception:
            pass
    else:
        try:
            camera.GainAuto.SetValue("Off")
            camera.Gain.SetValue(float(gain_db))
        except Exception:
            pass

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    actual_width = camera.Width.GetValue()
    actual_height = camera.Height.GetValue()
    actual_fps = get_actual_fps(camera)

    return {
        "width": float(actual_width),
        "height": float(actual_height),
        "fps": float(actual_fps),
    }


def build_converter() -> pylon.ImageFormatConverter:
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def letterbox_frame(frame: cv2.Mat, target_w: int, target_h: int) -> cv2.Mat:
    src_h, src_w = frame.shape[:2]
    if src_h == target_h and src_w == target_w:
        return frame

    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def draw_boxes(frame: cv2.Mat, boxes: Iterable[Iterable[float]]) -> None:
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2,
        )


def run_grab_test(
    camera: pylon.InstantCamera,
    converter: pylon.ImageFormatConverter,
    show: bool,
    model: YOLO,
    mode: str,
    conf: float,
    duration_s: float,
    target_w: int,
    target_h: int,
) -> None:
    window_name: str = "Basler Test" if show else ""
    start: float | None = None
    end: float | None = None
    frames: int = 0
    stop_requested: bool = False
    while camera.IsGrabbing() and not stop_requested:
        grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
        if grab and grab.GrabSucceeded():
            image = converter.Convert(grab)
            if start is None:
                start = perf_counter()
            frames += 1
            frame = image.GetArray()
            # Letterbox to target size (keeps aspect ratio, minimal runtime)
            letterboxed = letterbox_frame(frame, target_w, target_h)
            if mode == "track":
                results = model.track(
                    letterboxed,
                    imgsz=(target_h, target_w),
                    conf=conf,
                    verbose=False,
                    persist=True,
                )
            else:
                results = model(
                    letterboxed,
                    imgsz=(target_h, target_w),
                    conf=conf,
                    verbose=False,
                )
            boxes = []
            if results and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.tolist()
            if show:
                draw_boxes(letterboxed, boxes)
                if start:
                    elapsed_now = max(1e-6, perf_counter() - start)
                    fps_now = frames / elapsed_now
                    cv2.putText(
                        letterboxed,
                        f"FPS: {fps_now:.2f}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window_name, letterboxed)
                key: int = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("z"), 27):
                    stop_requested = True
        if grab:
            grab.Release()
        if duration_s > 0 and start is not None and (perf_counter() - start) >= duration_s:
            break
    end = perf_counter() if start is not None else None
    elapsed: float = (end - start) if (start is not None and end is not None) else 0.0
    avg_fps: float = frames / elapsed if elapsed else 0.0
    print(f"Frames: {frames}, Elapsed: {elapsed:.3f}s, Avg FPS: {avg_fps:.2f}")
    if show:
        cv2.destroyAllWindows()


def build_arg_parser() -> ArgumentParser:
    parser: ArgumentParser = ArgumentParser(description="Basler grab test")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display frames in a simple OpenCV window (press q or ESC to exit)",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLO engine/pt/onnx model",
    )
    parser.add_argument(
        "--mode",
        choices=["detect", "track"],
        default="detect",
        help="Run plain detection or ByteTrack-style tracking",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run after first frame; 0 means run until keypress/stop",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Target network width (letterbox applied if camera differs)",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Target network height (letterbox applied if camera differs)",
    )
    parser.add_argument(
        "--use-full-sensor",
        action="store_true",
        default=True,
        help="Use full sensor; disable to force cam width/height settings",
    )
    return parser


def main() -> None:
    parser: ArgumentParser = build_arg_parser()
    args: Namespace = parser.parse_args()
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera found")

    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))

    try:
        camera.Open()
        settings = configure_camera(
            camera,
            width=args.img_width,
            height=args.img_height,
            fps=DEFAULT_FPS,
            use_full_sensor=args.use_full_sensor,
            auto_exposure=False,
            exposure_us=DEFAULT_EXPOSURE_US,
            auto_gain=False,
            gain_db=DEFAULT_GAIN_DB,
        )

        print(
            f"Basler started: {int(settings['width'])}x{int(settings['height'])} "
            f"@ {settings['fps']:.2f} FPS"
        )

        converter = build_converter()
        model = YOLO(args.model_path)
        run_grab_test(
            camera,
            converter,
            args.show,
            model,
            args.mode,
            args.conf,
            args.duration,
            args.img_width,
            args.img_height,
        )

    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()


if __name__ == "__main__":
    main()
