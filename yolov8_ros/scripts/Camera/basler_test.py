#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from time import perf_counter
from collections.abc import Callable, Iterable
import numpy as np

import cv2
from pypylon import pylon
from ultralytics import YOLO

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

DURATION_SECONDS = 5.0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 0  # 0 => do not set, let camera run max
DEFAULT_EXPOSURE_US = 5000.0
DEFAULT_GAIN_DB = 5.0
DEFAULT_MODEL_PATH = "/home/ultraviolet/CV_Detection/yolov8_ros/models/best.engine"


def _record_setting(
    applied_settings: dict[str, str | float],
    name: str,
    getter: Callable[[], str | int | float],
) -> None:
    try:
        applied_settings[name] = getter()
    except Exception:
        applied_settings[name] = "(error)"


def get_pixel_format_symbolics(camera: pylon.InstantCamera) -> list[str]:
    try:
        if not hasattr(camera, "PixelFormat"):
            return []
        node = camera.PixelFormat
        if hasattr(node, "GetSymbolics"):
            return list(node.GetSymbolics())
        if hasattr(node, "Symbolics"):
            return list(node.Symbolics)
        return []
    except Exception:
        return []


def choose_mono8_format(symbolics: list[str]) -> str | None:
    if "Mono8" in symbolics:
        return "Mono8"
    for s in symbolics:
        if "Mono" in s and "8" in s:
            return s
    return None


def choose_preferred_color_format(symbolics: list[str]) -> str | None:
    preferred_order = [
        "BayerRG8",
        "BayerBG8",
        "BayerGB8",
        "BayerGR8",
        "RGB8",
        "BGR8",
    ]
    for fmt in preferred_order:
        if fmt in symbolics:
            return fmt
    return None


def _set_node_value_if_writable(node: object, value: object) -> bool:
    try:
        if hasattr(node, "IsWritable") and not node.IsWritable():
            return False
        node.SetValue(value)
        return True
    except Exception:
        return False


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
    applied_settings: dict[str, str | float] | None = None,
) -> dict[str, float]:
    max_width = camera.WidthMax.GetValue()
    max_height = camera.HeightMax.GetValue()
    target_width = max_width if use_full_sensor else min(width, max_width)
    target_height = max_height if use_full_sensor else min(height, max_height)
    _set_node_value_if_writable(camera.OffsetX, 0)
    _set_node_value_if_writable(camera.OffsetY, 0)
    _set_node_value_if_writable(camera.Width, target_width)
    _set_node_value_if_writable(camera.Height, target_height)
    if applied_settings is not None:
        _record_setting(applied_settings, "Width", lambda: camera.Width.GetValue())
        _record_setting(applied_settings, "Height", lambda: camera.Height.GetValue())
    symbolics = get_pixel_format_symbolics(camera)
    if applied_settings is not None:
        applied_settings["PixelFormat_available"] = (
            ", ".join(symbolics) if symbolics else "(none)"
        )
    if symbolics:
        # Prefer color-capable formats first. Falling back to mono can make
        # the image appear dark/flat even with higher exposure.
        chosen = choose_preferred_color_format(symbolics)
        if chosen is None:
            print("No color-capable format found, falling back to mono8")
            chosen = choose_mono8_format(symbolics)
        if chosen is not None:
            _set_node_value_if_writable(camera.PixelFormat, chosen)
    if applied_settings is not None:
        _record_setting(
            applied_settings, "PixelFormat", lambda: camera.PixelFormat.GetValue()
        )
    if fps > 0:
        try:
            if (
                camera.AcquisitionFrameRateEnable.GetValue()
                and camera.AcquisitionFrameRate.IsWritable()
            ):
                camera.AcquisitionFrameRate.SetValue(float(fps))
        except Exception:
            pass
    if auto_exposure:
        _set_node_value_if_writable(camera.ExposureAuto, "Continuous")
    else:
        _set_node_value_if_writable(camera.ExposureAuto, "Off")
        _set_node_value_if_writable(camera.ExposureTime, float(exposure_us))
    if auto_gain:
        _set_node_value_if_writable(camera.GainAuto, "Continuous")
    else:
        _set_node_value_if_writable(camera.GainAuto, "Off")
        _set_node_value_if_writable(camera.Gain, float(gain_db))
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


def letterbox_frame(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    src_h, src_w = frame.shape[:2]
    if src_h == target_h and src_w == target_w:
        return frame

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
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


def letterbox_frame_gpu(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Letterbox resize+pad on GPU (PyTorch CUDA). Returns numpy HWC BGR for display/model."""
    if frame is None or frame.size == 0:
        return frame
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return letterbox_frame(frame, target_w, target_h)
    try:
        src_h, src_w = frame.shape[:2]
        if src_h == target_h and src_w == target_w:
            return frame
        scale = min(target_w / src_w, target_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        device = torch.device("cuda")
        contiguous_frame = np.ascontiguousarray(frame)
        x = torch.from_numpy(contiguous_frame).to(device, non_blocking=True)
        x = x.permute(2, 0, 1).unsqueeze(0).float()
        x = torch.nn.functional.interpolate(
            x, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        x = torch.nn.functional.pad(
            x, (left, right, top, bottom), mode="constant", value=114.0
        )
        x = x.squeeze(0).permute(1, 2, 0).contiguous().byte()
        result = x.cpu().numpy()
        return np.ascontiguousarray(result, dtype=np.uint8)
    except Exception:
        return letterbox_frame(frame, target_w, target_h)


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
    fps_window: int,
    model_size: int | None = None,
    use_gpu_letterbox: bool = False,
) -> None:
    window_name: str = "Basler Test"
    if show:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception:
            show = False
    start: float | None = None  # first good frame
    batch_start: float | None = None  # batch timing start
    end: float | None = None
    frames: int = 0
    last_batch_fps: float = 0.0
    stop_requested: bool = False
    while camera.IsGrabbing() and not stop_requested:
        grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
        if grab and grab.GrabSucceeded():
            image = converter.Convert(grab)
            if start is None:
                start = perf_counter()
                batch_start = start
            frames += 1
            frame = image.GetArray()
            letterbox_fn = letterbox_frame_gpu if use_gpu_letterbox else letterbox_frame
            letterboxed = letterbox_fn(frame, target_w, target_h)
            # OpenCV expects C-contiguous uint8 layout (cv::Mat-compatible).
            letterboxed = np.ascontiguousarray(letterboxed, dtype=np.uint8)
            if show:
                # Show a live preview before inference so UI still appears
                # even if model inference blocks or fails.
                try:
                    cv2.imshow(window_name, letterboxed)
                    key_preview: int = cv2.waitKey(1) & 0xFF
                except Exception:
                    show = False
                    key_preview = 0
                if key_preview in (ord("q"), ord("z"), 27):
                    stop_requested = True
                    if grab:
                        grab.Release()
                    continue
            # Use model_size if specified (for quantized models with fixed input size),
            # otherwise use target dimensions
            if model_size is not None:
                imgsz_val = model_size
            else:
                imgsz_val = max(target_h, target_w)
            try:
                if mode == "track":
                    results = model.track(
                        letterboxed,
                        imgsz=imgsz_val,
                        conf=conf,
                        verbose=False,
                        persist=True,
                    )
                else:
                    results = model(
                        letterboxed,
                        imgsz=imgsz_val,
                        conf=conf,
                        verbose=False,
                    )
            except Exception:
                results = []
            boxes = []
            if results:
                first_result = results[0]
                if (
                    mode == "obb"
                    and hasattr(first_result, "obb")
                    and first_result.obb is not None
                ):
                    if (
                        hasattr(first_result.obb, "xyxy")
                        and first_result.obb.xyxy is not None
                    ):
                        boxes = first_result.obb.xyxy.tolist()
                    elif (
                        hasattr(first_result.obb, "xyxyxyxy")
                        and first_result.obb.xyxyxyxy is not None
                    ):
                        obb_points_raw = first_result.obb.xyxyxyxy
                        if hasattr(obb_points_raw, "cpu"):
                            obb_points = obb_points_raw.cpu().numpy()
                        elif hasattr(obb_points_raw, "numpy"):
                            obb_points = obb_points_raw.numpy()
                        else:
                            obb_points = np.asarray(obb_points_raw)
                        for polygon in obb_points:
                            xs = polygon[:, 0]
                            ys = polygon[:, 1]
                            boxes.append(
                                [
                                    float(xs.min()),
                                    float(ys.min()),
                                    float(xs.max()),
                                    float(ys.max()),
                                ]
                            )
                elif first_result.boxes is not None:
                    boxes = first_result.boxes.xyxy.tolist()

            if batch_start is not None and fps_window > 0 and frames % fps_window == 0:
                now = perf_counter()
                elapsed_batch = now - batch_start
                if elapsed_batch > 0:
                    last_batch_fps = fps_window / elapsed_batch
                    print(f"FPS({fps_window}): {last_batch_fps:.2f}")
                batch_start = now

            if show:
                draw_boxes(letterboxed, boxes)
                if last_batch_fps > 0:
                    cv2.putText(
                        letterboxed,
                        f"FPS({fps_window}): {last_batch_fps:.2f}",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                try:
                    cv2.imshow(window_name, letterboxed)
                    key: int = cv2.waitKey(1) & 0xFF
                except Exception:
                    show = False
                    key = 0
                if key in (ord("q"), ord("z"), 27):
                    stop_requested = True
        if grab:
            grab.Release()
        if (
            duration_s > 0
            and start is not None
            and (perf_counter() - start) >= duration_s
        ):
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
        default=True,
        help="Display frames in a simple OpenCV window (press q or ESC to exit)",
    )
    parser.add_argument(
        "--no-show",
        action="store_false",
        dest="show",
        help="Disable display window",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLO engine/pt/onnx model",
    )
    parser.add_argument(
        "--mode",
        choices=["detect", "track", "obb"],
        default="obb",
        help="Run plain detection, ByteTrack-style tracking, or oriented bounding boxes (OBB)",
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
    parser.add_argument(
        "--no-use-full-sensor",
        action="store_false",
        dest="use_full_sensor",
        help="Use native camera width/height settings (e.g., 640x480)",
    )
    parser.add_argument(
        "--fps-window",
        type=int,
        default=10,
        help="Frames per short-term FPS computation (e.g., 10)",
    )
    parser.add_argument(
        "--model-size",
        type=int,
        default=None,
        help="Model input size (square, e.g., 640). If not set, uses max(img-width, img-height)",
    )
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=DEFAULT_EXPOSURE_US,
        help="Manual exposure in microseconds (used when auto-exposure is off). Increase if too dark (e.g. 20000).",
    )
    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        help="Enable continuous auto-exposure (brightness adapts to scene).",
    )
    parser.add_argument(
        "--use-gpu-letterbox",
        action="store_true",
        help="Do letterbox resize+pad on GPU with PyTorch (requires CUDA). Can improve FPS.",
    )
    parser.add_argument(
        "--task",
        choices=["detect", "obb", "segment", "classify", "pose"],
        default=None,
        help="Explicit Ultralytics task. If omitted, inferred from --mode (obb->obb, otherwise detect).",
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
            auto_exposure=args.auto_exposure,
            exposure_us=args.exposure_us,
            auto_gain=False,
            gain_db=DEFAULT_GAIN_DB,
        )

        print(
            f"Basler started: {int(settings['width'])}x{int(settings['height'])} "
            f"@ {settings['fps']:.2f} FPS"
        )
        converter = build_converter()
        model_task = (
            args.task
            if args.task is not None
            else ("obb" if args.mode == "obb" else "detect")
        )
        model = YOLO(args.model_path, task=model_task)
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
            max(1, args.fps_window),
            args.model_size,
            args.use_gpu_letterbox,
        )

    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()


if __name__ == "__main__":
    main()
