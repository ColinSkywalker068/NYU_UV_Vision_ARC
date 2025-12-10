#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from time import perf_counter
from typing import Dict

import cv2
from pypylon import pylon

DURATION_SECONDS = 5.0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 60
DEFAULT_EXPOSURE_US = 5000.0
DEFAULT_GAIN_DB = 5.0


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
) -> Dict[str, float]:
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


def run_grab_test(
    camera: pylon.InstantCamera,
    converter: pylon.ImageFormatConverter,
    show: bool,
) -> None:
    window_name: str = "Basler Test" if show else ""
    start: float = perf_counter()
    frames: int = 0
    stop_requested: bool = False
    while (perf_counter() - start) < DURATION_SECONDS and camera.IsGrabbing() and not stop_requested:
        grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
        if grab and grab.GrabSucceeded():
            image = converter.Convert(grab)
            frames += 1
            if show:
                frame = image.GetArray()
                cv2.imshow(window_name, frame)
                key: int = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    stop_requested = True
        if grab:
            grab.Release()
    elapsed: float = perf_counter() - start
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
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            fps=DEFAULT_FPS,
            use_full_sensor=True,
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
        run_grab_test(camera, converter, args.show)

    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()


if __name__ == "__main__":
    main()
