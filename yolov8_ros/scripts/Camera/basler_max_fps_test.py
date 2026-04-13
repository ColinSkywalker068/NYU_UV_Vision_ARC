#!/usr/bin/env python3
"""Basler max-FPS benchmark — no YOLO, pure grab-to-BGR throughput.

Measures two metrics:
    Stream FPS  – how fast we receive frames from the sensor
    Proc FPS    – how fast we convert each frame to BGR8

Defaults are tuned for maximum frame rate:
    - Low exposure (2000 µs) so the sensor readout doesn't bottleneck
    - Low gain (5 dB) for acceptable image quality
    - BayerRG8 pixel format with pypylon converter (fastest path on Ace2)
    - AcquisitionFrameRate disabled (uncapped)
    - GrabStrategy_LatestImageOnly to avoid queue backpressure
"""

import argparse
import signal
import sys
import time
from typing import Optional

import cv2
import numpy as np
from pypylon import pylon

PREVIEW_MAX_WIDTH = 960
PREVIEW_MAX_HEIGHT = 540
HEADLESS_LOG_SEC = 1.0

# ── Globals for clean shutdown ──────────────────────────────────────────
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


# ── Helpers ─────────────────────────────────────────────────────────────
def tick_fps_ema(prev: float, dt: float) -> float:
    inst = 1.0 / max(dt, 1e-9)
    if prev <= 0.0:
        return inst
    return 0.9 * prev + 0.1 * inst


def frame_for_preview(frame: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return frame
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def _set_if_writable(node, value) -> bool:
    if node is None:
        return False
    try:
        if hasattr(node, "IsWritable") and not node.IsWritable():
            return False
        node.SetValue(value)
        return True
    except Exception:
        return False


def _align_to_increment(node, value: int) -> int:
    try:
        lo = int(node.GetMin())
        hi = int(node.GetMax())
        inc = int(node.GetInc()) if hasattr(node, "GetInc") else 1
    except Exception:
        return int(value)
    clamped = max(lo, min(hi, int(value)))
    if inc <= 1:
        return clamped
    return lo + ((clamped - lo) // inc) * inc


# ── Diagnostics ─────────────────────────────────────────────────────────
def print_camera_info(camera):
    """Print model, serial, available pixel formats, sensor limits."""
    info = camera.GetDeviceInfo()
    print(f"Model:  {info.GetModelName()}")
    try:
        print(f"Serial: {info.GetSerialNumber()}")
    except Exception:
        pass

    max_w = camera.WidthMax.GetValue()
    max_h = camera.HeightMax.GetValue()
    print(f"Sensor: {max_w} x {max_h}")

    try:
        symbolics = list(camera.PixelFormat.GetSymbolics())
        print(f"Pixel formats: {', '.join(symbolics)}")
    except Exception:
        print("Pixel formats: (could not enumerate)")

    try:
        exp_node = camera.ExposureTime
        print(f"Exposure range: {exp_node.GetMin():.0f} – {exp_node.GetMax():.0f} µs")
    except Exception:
        pass

    try:
        gain_node = camera.Gain
        print(f"Gain range: {gain_node.GetMin():.1f} – {gain_node.GetMax():.1f} dB")
    except Exception:
        pass


# ── Camera configuration ────────────────────────────────────────────────
def configure_camera(
    camera,
    *,
    fps_cap: int,
    exposure_us: float,
    gain_db: float,
    pixel_format: Optional[str],
):
    """Configure the camera for maximum throughput.

    Always uses full sensor resolution for maximum field of view.
    The caller resizes the frame in software after conversion.
    Returns dict with actual applied width, height, fps, pixel_format.
    """
    # Reset offsets, use full sensor
    _set_if_writable(camera.OffsetX, 0)
    _set_if_writable(camera.OffsetY, 0)

    max_w = camera.WidthMax.GetValue()
    max_h = camera.HeightMax.GetValue()

    _set_if_writable(camera.Width, max_w)
    _set_if_writable(camera.Height, max_h)

    # Pixel format — prefer BayerRG8 for speed, fall back to what's available
    chosen_fmt = None
    if pixel_format:
        if _set_if_writable(camera.PixelFormat, pixel_format):
            chosen_fmt = pixel_format
    if chosen_fmt is None:
        for fmt in ("BayerRG8", "BayerBG8", "BayerGB8", "BayerGR8", "RGB8", "BGR8"):
            if _set_if_writable(camera.PixelFormat, fmt):
                chosen_fmt = fmt
                break
    try:
        chosen_fmt = camera.PixelFormat.GetValue()
    except Exception:
        pass

    # Exposure — manual, low for max FPS
    _set_if_writable(camera.ExposureAuto, "Off")
    try:
        lo = float(camera.ExposureTime.GetMin())
        hi = float(camera.ExposureTime.GetMax())
        clamped = max(lo, min(hi, float(exposure_us)))
        _set_if_writable(camera.ExposureTime, clamped)
    except Exception:
        pass

    # Gain — manual
    _set_if_writable(camera.GainAuto, "Off")
    _set_if_writable(camera.Gain, float(gain_db))

    # FPS cap
    if fps_cap > 0:
        try:
            if hasattr(camera, "AcquisitionFrameRateEnable"):
                camera.AcquisitionFrameRateEnable.SetValue(True)
            if hasattr(camera, "AcquisitionFrameRate"):
                camera.AcquisitionFrameRate.SetValue(float(fps_cap))
        except Exception:
            pass
    else:
        # Uncapped — let the sensor run as fast as it can
        try:
            if hasattr(camera, "AcquisitionFrameRateEnable"):
                camera.AcquisitionFrameRateEnable.SetValue(False)
        except Exception:
            pass

    # Transport tuning
    try:
        sg = camera.GetStreamGrabberParams()
        try:
            sg.MaxTransferSize.SetValue(65536)
        except Exception:
            pass
        try:
            sg.NumMaxQueuedUrbs.SetValue(32)
        except Exception:
            pass
    except Exception:
        pass
    try:
        camera.MaxNumBuffer.SetValue(32)
    except Exception:
        pass

    # Read back actuals
    actual_w = int(camera.Width.GetValue())
    actual_h = int(camera.Height.GetValue())
    try:
        actual_fps = float(camera.ResultingFrameRate.GetValue())
    except Exception:
        actual_fps = 0.0

    return {
        "width": actual_w,
        "height": actual_h,
        "fps": actual_fps,
        "pixel_format": chosen_fmt,
    }


# ── Main loop ───────────────────────────────────────────────────────────
def run(
    output_width: int,
    output_height: int,
    fps_cap: int,
    exposure_us: float,
    gain_db: float,
    pixel_format: Optional[str],
    headless: bool,
) -> int:
    global _shutdown

    tl = pylon.TlFactory.GetInstance()
    devices = tl.EnumerateDevices()
    if not devices:
        print("ERROR: No Basler camera found.", file=sys.stderr)
        return 1

    camera = pylon.InstantCamera(tl.CreateDevice(devices[0]))
    camera.Open()

    print("=" * 50)
    print_camera_info(camera)
    print("=" * 50)

    actuals = configure_camera(
        camera,
        fps_cap=fps_cap,
        exposure_us=exposure_us,
        gain_db=gain_db,
        pixel_format=pixel_format,
    )

    print(
        f"\nSensor stream: {actuals['width']}x{actuals['height']} "
        f"@ {actuals['fps']:.1f} FPS (camera ceiling)  "
        f"format={actuals['pixel_format']}",
        flush=True,
    )
    print(f"Output resize: {output_width}x{output_height}")
    print(f"Exposure: {exposure_us:.0f} µs | Gain: {gain_db:.1f} dB")
    if fps_cap > 0:
        print(f"FPS cap: {fps_cap}")
    else:
        print("FPS cap: UNCAPPED")
    print("-" * 50, flush=True)

    # Build BGR converter (needed for Bayer/Mono formats)
    native_bgr = actuals["pixel_format"] in ("BGR8",)
    if native_bgr:
        converter = None
        print("Pixel path: native BGR8 (no conversion)")
    else:
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_LsbAligned
        print(f"Pixel path: {actuals['pixel_format']} → BGR8 via pypylon converter")

    # Start grabbing
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Warmup — discard first few frames
    for _ in range(5):
        try:
            grab = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if grab is not None:
                grab.Release()
        except Exception:
            pass

    proc_fps_ema = 0.0
    stream_fps_ema = 0.0
    prev_grab_end = None
    prev_log = time.monotonic()
    total_frames = 0
    t_start = time.monotonic()
    window_name = "Basler Max FPS Test"

    try:
        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while not _shutdown:
            # --- Grab frame ---
            try:
                grab = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            except Exception:
                continue
            if grab is None or not grab.GrabSucceeded():
                if grab is not None:
                    grab.Release()
                continue

            t_grab = time.monotonic()

            # --- Copy data out of pylon buffer immediately ---
            t_proc0 = time.monotonic()
            if native_bgr:
                frame = np.array(grab.GetArray(), copy=True)
            else:
                converted = converter.Convert(grab)
                frame = np.array(converted.GetArray(), copy=True)
            grab.Release()
            # --- pylon buffer fully released ---

            # Resize from full sensor to target output dimensions
            fh, fw = frame.shape[:2]
            if fw != output_width or fh != output_height:
                frame = cv2.resize(
                    frame, (output_width, output_height), interpolation=cv2.INTER_AREA
                )

            proc_fps_ema = tick_fps_ema(
                proc_fps_ema, max(time.monotonic() - t_proc0, 1e-9)
            )

            if prev_grab_end is not None:
                stream_fps_ema = tick_fps_ema(
                    stream_fps_ema, max(t_grab - prev_grab_end, 1e-9)
                )
            prev_grab_end = t_grab
            total_frames += 1

            if headless:
                now = time.monotonic()
                if now - prev_log >= HEADLESS_LOG_SEC:
                    elapsed = now - t_start
                    avg = total_frames / max(elapsed, 1e-9)
                    print(
                        f"Proc FPS {proc_fps_ema:.1f}  "
                        f"Stream FPS {stream_fps_ema:.1f}  "
                        f"Avg {avg:.1f}  "
                        f"Frames {total_frames}",
                        flush=True,
                    )
                    prev_log = now
                continue

            # --- Display ---
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
            cv2.putText(
                frame,
                f"Exposure: {exposure_us:.0f}us  Gain: {gain_db:.1f}dB",
                (8, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            view = frame_for_preview(frame, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)
            cv2.imshow(window_name, view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        elapsed = time.monotonic() - t_start
        avg_fps = total_frames / max(elapsed, 1e-9)
        print(f"\n{'=' * 50}")
        print(
            f"Summary: {total_frames} frames in {elapsed:.1f}s → {avg_fps:.1f} avg FPS"
        )
        print(f"Final Stream FPS: {stream_fps_ema:.1f}")
        print(f"Final Proc FPS:   {proc_fps_ema:.1f}")
        print(f"{'=' * 50}")

        # Clean shutdown: stop grab → close → destroy
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception:
            pass
        try:
            if camera.IsOpen():
                camera.Close()
        except Exception:
            pass
        try:
            camera.DestroyDevice()
        except Exception:
            pass
        if not headless:
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(
        description="Basler max-FPS benchmark — no YOLO, pure grab-to-BGR throughput."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Output frame width after resize (default: 640). Sensor always runs full res.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output frame height after resize (default: 480). Sensor always runs full res.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=0,
        help="FPS cap (default: 0 = uncapped, let sensor run max).",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        default=2000.0,
        help="Exposure time in µs (default: 2000 — low for max FPS).",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=5.0,
        help="Gain in dB (default: 5.0).",
    )
    parser.add_argument(
        "--pixel-format",
        type=str,
        default=None,
        help="Force a specific pixel format (e.g. BayerRG8, BGR8, Mono8). Default: auto-select.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Print FPS to stdout instead of showing a window.",
    )
    args = parser.parse_args()

    return run(
        output_width=args.width,
        output_height=args.height,
        fps_cap=args.fps,
        exposure_us=args.exposure,
        gain_db=args.gain,
        pixel_format=args.pixel_format,
        headless=args.headless,
    )


if __name__ == "__main__":
    sys.exit(main())
