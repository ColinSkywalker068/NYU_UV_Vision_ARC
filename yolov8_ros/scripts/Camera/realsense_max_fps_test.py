#!/usr/bin/env python3
"""RealSense max-native-FPS preview with split stream/process FPS, no YOLO.

Frame lifecycle:
    Every iteration: wait_for_frames → get_color_frame → copy raw bytes →
    IMMEDIATELY del color, del frames → process only Python-owned memory.
    No SDK frame reference ever survives across loop iterations.
"""

import argparse
import signal
import sys
import time
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs

PREVIEW_MAX_WIDTH = 960
PREVIEW_MAX_HEIGHT = 540
HEADLESS_LOG_SEC = 1.0
FORMAT_MAP = {
    "bgr8": rs.format.bgr8,
    "rgb8": rs.format.rgb8,
    "yuyv": rs.format.yuyv,
    "yuy2": rs.format.yuyv,
    "uyvy": rs.format.uyvy,
}

# Global flag for clean shutdown via signal handler
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    _shutdown = True


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


def list_supported_profiles(ctx, rs_module):
    """List all available color profiles from the first device."""
    devices = list(ctx.query_devices())
    if not devices:
        print("No devices found to list profiles.")
        return

    dev = devices[0]
    print(f"\nDiagnostics for device: {dev.get_info(rs_module.camera_info.name)}")
    try:
        usb_type = dev.get_info(rs_module.camera_info.usb_type_or_serial)
        print(f"USB Connection Type: {usb_type}")
        if "2.1" in usb_type or "2.0" in usb_type:
            print(
                "WARNING: Device is on USB 2.0. High FPS modes (60/90) are likely disabled by hardware."
            )
    except Exception:
        print("Could not determine USB connection type.")

    print("\nSupported COLOR stream profiles:")
    print(f"{'Width':>6} | {'Height':>6} | {'FPS':>4} | {'Format'}")
    print("-" * 35)

    for sensor in dev.query_sensors():
        try:
            if not sensor.is_color_sensor():
                continue
        except Exception:
            continue

        profiles = sensor.get_stream_profiles()
        sorted_profiles = []
        for p in profiles:
            try:
                vp = p.as_video_stream_profile()
                sorted_profiles.append(vp)
            except Exception:
                continue

        sorted_profiles.sort(
            key=lambda x: (x.fps(), x.width(), x.height()), reverse=True
        )

        for vp in sorted_profiles:
            print(f"{vp.width():>6} | {vp.height():>6} | {vp.fps():>4} | {vp.format()}")
    print("-" * 35)


def best_stream_for_format(
    ctx, width: int, height: int, format_name: str
) -> Optional[tuple[int, int, int]]:
    format_value = FORMAT_MAP[format_name]
    devs = list(ctx.query_devices())
    if not devs:
        return None
    best = None
    best_fps = -1
    for sensor in devs[0].query_sensors():
        try:
            if not sensor.is_color_sensor():
                continue
        except Exception:
            continue
        for profile in sensor.get_stream_profiles():
            try:
                vp = profile.as_video_stream_profile()
            except Exception:
                continue
            if vp.format() != format_value:
                continue
            if vp.width() != int(width) or vp.height() != int(height):
                continue
            fps = int(vp.fps())
            if fps > best_fps:
                best_fps = fps
                best = (int(vp.width()), int(vp.height()), fps)
    return best


def convert_raw_to_bgr(
    raw_data: np.ndarray, w: int, h: int, format_name: str
) -> np.ndarray:
    """Convert a raw numpy array (already copied out of SDK memory) to BGR8.

    raw_data is fully Python-owned — no SDK frame references exist anymore
    when this function is called.
    """
    if format_name == "bgr8":
        return raw_data

    if format_name == "rgb8":
        return cv2.cvtColor(raw_data, cv2.COLOR_RGB2BGR)

    if format_name in ("yuyv", "yuy2", "uyvy"):
        # YUYV comes as uint16 from pyrealsense2. Reinterpret as uint8.
        frame_uint8 = raw_data.view(np.uint8).reshape((h, w, 2))
        if format_name == "uyvy":
            return cv2.cvtColor(frame_uint8, cv2.COLOR_YUV2BGR_UYVY)
        else:
            return cv2.cvtColor(frame_uint8, cv2.COLOR_YUV2BGR_YUY2)

    raise ValueError(f"Unsupported preview conversion for format: {format_name}")


def run(
    width: int, height: int, format_name: str, fps: Optional[int], headless: bool
) -> int:
    global _shutdown

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        print("ERROR: No RealSense device found.", file=sys.stderr)
        return 1

    if format_name not in FORMAT_MAP:
        print(f"ERROR: Unsupported format {format_name}", file=sys.stderr)
        return 1

    if fps is not None and fps > 0:
        requested = (int(width), int(height), int(fps))
        request_mode = "exact"
    else:
        requested = best_stream_for_format(ctx, width, height, format_name)
        if requested is None:
            requested = (int(width), int(height), 60)
        request_mode = "best-native"

    print(
        f"Requesting {request_mode} {format_name.upper()} stream: "
        f"{requested[0]}x{requested[1]} @ {requested[2]} FPS",
        flush=True,
    )

    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        requested[0],
        requested[1],
        FORMAT_MAP[format_name],
        requested[2],
    )

    try:
        profile = pipeline.start(config)
    except Exception as exc:
        print(f"ERROR: Failed to start RealSense pipeline: {exc}", file=sys.stderr)
        if "couldn't resolve requests" in str(exc).lower():
            print(
                "\nSUGGESTION: The requested mode is not supported by your hardware or USB port."
            )
            list_supported_profiles(ctx, rs)
        return 1

    try:
        stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        active = (
            int(stream_profile.width()),
            int(stream_profile.height()),
            int(stream_profile.fps()),
        )
        print(
            f"Active {format_name.upper()} stream: {active[0]}x{active[1]} @ {active[2]} FPS",
            flush=True,
        )
    except Exception:
        active = None
        print("Active stream info unavailable.", flush=True)

    proc_fps_ema = 0.0
    stream_fps_ema = 0.0
    prev_grab_end = None
    prev_log = time.monotonic()
    window_name = "RealSense Max FPS Test"

    try:
        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while not _shutdown:
            # --- SDK interaction zone: touch SDK objects, copy bytes, then delete ---
            frames = pipeline.wait_for_frames()
            t_grab = time.monotonic()
            color = frames.get_color_frame()
            if not color:
                del frames
                continue

            # Extract everything we need from the SDK frame
            w = int(color.get_width())
            h = int(color.get_height())
            raw_data = np.asanyarray(color.get_data()).copy()

            # IMMEDIATELY release all SDK frame references.
            # This prevents pyrealsense2's C++ frame pool from double-freeing
            # when the GC collects stale frame objects from previous iterations.
            del color
            del frames
            # --- End SDK interaction zone ---

            if prev_grab_end is not None:
                stream_fps_ema = tick_fps_ema(
                    stream_fps_ema, max(t_grab - prev_grab_end, 1e-9)
                )
            prev_grab_end = t_grab

            t_proc0 = time.monotonic()
            frame = convert_raw_to_bgr(raw_data, w, h, format_name)
            proc_fps_ema = tick_fps_ema(
                proc_fps_ema, max(time.monotonic() - t_proc0, 1e-9)
            )

            if headless:
                now = time.monotonic()
                if now - prev_log >= HEADLESS_LOG_SEC:
                    print(
                        f"Proc FPS {proc_fps_ema:.1f}  Stream FPS {stream_fps_ema:.1f}",
                        flush=True,
                    )
                    prev_log = now
                continue

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
            view = frame_for_preview(frame, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)
            cv2.imshow(window_name, view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        # No SDK frame references exist at this point — safe to stop.
        pipeline.stop()
        if not headless:
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    parser = argparse.ArgumentParser(
        description="RealSense max-native-FPS preview with split stream/process FPS."
    )
    parser.add_argument("--width", type=int, default=640, help="Color stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color stream height.")
    parser.add_argument(
        "--format",
        choices=sorted(FORMAT_MAP.keys()),
        default="yuyv",
        help="Requested RealSense color format (default: yuyv for 60fps).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Exact FPS to request (default: 60). Set to 0 for best native FPS.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Print FPS once per second instead of showing a window.",
    )
    args = parser.parse_args()
    return run(
        width=args.width,
        height=args.height,
        format_name=args.format,
        fps=args.fps,
        headless=args.headless,
    )


if __name__ == "__main__":
    sys.exit(main())
