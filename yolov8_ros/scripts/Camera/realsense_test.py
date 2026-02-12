#!/usr/bin/env python3
"""
Minimal test script to verify pyrealsense2 and a RealSense camera work.
Run: python realsense_test.py [--show] [--frames N]
"""
import argparse
import sys

import numpy as np
import pyrealsense2 as rs


def run_test(show: bool = False, num_frames: int = 30) -> int:
    """Discover device, start color stream, grab frames. Return 0 on success."""
    print("pyrealsense2: checking context and devices...")
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("ERROR: No RealSense devices found.", file=sys.stderr)
        return 1
    dev = devices[0]
    try:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"  Device: {name} (serial: {serial})")
    except Exception as e:
        print(f"  Device info: {e}")

    print("Starting color stream 640x480 @ 30 FPS...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"ERROR: Failed to start pipeline: {e}", file=sys.stderr)
        return 1

    try:
        for i in range(num_frames):
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                print(f"  Frame {i}: no color frame")
                continue
            arr = np.asanyarray(color_frame.get_data())
            if i == 0:
                print(f"  First frame shape: {arr.shape}, dtype: {arr.dtype}")
            if show and i == num_frames - 1:
                try:
                    import cv2

                    cv2.imshow("RealSense test", arr)
                    print("  Press any key in the window to exit.")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except ImportError:
                    print("  Install opencv-python to use --show")
        print("OK: RealSense and pyrealsense2 are working.")
        return 0
    finally:
        pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test pyrealsense2 and RealSense camera (color stream only)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show last frame in a window (requires opencv-python).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="Number of frames to grab (default: 30).",
    )
    args = parser.parse_args()
    return run_test(show=args.show, num_frames=args.frames)


if __name__ == "__main__":
    sys.exit(main())
