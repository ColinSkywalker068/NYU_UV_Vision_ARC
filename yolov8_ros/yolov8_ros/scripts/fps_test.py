import argparse
import time
import signal
import gc
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Barebones YOLOv8 detection FPS test with Intel RealSense color stream."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb2.pt",
        help="Absolute path to YOLO model (.pt/.engine).",
    )
    parser.add_argument("--width", type=int, default=640, help="Color stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS.")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup frames before measuring."
    )
    parser.add_argument(
        "--show", action="store_true", help="Display color stream window."
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Max seconds to run (0 = unlimited).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = unlimited).",
    )
    return parser.parse_args()


def run_benchmark(
    model_path: str,
    width: int,
    height: int,
    stream_fps: int,
    warmup_frames: int,
    show: bool,
    duration_s: float,
    max_frames: int,
) -> None:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)

    model = YOLO(model_path)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, stream_fps)
    profile = pipeline.start(config)

    try:
        stop_requested = False

        def handle_signal(_sig, _frm):
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        if show:
            cv2.namedWindow("Color", cv2.WINDOW_NORMAL)

        # Warmup
        for _ in range(max(0, warmup_frames)):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data()).copy()
            _ = model(frame, verbose=False)

        # Measurement loop
        window_start_time = time.perf_counter()
        frames_in_window = 0
        infer_time_accum = 0.0
        total_frames = 0
        start_time = time.perf_counter()

        print("Starting detection FPS measurement. Press Ctrl+C to stop.")
        while True:
            if stop_requested:
                break
            if duration_s > 0 and (time.perf_counter() - start_time) >= duration_s:
                break
            if max_frames > 0 and total_frames >= max_frames:
                break
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data()).copy()

            t0 = time.perf_counter()
            _ = model(frame, verbose=False)
            infer_time = time.perf_counter() - t0

            infer_time_accum += infer_time
            frames_in_window += 1
            total_frames += 1

            if show:
                cv2.imshow("Color", frame)
                # Only poll for 'q' when a window is present
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.perf_counter()
            if now - window_start_time >= 1.0 and infer_time_accum > 0:
                det_fps = frames_in_window / infer_time_accum
                print(f"Det FPS: {det_fps:.2f} | Frames: {frames_in_window}")
                window_start_time = now
                frames_in_window = 0
                infer_time_accum = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        # Ensure frame objects are dereferenced before stopping the pipeline
        try:
            frames = None
            color_frame = None
            frame = None
        except NameError:
            pass
        time.sleep(0.05)
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        del pipeline
        del model
        gc.collect()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        model_path=args.model,
        width=args.width,
        height=args.height,
        stream_fps=args.fps,
        warmup_frames=args.warmup,
        show=args.show,
        duration_s=args.duration,
        max_frames=args.max_frames,
    )
