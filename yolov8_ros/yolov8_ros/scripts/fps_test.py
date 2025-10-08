import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs


def parse_args():
    parser = argparse.ArgumentParser(description="Barebones YOLOv8 detection FPS test with Intel RealSense color stream.")
    parser.add_argument(
        "--model",
        type=str,
        default="/Users/Kushal/NYU/CV_Detection/yolov8_ros/models/bestObb2.pt",
        help="Absolute path to YOLO model (.pt/.engine).",
    )
    parser.add_argument("--width", type=int, default=640, help="Color stream width.")
    parser.add_argument("--height", type=int, default=480, help="Color stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup frames before measuring.")
    return parser.parse_args()


def run_benchmark(model_path: str, width: int, height: int, stream_fps: int, warmup_frames: int) -> None:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)

    model = YOLO(model_path)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, stream_fps)
    profile = pipeline.start(config)

    try:
        # Warmup
        for _ in range(max(0, warmup_frames)):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            _ = model(frame, verbose=False)

        # Measurement loop
        window_start_time = time.perf_counter()
        frames_in_window = 0
        infer_time_accum = 0.0

        print("Starting detection FPS measurement. Press Ctrl+C to stop.")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            t0 = time.perf_counter()
            _ = model(frame, verbose=False)
            infer_time = time.perf_counter() - t0

            infer_time_accum += infer_time
            frames_in_window += 1

            now = time.perf_counter()
            if now - window_start_time >= 1.0 and infer_time_accum > 0:
                det_fps = frames_in_window / infer_time_accum
                print(f"Det FPS: {det_fps:.2f} | Frames: {frames_in_window}")
                window_start_time = now
                frames_in_window = 0
                infer_time_accum = 0.0

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        model_path=args.model,
        width=args.width,
        height=args.height,
        stream_fps=args.fps,
        warmup_frames=args.warmup,
    )


