import argparse
import time
import signal
import threading
import queue
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

    stop_event = threading.Event()
    frame_queue: "queue.Queue[rs.video_frame]" = queue.Queue(maxsize=1)

    def handle_signal(_sig, _frm):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if show:
        cv2.namedWindow("Color", cv2.WINDOW_NORMAL)

    def put_latest(q: "queue.Queue[rs.video_frame]", item: rs.video_frame) -> None:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                _ = q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(item)
            except queue.Full:
                pass

    def producer_loop() -> None:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            put_latest(frame_queue, color_frame)

    # Warmup on a few frames to stabilize model/runtime
    for _ in range(max(0, warmup_frames)):
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        warmup_img = np.asanyarray(color_frame.get_data())
        _ = model(warmup_img, verbose=False)

    ewma_alpha = 0.2
    ewma_infer_s: float | None = None
    total_infer_s = 0.0
    processed_frames = 0
    start_time = time.perf_counter()
    last_report = start_time

    def consumer_loop() -> None:
        nonlocal ewma_infer_s, total_infer_s, processed_frames, last_report
        while not stop_event.is_set():
            try:
                color_frame = frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            frame = np.asanyarray(color_frame.get_data())
            t0 = time.perf_counter()
            _ = model(frame, verbose=False)
            infer_s = time.perf_counter() - t0
            total_infer_s += infer_s
            processed_frames += 1
            ewma_infer_s = (
                infer_s
                if ewma_infer_s is None
                else (ewma_alpha * infer_s + (1.0 - ewma_alpha) * ewma_infer_s)
            )
            if show:
                cv2.imshow("Color", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
            now = time.perf_counter()
            if now - last_report >= 1.0:
                elapsed = max(1e-6, now - start_time)
                avg_fps = processed_frames / elapsed
                ewma_fps = (
                    (1.0 / ewma_infer_s) if (ewma_infer_s and ewma_infer_s > 0) else 0.0
                )
                print(
                    f"AVG FPS: {avg_fps:.2f} | EWMA Det FPS: {ewma_fps:.2f} | Frames: {processed_frames}"
                )
                last_report = now
            if max_frames > 0 and processed_frames >= max_frames:
                stop_event.set()
            if duration_s > 0 and (now - start_time) >= duration_s:
                stop_event.set()
            frame_queue.task_done()

    try:
        prod = threading.Thread(target=producer_loop, name="producer", daemon=True)
        cons = threading.Thread(target=consumer_loop, name="consumer", daemon=True)
        prod.start()
        cons.start()
        print("Starting detection FPS measurement. Press Ctrl+C or 'q' to stop.")
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        try:
            prod.join(timeout=2.0)
        except Exception:
            pass
        try:
            cons.join(timeout=2.0)
        except Exception:
            pass
        try:
            pipeline.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


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
