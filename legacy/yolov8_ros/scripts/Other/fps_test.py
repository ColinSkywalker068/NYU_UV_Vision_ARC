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
        default="/home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb2.engine",
        help="Absolute path to YOLO model (.pt/.engine).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="obb",
        choices=["detect", "obb"],
        help="YOLO task type to use.",
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
    task: str,
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

    model = YOLO(model_path, task=task)

    def enumerate_color_profiles() -> list[dict]:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found.")
        device = devices[0]
        color_sensor = None
        for s in device.query_sensors():
            try:
                name = s.get_info(rs.camera_info.name)
            except Exception:
                name = ""
            if "RGB" in name or "Color" in name:
                color_sensor = s
                break
        if color_sensor is None:
            raise RuntimeError("No RGB/Color sensor found on RealSense device.")
        profiles = []
        for p in color_sensor.get_stream_profiles():
            try:
                if p.stream_type() != rs.stream.color:
                    continue
                vsp = rs.video_stream_profile(p)
                profiles.append(
                    {
                        "width": vsp.width(),
                        "height": vsp.height(),
                        "fps": p.fps(),
                        "format": p.format(),
                    }
                )
            except Exception:
                continue
        return profiles

    def select_profile(
        all_profiles: list[dict], req_w: int, req_h: int, req_fps: int
    ) -> dict | None:
        if not all_profiles:
            return None
        prefer_formats = [rs.format.bgr8, rs.format.rgb8]
        same_res = [
            pr
            for pr in all_profiles
            if pr["width"] == req_w
            and pr["height"] == req_h
            and pr["format"] in prefer_formats
        ]
        if same_res:
            same_res.sort(
                key=lambda pr: (
                    abs(pr["fps"] - req_fps),
                    prefer_formats.index(pr["format"]),
                )
            )
            return same_res[0]
        # Fallback: closest aspect ratio, then nearest fps
        def aspect_ratio(w: int, h: int) -> float:
            return w / float(h) if h else 0.0

        req_ar = aspect_ratio(req_w, req_h)
        candidates = [pr for pr in all_profiles if pr["format"] in prefer_formats]
        if not candidates:
            candidates = all_profiles
        candidates.sort(
            key=lambda pr: (
                abs(aspect_ratio(pr["width"], pr["height"]) - req_ar),
                abs(pr["fps"] - req_fps),
            )
        )
        return candidates[0] if candidates else None

    pipeline = rs.pipeline()
    config = rs.config()
    selected_format = rs.format.bgr8
    try:
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, stream_fps)
        profile = pipeline.start(config)
        selected_format = rs.format.bgr8
        print(f"Selected Color Stream: {width}x{height} @ {stream_fps} (BGR8)")
    except Exception:
        pipeline = rs.pipeline()  # fresh pipeline
        config = rs.config()
        profiles = enumerate_color_profiles()
        chosen = select_profile(profiles, width, height, stream_fps)
        if chosen is None:
            raise RuntimeError("Unable to find a valid color stream profile.")
        config.enable_stream(
            rs.stream.color,
            chosen["width"],
            chosen["height"],
            chosen["format"],
            chosen["fps"],
        )
        profile = pipeline.start(config)
        selected_format = chosen["format"]
        fmt_name = (
            "BGR8"
            if selected_format == rs.format.bgr8
            else (
                "RGB8"
                if selected_format == rs.format.rgb8
                else str(int(selected_format))
            )
        )
        print(
            f"Selected Color Stream: {chosen['width']}x{chosen['height']} @ {chosen['fps']} ({fmt_name})"
        )

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
            if selected_format == rs.format.rgb8:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            t0 = time.perf_counter()
            _ = model(frame, verbose=False)
            infer_s = time.perf_counter() - t0
            print(
                f"Inference Time: {infer_s*1000:.2f} ms | Theoretical Max FPS: {1/infer_s:.2f}"
            )
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
        task=args.task,
        width=args.width,
        height=args.height,
        stream_fps=args.fps,
        warmup_frames=args.warmup,
        show=args.show,
        duration_s=args.duration,
        max_frames=args.max_frames,
    )
