import cv2
import numpy as np
import time
import sys
import os
import argparse

try:
    import pyrealsense2 as rs
except ImportError:
    print("[WARN] pyrealsense2 not installed. Please run `pip install pyrealsense2`")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cv_trackers import EuclideanTracker, KalmanMultiTracker


class ArmorDetector:
    def __init__(self):
        self.light_min_ratio = 3.0
        self.match_max_angle_diff = 10.0
        self.armor_min_ratio = 0.6
        self.armor_max_ratio = 5.0
        self.armor_max_y_diff_ratio = 0.5
        self.light_max_tilt = 40.0

    def fast_process(self, img, enemy_color, color_th=55, penalty_factor=0.0):
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        if penalty_factor > 0:
            if enemy_color == "red":
                penalty = cv2.addWeighted(b, 1.0, g, penalty_factor, 0)
                diff = cv2.subtract(r, penalty)
            else:
                penalty = cv2.addWeighted(r, 1.0, g, penalty_factor, 0)
                diff = cv2.subtract(b, penalty)
        else:
            if enemy_color == "red":
                diff = cv2.subtract(r, b)
            else:
                diff = cv2.subtract(b, r)

        _, binary = cv2.threshold(diff, color_th, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        return cv2.dilate(binary, kernel, iterations=1)

    def get_armor_centers(self, binary, min_area=10):
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        lights = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            length, width = (w, h) if w > h else (h, w)
            if width > 0 and (length / width) >= self.light_min_ratio:
                theta = (angle if w >= h else angle + 90) % 180
                tilt_angle = abs(theta - 90)

                if tilt_angle <= self.light_max_tilt:
                    lights.append(
                        {
                            "center": (cx, cy),
                            "length": length,
                            "angle": theta,
                            "area": area,
                            "box": np.intp(cv2.boxPoints(rect)),
                        }
                    )

        armors = []
        for i in range(len(lights)):
            for j in range(i + 1, len(lights)):
                l1, l2 = lights[i], lights[j]

                angle_diff = abs(l1["angle"] - l2["angle"])
                if angle_diff > self.match_max_angle_diff:
                    continue

                dist_x = abs(l1["center"][0] - l2["center"][0])
                dist_y = abs(l1["center"][1] - l2["center"][1])
                avg_len = (l1["length"] + l2["length"]) / 2.0
                
                if dist_y / avg_len > self.armor_max_y_diff_ratio:
                    continue

                ratio = dist_x / avg_len

                if self.armor_min_ratio < ratio < self.armor_max_ratio:
                    len_sim = 1.0 - (
                        abs(l1["length"] - l2["length"])
                        / max(l1["length"], l2["length"])
                    )
                    y_sim = 1.0 - min(
                        abs(l1["center"][1] - l2["center"][1]) / avg_len, 1.0
                    )
                    angle_sim = 1.0 - (angle_diff / self.match_max_angle_diff)
                    base_sim = len_sim + y_sim + angle_sim
                    score = base_sim + (avg_len / 20.0)

                    armors.append(
                        {
                            "center": (
                                int((l1["center"][0] + l2["center"][0]) / 2),
                                int((l1["center"][1] + l2["center"][1]) / 2),
                            ),
                            "score": score,
                            "lights": (l1, l2),
                        }
                    )
        return armors


def main(args):
    TARGET_COLOR = args.target
    DURATION = args.duration

    # Production RealSense Params
    params = {
        "blue": {
            "exp": 150,
            "color_th": 55,
            "wb": 5500,
            "saturation": 60,
            "contrast": 50,
            "penalty": 0.0,
        },
        "red": {
            "exp": 150,
            "color_th": 55,
            "wb": 3200,
            "saturation": 60,
            "contrast": 50,
            "penalty": 0.0,
        },
    }
    CONFIG = params[TARGET_COLOR]

    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 90)

        profile = pipeline.start(config)
        device = profile.get_device()
        color_sensor = None
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == "RGB Camera":
                color_sensor = s
                break

        if color_sensor:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
            color_sensor.set_option(rs.option.exposure, CONFIG["exp"])
            color_sensor.set_option(rs.option.white_balance, CONFIG["wb"])
            color_sensor.set_option(rs.option.saturation, CONFIG["saturation"])
            color_sensor.set_option(rs.option.contrast, CONFIG["contrast"])

        print(
            f"[INIT] RealSense {device.get_info(rs.camera_info.name)} ready at 90FPS."
        )
    except Exception as e:
        print(f"[FATAL] RealSense init error: {e}")
        return

    detector = ArmorDetector()
    tracker = (
        EuclideanTracker()
        if args.tracker == "euclidean"
        else (KalmanMultiTracker() if args.tracker == "kalman" else None)
    )

    last_time = time.time()
    frame_count = 0
    fps = 0

    dynamic_min_area = 10
    current_best_tid = None

    print(
        f"\n[BENCHMARK] Starting for {DURATION}s | Target: {TARGET_COLOR.upper()} | Tracker: {args.tracker}"
    )
    print("[RUNNING] Press 'Q' to stop. Output: [X, Y] coordinates")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            # 3. Algorithm Execution
            binary = detector.fast_process(
                img,
                args.target,
                color_th=CONFIG["color_th"],
                penalty_factor=CONFIG["penalty"],
            )
            armors = detector.get_armor_centers(binary, min_area=dynamic_min_area)
            armor_coords = [a["center"] for a in armors]

            if tracker:
                tracked_targets = tracker.update(armor_coords)
            else:
                tracked_targets = {f"Raw-{i}": c for i, c in enumerate(armor_coords)}

            display_best_score = 0.0
            if tracked_targets and armors:
                tid_scores = {}
                for tid, t_center in tracked_targets.items():
                    best_dist = float("inf")
                    raw_score = 0.0
                    mapped_armor = None
                    for a in armors:
                        dist = (t_center[0] - a["center"][0]) ** 2 + (
                            t_center[1] - a["center"][1]
                        ) ** 2
                        if dist < best_dist:
                            best_dist = dist
                            raw_score = a["score"]
                            mapped_armor = a

                    assigned_score = raw_score
                    if tid == current_best_tid:
                        assigned_score += 5.0

                    tid_scores[tid] = {
                        "score": assigned_score,
                        "raw": raw_score,
                        "armor": mapped_armor,
                    }

                current_best_tid = max(tid_scores, key=lambda k: tid_scores[k]["score"])
                display_best_score = tid_scores[current_best_tid]["raw"]
                best_center = tracked_targets[current_best_tid]

                best_raw_armor = tid_scores[current_best_tid]["armor"]
                if best_raw_armor:
                    l1, l2 = best_raw_armor["lights"]
                    avg_area = (l1["area"] + l2["area"]) / 2.0
                    dynamic_min_area = max(10, avg_area * 0.5)
                else:
                    dynamic_min_area = 10
            else:
                current_best_tid = None
                display_best_score = 0.0
                best_center = None
                dynamic_min_area = 10

            # 4. Performance Benchmarking
            frame_count += 1
            if frame_count % 30 == 0:
                now = time.time()
                fps = 30 / (now - last_time)
                last_time = now

                # 5. Data Output
                if best_center:
                    print(
                        f"FPS: {fps:5.1f} | Best Target [{current_best_tid}]: ({best_center[0]:4d}, {best_center[1]:4d}) Score: {display_best_score:.2f}"
                    )
                else:
                    print(f"FPS: {fps:5.1f} | Searching...")

            # Optional Visualization (Disable for absolute max performance)
            cv2.imshow("RealSense Bench", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("[INFO] Terminated by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RealSense Headless Hardware Benchmark"
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="red",
        choices=["red", "blue"],
        help="Target color to track ('red' or 'blue')",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="kalman",
        choices=["euclidean", "kalman", "none"],
        help="Tracker type to benchmark",
    )
    parser.add_argument(
        "-d", "--duration", type=int, default=60, help="Benchmark duration in seconds"
    )

    args = parser.parse_args()
    main(args)
