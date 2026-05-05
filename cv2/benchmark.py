import cv2
import numpy as np
from pypylon import pylon
import time
import sys
import os
import cProfile
import pstats
import io
import threading
import queue

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cv_trackers import EuclideanTracker, KalmanMultiTracker

import argparse

# --- DEFAULT CONFIGURATION FOR FALLBACK ---
# Default if no arguments are passed
TRACKER_TYPE = "kalman"


class ThreadedCamera:
    def __init__(self, camera, converter):
        self.camera = camera
        self.converter = converter
        self.q = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running and self.camera.IsGrabbing():
            try:
                res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if res.GrabSucceeded():
                    img = self.converter.Convert(res).GetArray()
                    if self.q.full():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                    self.q.put(img)
                res.Release()
            except Exception as e:
                pass

    def get_frame(self):
        try:
            return self.q.get(timeout=2.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self.thread.join()

class ArmorDetector:
    def __init__(self):
        # Constants from your golden parameters
        self.light_min_ratio = 3.0
        self.match_max_angle_diff = 10.0
        self.armor_min_ratio = 0.6
        self.armor_max_ratio = 5.0
        self.armor_max_y_diff_ratio = 0.5
        self.light_max_tilt = 40.0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))

    def fast_process(self, img, enemy_color, color_th=55):
        # Extract contiguous arrays for NEON SIMD acceleration
        if enemy_color == "red":
            primary = cv2.extractChannel(img, 2)
            secondary = cv2.extractChannel(img, 0)
        else:
            primary = cv2.extractChannel(img, 0)
            secondary = cv2.extractChannel(img, 2)

        diff = cv2.subtract(primary, secondary)

        _, binary = cv2.threshold(diff, color_th, 255, cv2.THRESH_BINARY)

        # Increased dilation to reconnect fragmented components of the target LED
        # using a taller rect since armor lights are vertical
        return cv2.dilate(binary, self.kernel, iterations=1)

    def get_armor_centers(self, binary, min_area=10):
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        lights = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 2 or h < 4 or w > h * 2:
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

        lights.sort(key=lambda l: l["center"][0])
        armors = []
        for i in range(len(lights)):
            for j in range(i + 1, len(lights)):
                l1, l2 = lights[i], lights[j]

                dist_x = l2["center"][0] - l1["center"][0]
                if dist_x > 300:
                    break

                angle_diff = abs(l1["angle"] - l2["angle"])
                if angle_diff > self.match_max_angle_diff:
                    continue

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
    # 1. Hardware Initialization (One-time cost)
    try:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        node = camera.GetNodeMap()

        # Apply your tested Golden Parameters
        node.GetNode("ExposureAuto").FromString("Off")
        node.GetNode("GainAuto").FromString("Off")
        node.GetNode("BalanceWhiteAuto").FromString("Off")
        node.GetNode("ExposureTime").SetValue(723.0)  # Your verified value
        node.GetNode("Gain").SetValue(0.0)
        node.GetNode("Gamma").SetValue(0.9)
        # Un-cap the hardware frame rate limit!
        try:
            node.GetNode("AcquisitionFrameRateEnable").SetValue(False)
        except:
            try:
                node.GetNode("AcquisitionFrameRateEnabled").SetValue(False)
            except:
                pass

        # Hardware Color Enhancement
        try:
            node.GetNode("ColorAdjustmentSelector").SetValue("Red")
            node.GetNode("ColorAdjustmentSaturation").SetValue(2.0)
            node.GetNode("ColorAdjustmentSelector").SetValue("Blue")
            node.GetNode("ColorAdjustmentSaturation").SetValue(2.0)
        except:
            pass

        # Set Inverse WB based on Target selection
        if args.target == "blue":
            node.GetNode("BalanceRatioSelector").SetValue("Red")
            node.GetNode("BalanceRatio").SetValue(3.5)
            node.GetNode("BalanceRatioSelector").SetValue("Green")
            node.GetNode("BalanceRatio").SetValue(1.0)
            node.GetNode("BalanceRatioSelector").SetValue("Blue")
            node.GetNode("BalanceRatio").SetValue(0.9)
        else:
            # Default to Red target tuned parameters (Math-free R-B compensation presets)
            node.GetNode("BalanceRatioSelector").SetValue("Red")
            node.GetNode("BalanceRatio").SetValue(0.8)
            node.GetNode("BalanceRatioSelector").SetValue("Green")
            node.GetNode("BalanceRatio").SetValue(0.8)
            node.GetNode("BalanceRatioSelector").SetValue("Blue")
            node.GetNode("BalanceRatio").SetValue(7.0)

        # Optimize Throughput for Orin
        try:
            node.GetNode("DeviceLinkThroughputLimit").SetValue(380000000)
        except:
            pass

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed

        print(f"[INIT] Camera {camera.GetDeviceInfo().GetModelName()} ready.")
    except Exception as e:
        print(f"[FATAL] Camera init error: {e}")
        return

    detector = ArmorDetector()
    tracker = (
        EuclideanTracker()
        if args.tracker == "euclidean"
        else (KalmanMultiTracker() if args.tracker == "kalman" else None)
    )
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    last_time = time.time()
    frame_count = 0
    fps = 0

    dynamic_min_area = 10
    current_best_tid = None

    print("[RUNNING] Press 'Q' to stop. Output: [X, Y] coordinates")

    pr = cProfile.Profile()
    pr.enable()
    
    threaded_cam = ThreadedCamera(camera, converter)

    try:
        while camera.IsGrabbing():
            img = threaded_cam.get_frame()
            if img is not None:
                # 2. Frame Capture (done in background)

                # Camera is physically mounted upside down, but we skip cv2.flip
                # to save ~4ms per frame. Detection works fine upside down.

                # 3. Algorithm Execution
                binary = detector.fast_process(img, args.target)
                armors = detector.get_armor_centers(binary, min_area=dynamic_min_area)

                # Math flip for coordinates only
                h, w = img.shape[:2]
                armor_coords = []
                for a in armors:
                    cx, cy = a["center"]
                    armor_coords.append((w - cx, h - cy))

                if tracker:
                    tracked_targets = tracker.update(armor_coords)
                else:
                    tracked_targets = {
                        f"Raw-{i}": c for i, c in enumerate(armor_coords)
                    }

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

                    current_best_tid = max(
                        tid_scores, key=lambda k: tid_scores[k]["score"]
                    )
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

                    # 5. Data Output (Simulating competition data stream)
                    if best_center:
                        print(
                            f"FPS: {fps:5.1f} | Best Target [{current_best_tid}]: ({best_center[0]:4d}, {best_center[1]:4d}) Score: {display_best_score:.2f}"
                        )
                    else:
                        print(f"FPS: {fps:5.1f} | Searching...")

                # Optional Visualization (Disable for absolute max performance)
                if not args.no_gui:
                    cv2.imshow("Bench", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

    except KeyboardInterrupt:
        print("[INFO] Terminated by user.")
    finally:
        pr.disable()
        threaded_cam.stop()
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()

        # Save profile
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats(50)

        prof_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "profile_output.txt"
        )
        with open(prof_path, "w") as f:
            f.write(s.getvalue())
        print(f"[INFO] Profiling results saved to {prof_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basler Headless Hardware Benchmark")
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="red",
        choices=["red", "blue"],
        help="Target enemy color to track ('red' or 'blue')",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=TRACKER_TYPE,
        choices=["euclidean", "kalman", "none"],
        help="Tracker type to benchmark ('euclidean', 'kalman', 'none')",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable all cv2.imshow GUI elements for maximum performance",
    )
    args = parser.parse_args()

    main(args)
