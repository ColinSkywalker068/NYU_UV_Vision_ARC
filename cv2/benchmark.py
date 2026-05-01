import cv2
import numpy as np
from pypylon import pylon
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cv_trackers import EuclideanTracker, KalmanMultiTracker

import argparse

# --- DEFAULT CONFIGURATION FOR FALLBACK ---
# Default if no arguments are passed
TRACKER_TYPE = "kalman"


class ArmorDetector:
    def __init__(self):
        # Constants from your golden parameters
        self.light_min_ratio = 3.0
        self.match_max_angle_diff = 10.0
        self.armor_min_ratio = 0.6
        self.armor_max_ratio = 5.0
        self.armor_max_y_diff_ratio = 0.5
        self.light_max_tilt = 40.0

    def fast_process(self, img, enemy_color, color_th=55):
        # Extract channels (zero-copy references)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        if enemy_color == "red":
            # Hardware WB Green & Hardware Saturation handle the background.
            # Revert to pure, fast subtraction for 100+ FPS.
            diff = cv2.subtract(r, b)
        else:
            diff = cv2.subtract(b, r)

        _, binary = cv2.threshold(diff, color_th, 255, cv2.THRESH_BINARY)

        # Increased dilation to reconnect fragmented components of the target LED
        # using a taller rect since armor lights are vertical
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
        # Hardware Debayering attempt
        try:
            node.GetNode("PixelFormat").SetValue("BGR8")
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

        # Optimize Throughput for Orin 2
        # node.GetNode("DeviceLinkThroughputLimit").SetValue(250000000)

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

    print("[RUNNING] Press 'Q' to stop. Output: [X, Y] coordinates")

    try:
        while camera.IsGrabbing():
            res = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if res.GrabSucceeded():
                # 2. Frame Capture
                img = converter.Convert(res).GetArray()
                res.Release()

                # Camera is physically mounted upside down, flip 180 degrees
                img = cv2.flip(img, -1)

                # 3. Algorithm Execution
                binary = detector.fast_process(img, args.target)
                armors = detector.get_armor_centers(binary, min_area=dynamic_min_area)
                armor_coords = [a["center"] for a in armors]

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
                cv2.imshow("Bench", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                res.Release()

    except KeyboardInterrupt:
        print("[INFO] Terminated by user.")
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()


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
    args = parser.parse_args()

    main(args)
