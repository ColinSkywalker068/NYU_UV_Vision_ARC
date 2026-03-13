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

    def fast_process(self, img, enemy_color, color_th=55):
        # Extract channels (zero-copy references)
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        if enemy_color == "red":
            # Yellow light has high Red + high Green. Red LED has low Green.
            # Subtract Blue entirely, and subtract half of Green to punish yellow.
            penalty = cv2.addWeighted(b, 1.0, g, 0.5, 0)
            diff = cv2.subtract(r, penalty)
        else:
            # Cyan light has high Blue + high Green. Blue LED has low Green.
            penalty = cv2.addWeighted(r, 1.0, g, 0.5, 0)
            diff = cv2.subtract(b, penalty)

        _, binary = cv2.threshold(diff, color_th, 255, cv2.THRESH_BINARY)

        # Increased dilation to reconnect fragmented components of the target LED
        # using a taller rect since armor lights are vertical
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        return cv2.dilate(binary, kernel, iterations=1)

    def get_armor_centers(self, binary):
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        lights = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            length, width = (w, h) if w > h else (h, w)
            if width > 0 and (length / width) >= self.light_min_ratio:
                # Normalize angle
                draw_angle = angle + 90 if w > h else angle
                if w <= h:
                    draw_angle = angle if angle < -45 else angle + 90
                lights.append(
                    {"center": (cx, cy), "length": length, "angle": draw_angle}
                )

        armors = []
        for i in range(len(lights)):
            for j in range(i + 1, len(lights)):
                l1, l2 = lights[i], lights[j]
                if abs(l1["angle"] - l2["angle"]) <= self.match_max_angle_diff:
                    dist_x = abs(l1["center"][0] - l2["center"][0])
                    avg_len = (l1["length"] + l2["length"]) / 2.0
                    ratio = dist_x / avg_len
                    if self.armor_min_ratio < ratio < self.armor_max_ratio:
                        armors.append(
                            (
                                int((l1["center"][0] + l2["center"][0]) / 2),
                                int((l1["center"][1] + l2["center"][1]) / 2),
                            )
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
<<<<<<< Updated upstream

        # Set Inverse WB
        node.GetNode("BalanceRatioSelector").SetValue("Red")
        node.GetNode("BalanceRatio").SetValue(0.5)
        node.GetNode("BalanceRatioSelector").SetValue("Green")
        node.GetNode("BalanceRatio").SetValue(1.6)
        node.GetNode("BalanceRatioSelector").SetValue("Blue")
        node.GetNode("BalanceRatio").SetValue(4.7)
=======
        
        # Set Inverse WB based on Target selection
        if args.target == "blue":
            node.GetNode("BalanceRatioSelector").SetValue("Red")
            node.GetNode("BalanceRatio").SetValue(3.5)
            node.GetNode("BalanceRatioSelector").SetValue("Green")
            node.GetNode("BalanceRatio").SetValue(1.0)
            node.GetNode("BalanceRatioSelector").SetValue("Blue")
            node.GetNode("BalanceRatio").SetValue(0.9)
        else:
            # Default to Red target tuned parameters
            node.GetNode("BalanceRatioSelector").SetValue("Red")
            node.GetNode("BalanceRatio").SetValue(0.5)
            node.GetNode("BalanceRatioSelector").SetValue("Green")
            node.GetNode("BalanceRatio").SetValue(1.6)
            node.GetNode("BalanceRatioSelector").SetValue("Blue")
            node.GetNode("BalanceRatio").SetValue(4.7)
>>>>>>> Stashed changes

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

                # 3. Algorithm Execution
                binary = detector.fast_process(img, args.target)
                armor_coords = detector.get_armor_centers(binary)

                if tracker:
                    tracked_targets = tracker.update(armor_coords)
                else:
                    tracked_targets = {
                        f"Raw-{i}": c for i, c in enumerate(armor_coords)
                    }

                # 4. Performance Benchmarking
                frame_count += 1
                if frame_count % 30 == 0:
                    now = time.time()
                    fps = 30 / (now - last_time)
                    last_time = now

                # 5. Data Output (Simulating competition data stream)
                if tracked_targets:
                    # Format as ID-[X,Y]
                    target_str = ", ".join(
                        [
                            f"ID:{tid}-[{c[0]},{c[1]}]"
                            for tid, c in tracked_targets.items()
                        ]
                    )
                    print(f"FPS: {fps:.1f} | Targets: {target_str}")
                else:
                    print(f"FPS: {fps:.1f} | Searching...")

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
