import cv2
import numpy as np
import time
import sys
import os
import argparse
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cv_trackers import EuclideanTracker, KalmanMultiTracker
except ImportError:
    print(
        "[WARN] cv_trackers not found. Tracking will be disabled or error out if selected."
    )


class ArmorDetector:
    def __init__(self):
        self.light_min_ratio = 3.0
        self.match_max_angle_diff = 10.0
        self.armor_min_ratio = 0.6
        self.armor_max_ratio = 5.0
        self.armor_max_y_diff_ratio = 0.5
        self.light_max_tilt = 40.0

    def preprocess(self, img, enemy_color, color_th=55, bright_th=200):
        # Extract channels
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # For native MP4, doing simple subtraction
        if enemy_color == "red":
            diff = cv2.subtract(r, b)
        else:
            diff = cv2.subtract(b, r)

        _, color_mask = cv2.threshold(diff, color_th, 255, cv2.THRESH_BINARY)
        _, bright_mask = cv2.threshold(gray, bright_th, 255, cv2.THRESH_BINARY)

        # Saturated white cores have R=255, B=255, so diff=0.
        # Dilate the color mask slightly to bridge over the pure white core.
        kernel_diff = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        color_mask = cv2.dilate(color_mask, kernel_diff, iterations=1)

        # Keep only the brightly lit areas (core) that are surrounded by the target color halo
        binary = cv2.bitwise_and(color_mask, bright_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        binary = cv2.dilate(binary, kernel, iterations=1)
        return binary

    def find_targets(self, binary, min_area=10):
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
            if width == 0 or (length / width) < self.light_min_ratio:
                continue

            # Calculate angle of the longest side relative to the x-axis [0, 180)
            theta = (angle if w >= h else angle + 90) % 180
            tilt_angle = abs(theta - 90)

            # Filter out horizontal bars
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
                    # Confidence Score metrics
                    # 1. Length similarity (front facing bars look identical in length)
                    len_sim = 1.0 - (
                        abs(l1["length"] - l2["length"])
                        / max(l1["length"], l2["length"])
                    )
                    # 2. Y-coordinate alignment (front facing bars are perfectly horizontal to each other)
                    y_sim = 1.0 - min(
                        abs(l1["center"][1] - l2["center"][1]) / avg_len, 1.0
                    )
                    # 3. Angle alignment
                    angle_sim = 1.0 - (angle_diff / self.match_max_angle_diff)

                    # Base geometric similarity (max 3.0)
                    base_sim = len_sim + y_sim + angle_sim

                    # Strongly prioritize targets that are closer (large size/length).
                    # Dividing by 20 so a closer target with length=100 adds +5.0,
                    # destroying a far target (length=20) which only gets +1.0 even if perfectly facing.
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
        return lights, armors


def main():
    parser = argparse.ArgumentParser(description="Test video qdetection logic on MP4")
    parser.add_argument(
        "video_path",
        type=str,
        nargs="?",
        default="cv2/videos/",
        help="Path to the MP4 file or directory containing videos",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="blue",
        choices=["red", "blue"],
        help="Target enemy color to track ('red' or 'blue')",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="euclidean",
        choices=["euclidean", "kalman", "none"],
        help="Tracker type to benchmark",
    )
    args = parser.parse_args()

    video_path = args.video_path

    mp4_files = []
    # If a directory is provided, find all mp4 files
    if os.path.isdir(video_path):
        mp4_files = glob.glob(os.path.join(video_path, "*.mp4"))
        if not mp4_files:
            print(f"[ERROR] No .mp4 files found in directory: {video_path}")
            return
        mp4_files.sort()
    else:
        mp4_files = [video_path]

    detector = ArmorDetector()
    current_target = args.target

    win_name = "Video Testing"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 960, 540)

    cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Binary", 960, 540)

    # Trackbar for threshold
    cv2.createTrackbar("ColorThresh", win_name, 35, 255, lambda x: None)
    cv2.createTrackbar("BrightThresh", win_name, 205, 255, lambda x: None)

    print(f"[INFO] Found {len(mp4_files)} videos.")
    print("[INFO] Controls: 'q' to quit all, 'n' to next video, 'SPACE' to pause/play")
    print("[INFO] Target: 'r' for Red, 'b' for Blue")

    quit_all = False

    for video_file in mp4_files:
        if quit_all:
            break

        if not os.path.exists(video_file):
            print(f"[ERROR] Video file not found: {video_file}")
            continue

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video: {video_file}")
            continue

        tracker = None
        if args.tracker == "euclidean":
            tracker = EuclideanTracker()
        elif args.tracker == "kalman":
            tracker = KalmanMultiTracker()

        paused = False
        prev_time = time.time()
        tracked_targets = {}
        dynamic_min_area = 10
        current_best_tid = None

        print(f"\n[INFO] Playing: {video_file} | Target: {current_target}")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] End of video.")
                    break

            # We can dynamically change threshold
            cth = cv2.getTrackbarPos("ColorThresh", win_name)
            bth = cv2.getTrackbarPos("BrightThresh", win_name)

            # CV Pipeline Execution
            binary = detector.preprocess(frame, current_target, cth, bth)
            lights, armors = detector.find_targets(binary, min_area=dynamic_min_area)

            armor_centers = [a["center"] for a in armors]

            if not paused:
                if tracker:
                    tracked_targets = tracker.update(armor_centers)
                else:
                    tracked_targets = {
                        f"Raw-{i}": c for i, c in enumerate(armor_centers)
                    }
            else:
                # When paused, don't update tracker (avoids filter going crazy)
                # Just show raw detections for tuning
                tracked_targets = {f"Tune-{i}": c for i, c in enumerate(armor_centers)}

            # Evaluate targets to find the best one, applying stickiness (loyalty bonus)
            display_best_score = 0.0
            if tracked_targets and armors:
                tid_scores = {}
                for tid, t_center in tracked_targets.items():
                    # Find score of the closest raw armor detection
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
                        assigned_score += 5.0  # Massive stickiness bonus

                    tid_scores[tid] = {
                        "score": assigned_score,
                        "raw": raw_score,
                        "armor": mapped_armor,
                    }

                # Pick the ID with the highest score
                current_best_tid = max(tid_scores, key=lambda k: tid_scores[k]["score"])
                display_best_score = tid_scores[current_best_tid]["raw"]

                # Update dynamic contour minimum area based on best armor
                if not paused:
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
                if not paused:
                    dynamic_min_area = 10

            # FPS Calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Drawing Results
            vis = frame.copy()

            # Draw lights
            for l in lights:
                cv2.drawContours(vis, [l["box"]], -1, (0, 255, 0), 1)

            # Draw tracked targets with IDs
            for tid, center in tracked_targets.items():
                is_best = tid == current_best_tid

                if is_best:
                    color = (255, 0, 255)  # Magenta for highest confidence
                    cv2.drawMarker(vis, center, color, cv2.MARKER_CROSS, 12, 2)
                    cv2.circle(vis, center, 8, color, 2)
                    cv2.putText(
                        vis,
                        f"BEST ID:{tid} ({display_best_score:.2f})",
                        (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                else:
                    color = (0, 0, 255)  # Regular Red
                    cv2.circle(vis, center, 5, color, -1)
                    cv2.putText(
                        vis,
                        f"ID:{tid}",
                        (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                    )

            # Draw UI
            cv2.putText(
                vis,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                f"Targets: {len(tracked_targets)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            target_color_display = (
                (255, 0, 0) if current_target == "blue" else (0, 0, 255)
            )
            cv2.putText(
                vis,
                f"TARGET: {current_target.upper()}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                target_color_display,
                2,
            )
            if paused:
                cv2.putText(
                    vis,
                    "PAUSED",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # Display windows
            cv2.imshow(win_name, vis)
            cv2.imshow("Binary", binary)

            # Delay adjusted based on pause state
            delay = (
                0 if paused else max(1, int(1000 / 30.0))
            )  # Assuming max 30 fps playback speed roughly
            key = cv2.waitKey(delay) & 0xFF

            if key == ord("q"):
                quit_all = True
                break
            elif key == ord("n"):
                break  # breaks inner loop, proceeds to next video
            elif key == ord(" "):
                paused = not paused
            elif key == ord("r"):
                current_target = "red"
                print("[INFO] Switched target to RED")
            elif key == ord("b"):
                current_target = "blue"
                print("[INFO] Switched target to BLUE")

        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
