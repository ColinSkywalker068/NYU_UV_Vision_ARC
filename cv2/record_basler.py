import cv2
import numpy as np
from pypylon import pylon
import time
import sys
import os
import threading
import queue

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cv_trackers import EuclideanTracker, KalmanMultiTracker

# --- CONFIGURATION TOGGLE ---
TRACKER_TYPE = "kalman"


class ThreadedVideoWriter:
    def __init__(self, filename, fourcc, fps, frameSize):
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frameSize)
        self.queue = queue.Queue(maxsize=512)
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.running = True
        self.thread.start()

    def _write_loop(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                pass

    def write(self, frame):
        if self.running:
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass

    def release(self):
        self.running = False
        self.thread.join()
        self.writer.release()


class ArmorDetector:
    def __init__(self):
        self.light_min_ratio = 3.0
        self.match_max_angle_diff = 10.0
        self.armor_min_ratio = 0.6
        self.armor_max_ratio = 5.0
        self.armor_max_y_diff_ratio = 0.5
        self.light_max_tilt = 40.0

    def preprocess(self, img, enemy_color, color_th, gamma_val, penalty_factor=0.0):
        if gamma_val != 1.0:
            invGamma = 1.0 / gamma_val
            table = np.array(
                [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
            ).astype("uint8")
            img = cv2.LUT(img, table)

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
        return lights, armors


class BaslerDriver:
    def __init__(self):
        try:
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice()
            )
            self.camera.Open()
            self.nodemap = self.camera.GetNodeMap()

            try:
                self.nodemap.GetNode("PixelFormat").SetValue("BGR8")
            except:
                pass

            try:
                self.nodemap.GetNode("ColorAdjustmentSelector").SetValue("Red")
                self.nodemap.GetNode("ColorAdjustmentSaturation").SetValue(1.5)
                self.nodemap.GetNode("ColorAdjustmentSelector").SetValue("Blue")
                self.nodemap.GetNode("ColorAdjustmentSaturation").SetValue(1.5)
            except:
                pass

            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            print(f"[CAM] Connected: {self.camera.GetDeviceInfo().GetModelName()}")
        except Exception as e:
            print(f"[ERR] Camera failed: {e}")
            sys.exit()

    def set_config(self, exp, gain, wb_r, wb_b, wb_g, gamma):
        try:
            self.nodemap.GetNode("ExposureAuto").FromString("Off")
            self.nodemap.GetNode("GainAuto").FromString("Off")
            self.nodemap.GetNode("BalanceWhiteAuto").FromString("Off")
            self.nodemap.GetNode("ExposureTime").SetValue(float(exp))
            self.nodemap.GetNode("Gain").SetValue(float(gain))
            self.nodemap.GetNode("Gamma").SetValue(float(gamma))

            self.nodemap.GetNode("BalanceRatioSelector").SetValue("Red")
            self.nodemap.GetNode("BalanceRatio").SetValue(float(wb_r))
            self.nodemap.GetNode("BalanceRatioSelector").SetValue("Blue")
            self.nodemap.GetNode("BalanceRatio").SetValue(float(wb_b))
            self.nodemap.GetNode("BalanceRatioSelector").SetValue("Green")
            self.nodemap.GetNode("BalanceRatio").SetValue(float(wb_g))
        except:
            pass

    def get_frame(self):
        if not self.camera.IsGrabbing():
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if res.GrabSucceeded():
            img = self.converter.Convert(res).GetArray()
            res.Release()
            # Camera is physically mounted upside down, flip 180 degrees
            return cv2.flip(img, -1)
        res.Release()
        return None

    def get_temperature(self):
        try:
            try:
                temp = self.nodemap.GetNode("DeviceTemperature").GetValue()
            except:
                temp = self.nodemap.GetNode("BslTemperatureMax").GetValue()

            try:
                state = self.nodemap.GetNode("TemperatureState").ToString()
            except:
                try:
                    state = self.nodemap.GetNode("BslTemperatureStatus").ToString()
                except:
                    state = "Unknown"

            return temp, state
        except:
            return 0.0, "Unknown"


def main():
    cam = BaslerDriver()
    detector = ArmorDetector()
    tracker = (
        EuclideanTracker()
        if TRACKER_TYPE == "euclidean"
        else (KalmanMultiTracker() if TRACKER_TYPE == "kalman" else None)
    )

    params = {
        "blue": {
            "exp": 723,
            "color_th": 55,
            "wb_b": 0.9,
            "wb_g": 1.0,
            "wb_r": 3.5,
            "gamma": 0.9,
            "penalty": 0.0,
        },
        "red": {
            "exp": 723,
            "color_th": 55,
            "wb_b": 7.0,
            "wb_g": 0.8,
            "wb_r": 0.8,
            "gamma": 0.9,
            "penalty": 0.0,
        },
    }
    mode = "blue"

    win_main = "Basler Recorder - Main"
    win_bin = "Basler Recorder - Binary"

    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_bin, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(win_main, 960, 540)
    cv2.resizeWindow(win_bin, 960, 540)

    cv2.createTrackbar("Exposure", win_main, params[mode]["exp"], 2000, lambda x: None)
    cv2.createTrackbar(
        "ColorThresh", win_main, params[mode]["color_th"], 100, lambda x: None
    )
    cv2.createTrackbar(
        "WB Red", win_main, int(params[mode]["wb_r"] * 10), 100, lambda x: None
    )
    cv2.createTrackbar(
        "WB Green", win_main, int(params[mode]["wb_g"] * 10), 100, lambda x: None
    )
    cv2.createTrackbar(
        "WB Blue", win_main, int(params[mode]["wb_b"] * 10), 100, lambda x: None
    )
    cv2.createTrackbar(
        "GrnPenalty", win_main, int(params[mode]["penalty"] * 100), 100, lambda x: None
    )

    prev_time = time.time()
    print("\n[INFO] R: RED Mode | B: BLUE Mode | SPACE: Toggle Record | Q: Quit\n")

    is_recording = False
    video_writer = None
    output_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(output_dir, exist_ok=True)

    dynamic_min_area = 10
    current_best_tid = None
    frame_count = 0

    while True:
        frame_count += 1
        exp = cv2.getTrackbarPos("Exposure", win_main)
        cth = cv2.getTrackbarPos("ColorThresh", win_main)
        wb_r = cv2.getTrackbarPos("WB Red", win_main) / 10.0
        wb_g = cv2.getTrackbarPos("WB Green", win_main) / 10.0
        wb_b = cv2.getTrackbarPos("WB Blue", win_main) / 10.0
        penalty_val = cv2.getTrackbarPos("GrnPenalty", win_main) / 100.0

        cam.set_config(exp, 0, wb_r, wb_b, wb_g, params[mode]["gamma"])

        frame = cam.get_frame()
        if frame is None:
            continue

        # Write RAW frame if recording
        if is_recording and video_writer is not None:
            video_writer.write(frame)

        binary = detector.preprocess(
            frame, mode, cth, params[mode]["gamma"], penalty_factor=penalty_val
        )
        lights, armors = detector.find_targets(binary, min_area=dynamic_min_area)
        armor_centers = [a["center"] for a in armors]

        if tracker:
            tracked_targets = tracker.update(armor_centers)
        else:
            tracked_targets = {f"Raw-{i}": c for i, c in enumerate(armor_centers)}

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
            dynamic_min_area = 10

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        vis = frame.copy()
        for l in lights:
            cv2.drawContours(vis, [l["box"]], -1, (0, 255, 0), 2)

        for tid, center in tracked_targets.items():
            is_best = tid == current_best_tid
            if is_best:
                color = (255, 0, 255)
                cv2.drawMarker(vis, center, color, cv2.MARKER_CROSS, 12, 2)
                cv2.circle(vis, center, 8, color, 2)
                cv2.putText(
                    vis,
                    f"BEST ID:{tid} ({display_best_score:.2f})",
                    (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )
            else:
                color = (0, 0, 255)
                cv2.circle(vis, center, 8, color, -1)
                cv2.putText(
                    vis,
                    f"ID:{tid}",
                    (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

        cv2.putText(
            vis,
            f"FPS: {int(fps)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        if is_recording:
            # Flashing red RECORDING indicator
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(
                    vis, "[REC]", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3
                )

        target_color_display = (255, 0, 0) if mode == "blue" else (0, 0, 255)
        cv2.putText(
            vis,
            f"TARGET: {mode.upper()}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            target_color_display,
            2,
        )

        if not is_recording or frame_count % 5 == 0:
            cv2.imshow(win_main, vis)
            cv2.imshow(win_bin, binary)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if not is_recording:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(
                    output_dir, f"basler_raw_{mode}_{timestamp}.mp4"
                )
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = ThreadedVideoWriter(filename, fourcc, 80.0, (w, h))
                is_recording = True
                log_str = (
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RECORDING: {filename}\n"
                    f"Target: {mode.upper()}, Exposure: {exp}, Threshold: {cth}, Green Penalty: {penalty_val}\n"
                    f"WB Red: {wb_r}, WB Green: {wb_g}, WB Blue: {wb_b}\n"
                    f"------------------------------------------\n"
                )
                with open("tuning_log.txt", "a") as f:
                    f.write(log_str)
                print(f"[INFO] Started recording: {filename}")
                print(log_str)
            else:
                is_recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("[INFO] Stopped recording.")
        elif key == ord("r"):
            if mode != "red":
                mode = "red"
                cv2.setTrackbarPos("Exposure", win_main, params[mode]["exp"])
                cv2.setTrackbarPos("ColorThresh", win_main, params[mode]["color_th"])
                cv2.setTrackbarPos("WB Red", win_main, int(params[mode]["wb_r"] * 10))
                cv2.setTrackbarPos("WB Green", win_main, int(params[mode]["wb_g"] * 10))
                cv2.setTrackbarPos("WB Blue", win_main, int(params[mode]["wb_b"] * 10))
                cv2.setTrackbarPos(
                    "GrnPenalty", win_main, int(params[mode]["penalty"] * 100)
                )
        elif key == ord("b"):
            if mode != "blue":
                mode = "blue"
                cv2.setTrackbarPos("Exposure", win_main, params[mode]["exp"])
                cv2.setTrackbarPos("ColorThresh", win_main, params[mode]["color_th"])
                cv2.setTrackbarPos("WB Red", win_main, int(params[mode]["wb_r"] * 10))
                cv2.setTrackbarPos("WB Green", win_main, int(params[mode]["wb_g"] * 10))
                cv2.setTrackbarPos("WB Blue", win_main, int(params[mode]["wb_b"] * 10))
                cv2.setTrackbarPos(
                    "GrnPenalty", win_main, int(params[mode]["penalty"] * 100)
                )

    if video_writer is not None:
        video_writer.release()

    cam.camera.Close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
