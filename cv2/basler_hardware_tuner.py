import cv2
import numpy as np
from pypylon import pylon
from pypylon import genicam
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cv_trackers import EuclideanTracker, KalmanMultiTracker

# --- CONFIGURATION TOGGLE ---
# Change to 'kalman' to test Kalman Filter tracking on Orin
# Change to 'euclidean' for maximum speed
# Change to 'none' to disable tracking and test raw detection FPS
TRACKER_TYPE = "euclidean"


class ArmorDetector:
    def __init__(self):
        self.light_min_ratio = 3.0
        self.match_max_angle_diff = 10.0
        self.armor_min_ratio = 0.6
        self.armor_max_ratio = 5.0

    def preprocess(self, img, enemy_color, color_th, gamma_val):
        # Apply SW Gamma if hardware gamma is not sufficient
        if gamma_val != 1.0:
            invGamma = 1.0 / gamma_val
            table = np.array(
                [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
            ).astype("uint8")
            img = cv2.LUT(img, table)

        # Core inverse-WB logic: (B-R) for blue, (R-B) for red
        # Using NumPy slicing for zero-copy memory access
        if enemy_color == "red":
            diff = cv2.subtract(img[:, :, 2], img[:, :, 0])
        else:
            diff = cv2.subtract(img[:, :, 0], img[:, :, 2])

        _, binary = cv2.threshold(diff, color_th, 255, cv2.THRESH_BINARY)

        # Quick dilation to merge fragmented light bars
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        binary = cv2.dilate(binary, kernel, iterations=1)
        return binary

    def find_targets(self, binary):
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        lights = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 15:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            length, width = (w, h) if w > h else (h, w)
            if width == 0 or (length / width) < self.light_min_ratio:
                continue

            # Angle normalization
            draw_angle = angle + 90 if w > h else angle
            if w <= h:
                draw_angle = angle if angle < -45 else angle + 90
            lights.append(
                {
                    "center": (cx, cy),
                    "length": length,
                    "angle": draw_angle,
                    "box": np.int0(cv2.boxPoints(rect)),
                }
            )

        armors = []
        for i in range(len(lights)):
            for j in range(i + 1, len(lights)):
                l1, l2 = lights[i], lights[j]
                if abs(l1["angle"] - l2["angle"]) > self.match_max_angle_diff:
                    continue
                dist_x = abs(l1["center"][0] - l2["center"][0])
                avg_len = (l1["length"] + l2["length"]) / 2.0
                ratio = dist_x / avg_len
                if self.armor_min_ratio < ratio < self.armor_max_ratio:
                    armors.append(
                        {
                            "center": (
                                int((l1["center"][0] + l2["center"][0]) / 2),
                                int((l1["center"][1] + l2["center"][1]) / 2),
                            )
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
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            print(f"[CAM] Connected: {self.camera.GetDeviceInfo().GetModelName()}")
        except Exception as e:
            print(f"[ERR] Camera failed: {e}")
            sys.exit()

    def set_config(self, exp, gain, wb_r, wb_b, gamma):
        try:
            self.nodemap.GetNode("ExposureAuto").FromString("Off")
            self.nodemap.GetNode("GainAuto").FromString("Off")
            self.nodemap.GetNode("BalanceWhiteAuto").FromString("Off")
            self.nodemap.GetNode("ExposureTime").SetValue(float(exp))
            self.nodemap.GetNode("Gain").SetValue(float(gain))
            self.nodemap.GetNode("Gamma").SetValue(float(gamma))

            # Apply your "Inverse WB" strategy
            self.nodemap.GetNode("BalanceRatioSelector").SetValue("Red")
            self.nodemap.GetNode("BalanceRatio").SetValue(float(wb_r))
            self.nodemap.GetNode("BalanceRatioSelector").SetValue("Blue")
            self.nodemap.GetNode("BalanceRatio").SetValue(float(wb_b))
        except:
            pass

    def get_frame(self):
        if not self.camera.IsGrabbing():
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if res.GrabSucceeded():
            img = self.converter.Convert(res).GetArray()
            res.Release()
            return img
        res.Release()
        return None


def main():
    cam = BaslerDriver()
    detector = ArmorDetector()
    tracker = (
        EuclideanTracker()
        if TRACKER_TYPE == "euclidean"
        else (KalmanMultiTracker() if TRACKER_TYPE == "kalman" else None)
    )
    # Target configurations (Golden Params from testing)
    params = {
        "blue": {"exp": 723, "color_th": 55, "wb_b": 0.9, "wb_r": 3.5, "gamma": 0.9},
        "red": {"exp": 723, "color_th": 55, "wb_b": 3.5, "wb_r": 0.9, "gamma": 0.9},
    }
    mode = "blue"

    win_main = "Orin Benchmark - Main"
    win_bin = "Orin Benchmark - Binary"

    # Initialize both windows as NORMAL for resizing
    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_bin, cv2.WINDOW_NORMAL)

    # Set default sizes for initial launch
    cv2.resizeWindow(win_main, 960, 540)
    cv2.resizeWindow(win_bin, 960, 540)

    # Trackbars on main window
    cv2.createTrackbar("Exposure", win_main, params[mode]["exp"], 2000, lambda x: None)
    cv2.createTrackbar(
        "ColorThresh", win_main, params[mode]["color_th"], 100, lambda x: None
    )

    prev_time = time.time()
    print("\n[INFO] R: RED Mode | B: BLUE Mode | Q: Quit | S: Save Params\n")

    while True:
        # Real-time Hardware Sync
        exp = cv2.getTrackbarPos("Exposure", win_main)
        cth = cv2.getTrackbarPos("ColorThresh", win_main)
        cam.set_config(
            exp, 0, params[mode]["wb_r"], params[mode]["wb_b"], params[mode]["gamma"]
        )

        frame = cam.get_frame()
        if frame is None:
            continue

        # CV Pipeline Execution
        binary = detector.preprocess(frame, mode, cth, params[mode]["gamma"])
        lights, armors = detector.find_targets(binary)
        armor_centers = [a["center"] for a in armors]

        if tracker:
            tracked_targets = tracker.update(armor_centers)
        else:
            tracked_targets = {f"Raw-{i}": c for i, c in enumerate(armor_centers)}

        # FPS Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # Drawing Results
        vis = frame.copy()
        for l in lights:
            cv2.drawContours(vis, [l["box"]], -1, (0, 255, 0), 2)

        # Draw tracked targets with IDs
        for tid, center in tracked_targets.items():
            cv2.circle(vis, center, 8, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"ID:{tid}",
                (center[0] + 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        # Dashboard Information
        cv2.putText(
            vis,
            f"FPS: {int(fps)}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        target_color_display = (255, 0, 0) if mode == "blue" else (0, 0, 255)
        cv2.putText(
            vis,
            f"TARGET: {mode.upper()}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            target_color_display,
            2,
        )

        # Displaying both windows
        cv2.imshow(win_main, vis)
        cv2.imshow(win_bin, binary)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            mode = "red"
        elif key == ord("b"):
            mode = "blue"
        elif key == ord("s"):
            print(f"\n--- Current Configuration [{mode.upper()}] ---")
            print(f"Exposure: {exp}, Threshold: {cth}")
            print(f"WB Red: {params[mode]['wb_r']}, WB Blue: {params[mode]['wb_b']}")
            print("------------------------------------------\n")

    cam.camera.Close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
