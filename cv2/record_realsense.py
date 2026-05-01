import cv2
import numpy as np
import time
import sys
import os

try:
    import pyrealsense2 as rs
except ImportError:
    print("[WARN] pyrealsense2 not installed. Please run `pip install pyrealsense2`")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from cv_trackers import EuclideanTracker, KalmanMultiTracker
except ImportError:
    pass

# --- CONFIGURATION TOGGLE ---
TRACKER_TYPE = "kalman"

class ArmorDetector:
    def __init__(self):
        self.light_min_ratio = 3.0
        self.match_max_angle_diff = 10.0
        self.armor_min_ratio = 0.6
        self.armor_max_ratio = 5.0
        self.light_max_tilt = 40.0

    def preprocess(self, img, enemy_color, color_th, penalty_factor=0.0):
        # RealSense has decent hardware color, we won't manually do gamma mapping here 
        # unless necessary. For max speed we stick to subtraction.
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
                avg_len = (l1["length"] + l2["length"]) / 2.0
                ratio = dist_x / avg_len
                
                if self.armor_min_ratio < ratio < self.armor_max_ratio:
                    len_sim = 1.0 - (abs(l1["length"] - l2["length"]) / max(l1["length"], l2["length"]))
                    y_sim = 1.0 - min(abs(l1["center"][1] - l2["center"][1]) / avg_len, 1.0)
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
                            "lights": (l1, l2)
                        }
                    )
        return lights, armors


class RealSenseDriver:
    def __init__(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            # D455 optimally streams at 848x480 for 90 FPS, or 1280x720 at 30 FPS.
            # Using 848x480 at 90FPS to match the high speed nature of the Basler setup.
            self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 90)
            
            self.profile = self.pipeline.start(self.config)
            
            # Access the color sensor to tune hardware params
            device = self.profile.get_device()
            self.color_sensor = None
            for s in device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    self.color_sensor = s
                    break

            if self.color_sensor:
                # Disable auto features to allow manual control
                self.color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                self.color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

            print(f"[CAM] RealSense connected: {device.get_info(rs.camera_info.name)}")
        except Exception as e:
            print(f"[ERR] RealSense Camera failed: {e}")
            sys.exit()

    def set_config(self, exp, wb, saturation, contrast):
        if not self.color_sensor:
            return
        try:
            # Exposure for RealSense is typically in microseconds (e.g. 50-10000)
            self.color_sensor.set_option(rs.option.exposure, exp)
            # White Balance is a temperature value in Kelvin (2800 - 6500)
            self.color_sensor.set_option(rs.option.white_balance, wb)
            self.color_sensor.set_option(rs.option.saturation, saturation)
            self.color_sensor.set_option(rs.option.contrast, contrast)
        except:
            pass

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def get_temperature(self):
        # D455 does have an ASIC temperature sensor but it requires querying the proper sensor.
        return 0.0, "Unknown"

    def close(self):
        self.pipeline.stop()


def main():
    cam = RealSenseDriver()
    detector = ArmorDetector()
    tracker = (
        EuclideanTracker()
        if TRACKER_TYPE == "euclidean"
        else (KalmanMultiTracker() if TRACKER_TYPE == "kalman" else None)
    )

    # Note: RealSense parameters are completely different from Basler
    # WB is a single color temperature. 
    params = {
        "blue": {
            "exp": 150,
            "color_th": 55,
            "wb": 5500,        # Cooler temperature brings out blue
            "saturation": 60,
            "contrast": 50,
            "penalty": 0.0,
        },
        "red": {
            "exp": 150,
            "color_th": 55,
            "wb": 3200,        # Warmer temperature brings out red
            "saturation": 60,
            "contrast": 50,
            "penalty": 0.0,
        },
    }
    mode = "blue"

    win_main = "RealSense Recorder - Main"
    win_bin = "RealSense Recorder - Binary"

    cv2.namedWindow(win_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_bin, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_main, 848, 480)
    cv2.resizeWindow(win_bin, 848, 480)

    cv2.createTrackbar("Exposure", win_main, params[mode]["exp"], 5000, lambda x: None)
    cv2.createTrackbar("ColorThresh", win_main, params[mode]["color_th"], 100, lambda x: None)
    cv2.createTrackbar("WhiteBalance", win_main, params[mode]["wb"], 6500, lambda x: None)
    cv2.setTrackbarMin("WhiteBalance", win_main, 2800)
    cv2.createTrackbar("Saturation", win_main, params[mode]["saturation"], 100, lambda x: None)
    cv2.createTrackbar("Contrast", win_main, params[mode]["contrast"], 100, lambda x: None)
    cv2.createTrackbar("GrnPenalty", win_main, int(params[mode]["penalty"] * 100), 100, lambda x: None)

    prev_time = time.time()
    print("\n[INFO] R: RED Mode | B: BLUE Mode | SPACE: Toggle Record | Q: Quit\n")

    is_recording = False
    video_writer = None
    output_dir = os.path.join(os.path.dirname(__file__), "videos")
    os.makedirs(output_dir, exist_ok=True)
    
    dynamic_min_area = 10
    current_best_tid = None

    while True:
        exp = max(1, cv2.getTrackbarPos("Exposure", win_main))
        cth = cv2.getTrackbarPos("ColorThresh", win_main)
        wb = max(2800, cv2.getTrackbarPos("WhiteBalance", win_main))
        sat = cv2.getTrackbarPos("Saturation", win_main)
        contrast = cv2.getTrackbarPos("Contrast", win_main)
        penalty_val = cv2.getTrackbarPos("GrnPenalty", win_main) / 100.0

        cam.set_config(exp, wb, sat, contrast)

        frame = cam.get_frame()
        if frame is None:
            continue

        # Write RAW frame if recording
        if is_recording and video_writer is not None:
            video_writer.write(frame)

        binary = detector.preprocess(frame, mode, cth, penalty_factor=penalty_val)
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
                best_dist = float('inf')
                raw_score = 0.0
                mapped_armor = None
                for a in armors:
                    dist = (t_center[0] - a["center"][0])**2 + (t_center[1] - a["center"][1])**2
                    if dist < best_dist:
                        best_dist = dist
                        raw_score = a["score"]
                        mapped_armor = a
                
                assigned_score = raw_score
                if tid == current_best_tid:
                    assigned_score += 5.0

                tid_scores[tid] = {"score": assigned_score, "raw": raw_score, "armor": mapped_armor}
            
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
            is_best = (tid == current_best_tid)
            if is_best:
                color = (255, 0, 255)
                cv2.drawMarker(vis, center, color, cv2.MARKER_CROSS, 12, 2)
                cv2.circle(vis, center, 8, color, 2)
                cv2.putText(vis, f"BEST ID:{tid} ({display_best_score:.2f})", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                color = (0, 0, 255)
                cv2.circle(vis, center, 8, color, -1)
                cv2.putText(vis, f"ID:{tid}", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(vis, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if is_recording:
            # Flashing red RECORDING indicator
            if int(time.time() * 2) % 2 == 0:
                cv2.putText(vis, "[REC]", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        target_color_display = (255, 0, 0) if mode == "blue" else (0, 0, 255)
        cv2.putText(vis, f"TARGET: {mode.upper()}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, target_color_display, 2)

        cv2.imshow(win_main, vis)
        cv2.imshow(win_bin, binary)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if not is_recording:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"realsense_raw_{mode}_{timestamp}.mp4")
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(filename, fourcc, 60.0, (w, h))
                is_recording = True
                log_str = (
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] RECORDING: {filename}\n"
                    f"Target: {mode.upper()}, Exposure: {exp}, Threshold: {cth}, Green Penalty: {penalty_val}\n"
                    f"RealSense Temp: {wb}K, Saturation: {sat}, Contrast: {contrast}\n"
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
                cv2.setTrackbarPos("WhiteBalance", win_main, params[mode]["wb"])
                cv2.setTrackbarPos("Saturation", win_main, params[mode]["saturation"])
                cv2.setTrackbarPos("Contrast", win_main, params[mode]["contrast"])
                cv2.setTrackbarPos("GrnPenalty", win_main, int(params[mode]["penalty"] * 100))
        elif key == ord("b"):
            if mode != "blue":
                mode = "blue"
                cv2.setTrackbarPos("Exposure", win_main, params[mode]["exp"])
                cv2.setTrackbarPos("ColorThresh", win_main, params[mode]["color_th"])
                cv2.setTrackbarPos("WhiteBalance", win_main, params[mode]["wb"])
                cv2.setTrackbarPos("Saturation", win_main, params[mode]["saturation"])
                cv2.setTrackbarPos("Contrast", win_main, params[mode]["contrast"])
                cv2.setTrackbarPos("GrnPenalty", win_main, int(params[mode]["penalty"] * 100))

    if video_writer is not None:
        video_writer.release()

    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
