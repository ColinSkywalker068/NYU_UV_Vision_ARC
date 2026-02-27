import cv2
import numpy as np
from pypylon import pylon
import time
import sys

class ArmorDetector:
    def __init__(self):
        # Constants from your golden parameters
        self.light_min_ratio = 3.0
        self.match_max_angle_diff = 10.0
        self.armor_min_ratio = 0.6
        self.armor_max_ratio = 5.0

    def fast_process(self, img, enemy_color, color_th=55):
        # Efficient channel splitting
        b, g, r = cv2.split(img)
        
        # Color subtraction based on target
        if enemy_color == 'red':
            diff = cv2.subtract(r, b)
        else:
            diff = cv2.subtract(b, r)
        
        _, binary = cv2.threshold(diff, color_th, 255, cv2.THRESH_BINARY)
        
        # Minimal dilation for performance
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
        return cv2.dilate(binary, kernel, iterations=1)

    def get_armor_centers(self, binary):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lights = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 15: continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            length, width = (w, h) if w > h else (h, w)
            if width > 0 and (length / width) >= self.light_min_ratio:
                # Normalize angle
                draw_angle = angle + 90 if w > h else angle
                if w <= h: draw_angle = angle if angle < -45 else angle + 90
                lights.append({'center': (cx, cy), 'length': length, 'angle': draw_angle})
        
        armors = []
        for i in range(len(lights)):
            for j in range(i + 1, len(lights)):
                l1, l2 = lights[i], lights[j]
                if abs(l1['angle'] - l2['angle']) <= self.match_max_angle_diff:
                    dist_x = abs(l1['center'][0] - l2['center'][0])
                    avg_len = (l1['length'] + l2['length']) / 2.0
                    ratio = dist_x / avg_len
                    if self.armor_min_ratio < ratio < self.armor_max_ratio:
                        armors.append((int((l1['center'][0]+l2['center'][0])/2), 
                                       int((l1['center'][1]+l2['center'][1])/2)))
        return armors

def main():
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
        
        # Set Inverse WB: Targeting BLUE as default
        node.GetNode("BalanceRatioSelector").SetValue("Red")
        node.GetNode("BalanceRatio").SetValue(3.5)
        node.GetNode("BalanceRatioSelector").SetValue("Blue")
        node.GetNode("BalanceRatio").SetValue(0.9)
        
        # Optimize Throughput for Orin 2
        # node.GetNode("DeviceLinkThroughputLimit").SetValue(250000000) 
        
        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        
        print(f"[INIT] Camera {camera.GetDeviceInfo().GetModelName()} ready.")
    except Exception as e:
        print(f"[FATAL] Camera init error: {e}")
        return

    detector = ArmorDetector()
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
                binary = detector.fast_process(img, 'blue')
                armor_coords = detector.get_armor_centers(binary)

                # 4. Performance Benchmarking
                frame_count += 1
                if frame_count % 30 == 0:
                    now = time.time()
                    fps = 30 / (now - last_time)
                    last_time = now
                
                # 5. Data Output (Simulating competition data stream)
                if armor_coords:
                    print(f"FPS: {fps:.1f} | Targets: {armor_coords}")
                else:
                    print(f"FPS: {fps:.1f} | Searching...")

                # Optional Visualization (Disable for absolute max performance)
                cv2.imshow("Bench", img)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            else:
                res.Release()

    except KeyboardInterrupt:
        print("[INFO] Terminated by user.")
    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()