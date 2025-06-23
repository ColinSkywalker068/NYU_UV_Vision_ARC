import pyrealsense2 as rs
import cv2
import numpy as np
import os
import random
import time
import argparse
from ultralytics import YOLO, RTDETR, NAS

os.environ["GDK_SYNCHRONIZE"] = "1" # If using GTK-based cv2.imshow concurrently

# Team and Class Definitions
RED_TEAM = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
BLUE_TEAM = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111)

RED_CLASS = 1
BLUE_CLASS = 0

# Color Ranges for HSV
BLUE_RANGES = [
    [np.array((100, 150, 150), np.uint8), np.array((128, 255, 255), np.uint8)]
]
RED_RANGES = [
    [np.array((0, 150, 150), np.uint8), np.array((10, 255, 255), np.uint8)],
    [np.array((170, 150, 150), np.uint8), np.array((180, 255, 255), np.uint8)],
]

# Bounding Box Drawing Parameters
BBOX_RAW_COLOR = (0, 255, 0)  # Green for raw YOLO detection
BBOX_KF_COLOR = (255, 0, 0)   # Blue for Kalman Filter prediction
BBOX_LINE_SIZE = 2

# ──────────────────────────────────────────────────────────────
# Kalman Filter Tracker Classes
# ──────────────────────────────────────────────────────────────
class ArmorPlateTracker:
    def __init__(self, initial_x, initial_y, width, height):
        self.kf = cv2.KalmanFilter(6, 2) # 6 states (x,y,vx,vy,ax,ay), 2 measurements (x,y)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)

        # --- TUNABLE KALMAN PARAMETERS ---
        proc_noise_pos = 0.01
        proc_noise_vel = 0.05
        proc_noise_acc = 0.1
        
        self.kf.processNoiseCov = np.diag([
            proc_noise_pos, proc_noise_pos, 
            proc_noise_vel, proc_noise_vel,
            proc_noise_acc, proc_noise_acc
        ]).astype(np.float32) * 1.0

        meas_noise = 0.15
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * meas_noise
        # --- END TUNABLE KALMAN PARAMETERS ---

        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1

        # Initial state: [x, y, vx, vy, ax, ay]'
        self.kf.statePost = np.array([
            np.float32(initial_x), np.float32(initial_y), 0, 0, 0, 0
        ], np.float32).reshape(6, 1)

        self.color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.width = float(width)
        self.height = float(height)
        self._last_dt_used = 0.033

    def _update_transition_matrix(self, dt):
        self._last_dt_used = dt
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)

    def predict(self, dt):
        self._update_transition_matrix(dt)
        prediction_state = self.kf.predict()
        return int(prediction_state[0, 0]), int(prediction_state[1, 0])

    def update(self, x, y, width, height):
        measurement = np.array([[np.float32(x)], [np.float32(y)]], np.float32)
        self.kf.correct(measurement)
        self.width = float(width)
        self.height = float(height)

class TrackerWrapper:
    def __init__(self, tracker: ArmorPlateTracker):
        self.tracker = tracker
        self.lost_frames = 0

# ──────────────────────────────────────────────────────────────
# Main Detection Class
# ──────────────────────────────────────────────────────────────
class YOLOv8Node:
    def __init__(self, model_path="/home/ultraviolet/CV_Detection/yolov8_ros/yolov8_ros/best.engine", 
                 task="detect", model_type="yolo", half=True, imgsz=640, show=True, conf=0.05,
                 tracker_timeout=15, kalman_initial_dt=1.0/60.0, prediction_lookahead_steps=0.5,
                 robot_id=1, use_webcam=False, camera_width=640, camera_height=480):
        
        print("Initializing YOLOv8Node with Kalman Filter & Lag Compensation...")
        
        # Parameters
        self.task = task
        self.half = half
        self.imgsz = imgsz
        self.show = show
        self.conf = conf
        self.tracker_timeout_val = tracker_timeout
        self.kalman_initial_dt_val = kalman_initial_dt
        self.prediction_lookahead_steps_val = prediction_lookahead_steps
        
        print(f"Model Path: {model_path}")
        print(f"Model Type: {model_type}, Task: {task}, ImgSz: {imgsz}")
        print(f"Confidence: {conf}, Show Viz: {show}, Half Prec: {half}")
        print(f"Tracker Timeout: {tracker_timeout} frames")
        print(f"Kalman Initial dt: {kalman_initial_dt:.4f}s")
        print(f"Prediction Lookahead Steps: {prediction_lookahead_steps}")
        
        # Set target class based on robot_id
        if robot_id in RED_TEAM:
            self.target_class = BLUE_CLASS
            print(f"Robot ID {robot_id} is RED_TEAM. Targeting BLUE_CLASS.")
        elif robot_id in BLUE_TEAM:
            self.target_class = RED_CLASS
            print(f"Robot ID {robot_id} is BLUE_TEAM. Targeting RED_CLASS.")
        else:
            self.target_class = BLUE_CLASS
            print(f"Robot ID {robot_id} not in known teams. Defaulting to BLUE_CLASS.")

        # Model Initialization
        if model_type == "yolo": 
            self.model = YOLO(model_path, task=task)
        elif model_type == "rtdetr": 
            self.model = RTDETR(model_path)
        elif model_type == "nas": 
            self.model = NAS(model_path)
        else:
            raise NotImplementedError("Unsupported model type specified.")
        print(f"YOLO Model '{model_type}' initialized successfully.")

        # Camera Initialization
        if use_webcam:
            print("Initializing webcam...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.pipeline = None
        else:
            print("Initializing RealSense camera...")
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, camera_width, camera_height, rs.format.bgr8, 30)
            self.pipeline.start(config)
            self.cap = None
        
        # State Variables
        self.active_trackers = {}
        self.last_frame_time = None
        self.bbox_data = [imgsz//2, imgsz//2, imgsz//2, imgsz//2]  # Default bbox
        self.override_active = False

    def get_frame(self):
        """Get frame from either RealSense or webcam"""
        if self.pipeline:  # RealSense
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            return np.asanyarray(color_frame.get_data())
        else:  # Webcam
            ret, frame = self.cap.read()
            return frame if ret else None

    def process_frame(self, frame):
        """Main frame processing function - replaces the old listener_callback"""
        # Calculate dynamic dt
        current_time = time.time()
        dt = self.kalman_initial_dt_val 
        if self.last_frame_time is not None:
            time_diff_s = current_time - self.last_frame_time
            if 0.001 < time_diff_s < 1.0:
                dt = time_diff_s
            else:
                print(f"Unusual dt calculated ({time_diff_s:.4f}s), using fallback dt: {dt:.4f}s")
        self.last_frame_time = current_time

        original_height, original_width = frame.shape[:2]

        # Robust Padding/Resizing (Letterboxing)
        target_h, target_w = self.imgsz, self.imgsz
        scale = 0.0
        if original_width > 0 and original_height > 0:
            scale = min(target_w / original_width, target_h / original_height)
        new_w, new_h = int(original_width * scale), int(original_height * scale)
        
        padded_image = np.full((self.imgsz, self.imgsz, 3), (114, 114, 114), dtype=np.uint8)
        if new_w > 0 and new_h > 0:
            resized_image = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
            padded_image[dh:dh+new_h, dw:dw+new_w] = resized_image
        else:
            print(f"Invalid resize dimensions ({new_w}x{new_h}). Resizing directly (may distort).")
            padded_image = cv2.resize(frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        # YOLO Model Tracking
        results = self.model.track(
            padded_image,
            half=self.half,
            imgsz=self.imgsz,
            show=False,
            conf=self.conf,
            persist=True,
            tracker='bytetrack.yaml'
        )
        
        vis_frame = padded_image.copy() if self.show else None
        current_frame_processed_track_ids = set()
        published_bbox_this_frame = False

        if results and len(results) > 0:
            res_obj = results[0]
            
            boxes_data = None
            if hasattr(res_obj, "boxes") and res_obj.boxes is not None: 
                boxes_data = res_obj.boxes
            elif hasattr(res_obj, "obb") and res_obj.obb is not None and self.task == "obb": 
                boxes_data = res_obj.obb
            
            if boxes_data is not None and hasattr(boxes_data, 'xyxy') and boxes_data.xyxy is not None:
                bbox_coords_list = boxes_data.xyxy.cpu().int().tolist()

                track_ids_list = []
                if hasattr(boxes_data, 'id') and boxes_data.id is not None:
                    track_ids_list = boxes_data.id.cpu().int().tolist()
                else:
                    track_ids_list = [None] * len(bbox_coords_list)
                
                for i, bbox_xyxy in enumerate(bbox_coords_list):
                    track_id = track_ids_list[i]
                    x1, y1, x2, y2 = bbox_xyxy

                    if self.show and vis_frame is not None:
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), BBOX_RAW_COLOR, BBOX_LINE_SIZE)
                        if track_id is not None: 
                            cv2.putText(vis_frame, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_RAW_COLOR, 1)

                    roi_img = padded_image[max(0,y1):min(y2, self.imgsz), max(0,x1):min(x2, self.imgsz)]
                    if roi_img.shape[0] == 0 or roi_img.shape[1] == 0: 
                        continue
                    
                    is_enemy, _ = self.check_enemy(roi_img)

                    if is_enemy and track_id is not None:
                        current_frame_processed_track_ids.add(track_id)
                        center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        width_det, height_det = float(abs(x2 - x1)), float(abs(y2 - y1))

                        if track_id not in self.active_trackers:
                            print(f"Creating new Kalman tracker for enemy ID: {track_id}")
                            self.active_trackers[track_id] = TrackerWrapper(ArmorPlateTracker(center_x, center_y, width_det, height_det))
                        else:
                            self.active_trackers[track_id].tracker.update(center_x, center_y, width_det, height_det)
                        self.active_trackers[track_id].lost_frames = 0

        # Manage and predict from active Kalman trackers
        active_ids_to_remove = []
        for track_id, tw_wrapper in self.active_trackers.items():
            if track_id not in current_frame_processed_track_ids:
                tw_wrapper.lost_frames += 1
            
            if tw_wrapper.lost_frames >= self.tracker_timeout_val:
                active_ids_to_remove.append(track_id)
                print(f"Removing tracker ID {track_id} (lost {tw_wrapper.lost_frames} frames).")
                continue

            if not published_bbox_this_frame and tw_wrapper.lost_frames == 0:
                pred_x_curr, pred_y_curr = tw_wrapper.tracker.predict(dt)

                # Get latest corrected velocity and acceleration for lookahead
                last_corrected_state = tw_wrapper.tracker.kf.statePost.flatten()
                vx, vy = last_corrected_state[2], last_corrected_state[3]
                ax, ay = last_corrected_state[4], last_corrected_state[5]

                additional_lookahead_time = self.prediction_lookahead_steps_val * dt
                
                pred_x_final = pred_x_curr + vx * additional_lookahead_time + 0.5 * ax * (additional_lookahead_time**2)
                pred_y_final = pred_y_curr + vy * additional_lookahead_time + 0.5 * ay * (additional_lookahead_time**2)

                kf_width, kf_height = tw_wrapper.tracker.width, tw_wrapper.tracker.height
                pub_x1 = int(pred_x_final - kf_width / 2)
                pub_y1 = int(pred_y_final - kf_height / 2)
                pub_x2 = int(pred_x_final + kf_width / 2)
                pub_y2 = int(pred_y_final + kf_height / 2)
                
                # Clamp coordinates to image boundaries
                pub_x1 = max(0, min(pub_x1, self.imgsz -1))
                pub_y1 = max(0, min(pub_y1, self.imgsz -1))
                pub_x2 = max(0, min(pub_x2, self.imgsz -1))
                pub_y2 = max(0, min(pub_y2, self.imgsz -1))

                if pub_x1 < pub_x2 and pub_y1 < pub_y2:
                    self.bbox_data = [pub_x1, pub_y1, pub_x2, pub_y2]
                    self.override_active = True
                    published_bbox_this_frame = True
                    print(f"Updated KF BBox ID {track_id} (Lookahead): {self.bbox_data}")

                    if self.show and vis_frame is not None:
                        cv2.rectangle(vis_frame, (pub_x1, pub_y1), (pub_x2, pub_y2), BBOX_KF_COLOR, BBOX_LINE_SIZE)
                        text_y_pos = pub_y1 - 25 if pub_y1 > 25 else pub_y1 + int(kf_height) + 15
                        cv2.putText(vis_frame, f"KF_ID:{track_id}", (pub_x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_KF_COLOR, 1)
                else:
                    print(f"KF ID {track_id} produced invalid lookahead box ({[pub_x1, pub_y1, pub_x2, pub_y2]}). Not updating.")
        
        for track_id_to_remove in active_ids_to_remove:
            if track_id_to_remove in self.active_trackers: 
                del self.active_trackers[track_id_to_remove]

        if not published_bbox_this_frame:
            default_center = self.imgsz // 2
            self.bbox_data = [default_center, default_center, default_center, default_center]
            self.override_active = False

        if self.show and vis_frame is not None:
            cv2.imshow("YOLOv8 Kalman Tracking", vis_frame)
            cv2.waitKey(1)

        return vis_frame

    def check_enemy(self, roi_image):
        """Enemy validation method"""
        if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
            print("check_enemy: Invalid ROI shape (0 dimension); returning False.")
            return False, None

        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        target_ranges = BLUE_RANGES if self.target_class == BLUE_CLASS else RED_RANGES
        
        final_mask = np.zeros_like(hsv_roi[:,:,0])
        for hsv_range in target_ranges:
            current_mask = cv2.inRange(hsv_roi, hsv_range[0], hsv_range[1])
            final_mask = cv2.bitwise_or(final_mask, current_mask)
            
        color_area = np.count_nonzero(final_mask)
        min_color_area_threshold = roi_image.shape[0] * roi_image.shape[1] * 0.01 
        is_it_enemy = color_area > min_color_area_threshold
        
        return is_it_enemy, color_area

    def get_bbox_data(self):
        """Get current bounding box data"""
        return self.bbox_data, self.override_active

    def run(self):
        """Main execution loop"""
        print("Starting detection loop. Press 'q' to quit, 't' to toggle target class...")
        
        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    print("Failed to get frame")
                    continue

                # Process frame
                processed_frame = self.process_frame(frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    self.target_class = BLUE_CLASS if self.target_class == RED_CLASS else RED_CLASS
                    target_str = 'BLUE' if self.target_class == BLUE_CLASS else 'RED'
                    print(f"Switched target class to: {target_str}")
                elif key == ord('s'):
                    # Save current bbox data to file for external access
                    bbox_data, override = self.get_bbox_data()
                    with open('/tmp/bbox_data.txt', 'w') as f:
                        f.write(f"{bbox_data[0]},{bbox_data[1]},{bbox_data[2]},{bbox_data[3]},{override}\n")
                    print(f"Saved bbox data: {bbox_data}, override: {override}")

        except KeyboardInterrupt:
            print("Keyboard interrupt received, shutting down.")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.pipeline:
            self.pipeline.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='YOLOv8 Detection with Kalman Filter using RealSense/Webcam')
    parser.add_argument('--model_path', default='/home/ultraviolet/CV_Detection/yolov8_ros/yolov8_ros/best.engine', help='Path to YOLO model')
    parser.add_argument('--task', default='detect', choices=['detect', 'obb'], help='Detection task')
    parser.add_argument('--type', default='yolo', choices=['yolo', 'rtdetr', 'nas'], help='Model type')
    parser.add_argument('--half', action='store_true', default=True, help='Use FP16 precision')
    parser.add_argument('--imgsz', type=int, default=640, help='Model input image size')
    parser.add_argument('--show', action='store_true', default=True, help='Show visualization')
    parser.add_argument('--conf', type=float, default=0.05, help='Confidence threshold')
    parser.add_argument('--tracker_timeout', type=int, default=15, help='Frames to keep lost track')
    parser.add_argument('--kalman_dt', type=float, default=1.0/60.0, help='Initial Kalman dt')
    parser.add_argument('--lookahead', type=float, default=0.5, help='Prediction lookahead steps')
    parser.add_argument('--robot_id', type=int, default=1, help='Robot ID for team assignment')
    parser.add_argument('--webcam', action='store_true', help='Use webcam instead of RealSense')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    
    args = parser.parse_args()
    
    detector = YOLOv8Node(
        model_path=args.model_path,
        task=args.task,
        model_type=args.type,
        half=args.half,
        imgsz=args.imgsz,
        show=args.show,
        conf=args.conf,
        tracker_timeout=args.tracker_timeout,
        kalman_initial_dt=args.kalman_dt,
        prediction_lookahead_steps=args.lookahead,
        robot_id=args.robot_id,
        use_webcam=args.webcam,
        camera_width=args.width,
        camera_height=args.height
    )
    
    detector.run()

if __name__ == "__main__":
    main()