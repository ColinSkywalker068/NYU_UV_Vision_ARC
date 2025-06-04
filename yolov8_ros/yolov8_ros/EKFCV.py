import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, Int32MultiArray, Bool
from cv_bridge import CvBridge
from ultralytics import YOLO, RTDETR, NAS # Ensure ultralytics is installed
import cv2
import numpy as np
import os
import random # For Kalman tracker color

# SAHI Imports for Slicing Aided Hyper Inference
from sahi.slicing import slice_image
from sahi.prediction import ObjectPrediction, PredictionResult

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
BBOX_KF_COLOR = (255, 0, 0)   # Blue for Kalman Filter prediction (active, not selected)
BBOX_SELECTED_KF_COLOR = (0, 255, 255) # Yellow for the selected KF target
BBOX_SAHI_COLOR = (255, 0, 255) # Magenta for SAHI NMS detections
BBOX_LINE_SIZE = 2

# ──────────────────────────────────────────────────────────────
# Kalman Filter Tracker Classes
# ──────────────────────────────────────────────────────────────
class ArmorPlateTracker:
    def __init__(self, initial_x, initial_y, width, height):
        self.kf = cv2.KalmanFilter(6, 2) 
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)

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
        
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1 

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
            [1,0,dt,0,0.5*dt*dt,0], [0,1,0,dt,0,0.5*dt*dt],
            [0,0,1,0,dt,0], [0,0,0,1,0,dt],
            [0,0,0,0,1,0], [0,0,0,0,0,1]
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

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__("yolov8_node")
        self.get_logger().info("Initializing YOLOv8Node with SAHI, Kalman Filter & Advanced Features...")

        # --- ROS Parameters ---
        self.declare_parameter("model_path", "/home/ultraviolet/CV_Detection/yolov8_ros/models/bestObb.pt") # Default if not overridden
        self.declare_parameter("task", "detect")
        self.declare_parameter("type", "yolo")  
        self.declare_parameter("half", True)    
        self.declare_parameter("imgsz", 640)    
        self.declare_parameter("show", True)    
        self.declare_parameter("conf", 0.25)    
        
        self.declare_parameter("tracker_timeout", 15)      
        self.declare_parameter("kalman_initial_dt", 1.0/30.0) 
        self.declare_parameter("prediction_lookahead_steps", 0.5)
        self.declare_parameter("target_selection_strategy", "first") # "first", "biggest_area", "closest_to_center"

        self.declare_parameter("use_sahi", False) 
        self.declare_parameter("sahi_slice_height", 512)
        self.declare_parameter("sahi_slice_width", 512)
        self.declare_parameter("sahi_overlap_height_ratio", 0.2)
        self.declare_parameter("sahi_overlap_width_ratio", 0.2)
        self.declare_parameter("sahi_nms_iou_threshold", 0.5)

        # Get parameter values
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.task = self.get_parameter("task").get_parameter_value().string_value
        model_type = self.get_parameter("type").get_parameter_value().string_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.imgsz = self.get_parameter("imgsz").get_parameter_value().integer_value
        self.show = self.get_parameter("show").get_parameter_value().bool_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.tracker_timeout_val = self.get_parameter("tracker_timeout").get_parameter_value().integer_value
        self.kalman_initial_dt_val = self.get_parameter("kalman_initial_dt").get_parameter_value().double_value
        self.prediction_lookahead_steps_val = self.get_parameter("prediction_lookahead_steps").get_parameter_value().double_value
        self.selection_strategy = self.get_parameter("target_selection_strategy").get_parameter_value().string_value
        
        self.use_sahi = self.get_parameter("use_sahi").get_parameter_value().bool_value
        self.sahi_slice_h = self.get_parameter("sahi_slice_height").get_parameter_value().integer_value
        self.sahi_slice_w = self.get_parameter("sahi_slice_width").get_parameter_value().integer_value
        self.sahi_ov_h_ratio = self.get_parameter("sahi_overlap_height_ratio").get_parameter_value().double_value
        self.sahi_ov_w_ratio = self.get_parameter("sahi_overlap_width_ratio").get_parameter_value().double_value
        self.sahi_nms_thresh = self.get_parameter("sahi_nms_iou_threshold").get_parameter_value().double_value

        self.get_logger().info(f"Model Path: {model_path}")
        self.get_logger().info(f"Settings: Task={self.task}, Type={model_type}, ImgSz={self.imgsz}, Conf={self.conf}, Show={self.show}, Half={self.half}")
        self.get_logger().info(f"Kalman: Timeout={self.tracker_timeout_val}, InitialDT={self.kalman_initial_dt_val:.4f}, LookAhead={self.prediction_lookahead_steps_val}")
        self.get_logger().info(f"Target Selection: {self.selection_strategy}")
        if self.use_sahi:
            self.get_logger().info(f"SAHI Enabled: Slice({self.sahi_slice_h}x{self.sahi_slice_w}), Overlap({self.sahi_ov_h_ratio:.2f}x{self.sahi_ov_w_ratio:.2f}), NMS_Thresh={self.sahi_nms_thresh:.2f}")
        
        self.target_class = BLUE_CLASS 
        if model_type == "yolo": self.model = YOLO(model_path, task=self.task)
        elif model_type == "rtdetr": self.model = RTDETR(model_path)
        elif model_type == "nas": self.model = NAS(model_path)
        else: self.get_logger().error(f"Unsupported model: {model_type}"); raise NotImplementedError
        self.get_logger().info(f"YOLO Model '{model_type}' initialized.")

        self.subscription = self.create_subscription(Image, "video_frames", self.listener_callback, 10)
        self.bridge = CvBridge()
        self.int_subscription = self.create_subscription(Int32, "robot_id", self.id_listener_callback, 10)
        self.bbox_publisher = self.create_publisher(Int32MultiArray, "bounding_box", 10)
        self.override_publisher = self.create_publisher(Bool, "override", 10)
        self.get_logger().info("ROS I/O initialized.")

        self.override = Bool(); self.bbox_msg = Int32MultiArray()
        default_center = self.imgsz // 2
        self.bbox_msg.data = [default_center, default_center, default_center, default_center]
        self.active_trackers = {}; self.last_frame_time = None

    def reduce_white_regions(self, hsv): # Modifies hsv in-place
        white_mask = cv2.inRange(hsv, np.array([0,0,200]), np.array([180,30,255]))
        hsv[:,:,2][white_mask > 0] = (hsv[:,:,2][white_mask > 0] * 0.7).astype(np.uint8)

    def create_color_mask(self, hsv, color_ranges):
        mask = np.zeros_like(hsv[:,:,0])
        for low, high in color_ranges: mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
        return mask

    def enhance_image(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.reduce_white_regions(hsv)
        color_mask = cv2.bitwise_or(self.create_color_mask(hsv, RED_RANGES), self.create_color_mask(hsv, BLUE_RANGES))
        hsv[:,:,2] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(hsv[:,:,2])
        enhanced_hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        color_regions = cv2.bitwise_and(enhanced_hsv_bgr, enhanced_hsv_bgr, mask=color_mask)
        result = cv2.addWeighted(enhanced_hsv_bgr, 0.7, color_regions, 0.3, 0)
        return cv2.bilateralFilter(result, 9, 75, 75)

    def listener_callback(self, msg: Image):
        current_time = self.get_clock().now(); dt = self.kalman_initial_dt_val 
        if self.last_frame_time:
            time_diff_s=(current_time-self.last_frame_time).nanoseconds/1e9
            if 0.001 < time_diff_s < 1.0: dt = time_diff_s
            else: self.get_logger().warn(f"Unusual dt ({time_diff_s:.4f}s), using fallback: {dt:.4f}s")
        self.last_frame_time = current_time; self.get_logger().debug(f"dt: {dt:.4f}s")

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        original_height, original_width = frame.shape[:2]
        if self.show: cv2.imshow("1. Raw Input", frame); cv2.waitKey(1)
        # enhanced_frame = self.enhance_image(frame.copy())
        enhanced_frame = frame 

        # --- SAHI Processing (Visualized Separately For Now) ---
        if self.use_sahi and original_width > 0 and original_height > 0:
            self.get_logger().info("--- SAHI Slicing ---")
            slice_gen = slice_image(image=enhanced_frame, slice_height=self.sahi_slice_h, slice_width=self.sahi_slice_w,
                                    overlap_height_ratio=self.sahi_ov_h_ratio, overlap_width_ratio=self.sahi_ov_w_ratio)
            raw_sahi_obj_preds = []
            for i, slice_item in enumerate(slice_gen):
                slice_np, (start_x,start_y) = slice_item.image, slice_item.starting_pixel
                if not (slice_np is None or slice_np.shape[0]==0 or slice_np.shape[1]==0):
                    preds_on_slice = self.model.predict(source=slice_np, conf=self.conf, half=self.half, verbose=False)
                    if preds_on_slice and preds_on_slice[0].boxes is not None and preds_on_slice[0].boxes.xyxy is not None:
                        for j in range(len(preds_on_slice[0].boxes.xyxy)):
                            raw_sahi_obj_preds.append(ObjectPrediction(
                                bbox=preds_on_slice[0].boxes.xyxy.cpu().numpy()[j].tolist(),
                                category_id=int(preds_on_slice[0].boxes.cls.cpu().numpy()[j]),
                                score=float(preds_on_slice[0].boxes.conf.cpu().numpy()[j]),
                                shift_amount=[start_x, start_y], full_shape=[original_height, original_width]))
            self.get_logger().debug(f"SAHI: {len(raw_sahi_obj_preds)} raw slice detections.")
            if raw_sahi_obj_preds:
                sahi_pred_res = PredictionResult(raw_sahi_obj_preds, (original_height,original_width))
                sahi_merged = sahi_pred_res.get_merged_predictions("NMS", self.sahi_nms_thresh, "IOU")
                self.get_logger().info(f"SAHI: {len(sahi_merged.object_prediction_list)} final NMS detections.")
                if self.show and sahi_merged.object_prediction_list:
                    sahi_vis = frame.copy()
                    for op in sahi_merged.object_prediction_list:
                        x1,y1,x2,y2 = op.bbox.to_xyxy()
                        cv2.rectangle(sahi_vis, (int(x1),int(y1)),(int(x2),int(y2)), BBOX_SAHI_COLOR, BBOX_LINE_SIZE)
                        cv2.putText(sahi_vis, f"S:{op.category_id} {op.score:.2f}", (int(x1),int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, BBOX_SAHI_COLOR,1)
                    cv2.imshow("4. SAHI NMS (Original Img)", sahi_vis); cv2.waitKey(1)
        # --- End SAHI ---

        # --- Full-Frame YOLO + Kalman Filter Pipeline ---
        target_h, target_w = self.imgsz, self.imgsz; scale = 0.0
        if original_width > 0 and original_height > 0: scale = min(target_w/original_width, target_h/original_height)
        new_w, new_h = int(original_width*scale), int(original_height*scale)
        padded_image = np.full((self.imgsz,self.imgsz,3), (114,114,114), dtype=np.uint8)
        if new_w > 0 and new_h > 0:
            dw,dh=(target_w-new_w)//2,(target_h-new_h)//2
            padded_image[dh:dh+new_h, dw:dw+new_w] = cv2.resize(enhanced_frame,(new_w,new_h), interpolation=cv2.INTER_LINEAR)
        else: padded_image = cv2.resize(enhanced_frame,(self.imgsz,self.imgsz),interpolation=cv2.INTER_LINEAR)
        if self.show: cv2.imshow("2. Padded Full YOLO Input", padded_image); cv2.waitKey(1)

        results = self.model.track(padded_image, half=self.half, imgsz=self.imgsz, show=False, conf=self.conf, persist=True, tracker='bytetrack.yaml')
        
        vis_yolo_kf = padded_image.copy() if self.show else None
        current_frame_track_ids = set()
        if results and results[0]:
            boxes_data = results[0].boxes if hasattr(results[0],"boxes") and results[0].boxes else (results[0].obb if hasattr(results[0],"obb") and results[0].obb and self.task=="obb" else None)
            if boxes_data and hasattr(boxes_data,'xyxy') and boxes_data.xyxy is not None:
                for i, (x1,y1,x2,y2) in enumerate(boxes_data.xyxy.cpu().int().tolist()):
                    track_id = int(boxes_data.id.cpu().numpy()[i]) if hasattr(boxes_data,'id') and boxes_data.id is not None else None
                    if self.show and vis_yolo_kf: cv2.rectangle(vis_yolo_kf,(x1,y1),(x2,y2),BBOX_RAW_COLOR,1); cv2.putText(vis_yolo_kf,f"R:{track_id}",(x1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.4,BBOX_RAW_COLOR,1)
                    roi=padded_image[max(0,y1):min(y2,self.imgsz),max(0,x1):min(x2,self.imgsz)]
                    if roi.shape[0]==0 or roi.shape[1]==0: continue
                    is_enemy,_ = self.check_enemy(roi)
                    if is_enemy and track_id is not None:
                        current_frame_track_ids.add(track_id)
                        cx,cy,w,h = (x1+x2)/2.0,(y1+y2)/2.0,float(abs(x2-x1)),float(abs(y2-y1))
                        if track_id not in self.active_trackers: self.active_trackers[track_id]=TrackerWrapper(ArmorPlateTracker(cx,cy,w,h))
                        else: self.active_trackers[track_id].tracker.update(cx,cy,w,h)
                        self.active_trackers[track_id].lost_frames=0
        
        active_ids_to_remove=[]; valid_kf_preds=[]
        fcx,fcy=self.imgsz/2.0,self.imgsz/2.0
        for tid,tw in self.active_trackers.items():
            if tid not in current_frame_track_ids: tw.lost_frames+=1
            if tw.lost_frames >= self.tracker_timeout_val: active_ids_to_remove.append(tid); continue
            if tw.lost_frames==0:
                pcx,pcy=tw.tracker.predict(dt); state=tw.tracker.kf.statePost.flatten()
                vx,vy,ax,ay=state[2],state[3],state[4],state[5]
                look_t=self.prediction_lookahead_steps_val*dt
                pfx,pfy=pcx+vx*look_t+0.5*ax*look_t**2, pcy+vy*look_t+0.5*ay*look_t**2
                kw,kh=tw.tracker.width,tw.tracker.height
                if kw<=0 or kh<=0: continue
                x1,y1,x2,y2=[max(0,min(int(c),self.imgsz-1)) for c in [pfx-kw/2,pfy-kh/2,pfx+kw/2,pfy+kh/2]]
                if not(x1<x2 and y1<y2): continue
                valid_kf_preds.append({"id":tid,"box_coords":[x1,y1,x2,y2],"area":kw*kh,"dist_sq_to_center":(pfx-fcx)**2+(pfy-fcy)**2})
                if self.show and vis_yolo_kf: cv2.rectangle(vis_yolo_kf,(x1,y1),(x2,y2),BBOX_KF_COLOR,1); cv2.putText(vis_yolo_kf,f"K:{tid}",(x1,y1-2),cv2.FONT_HERSHEY_SIMPLEX,0.4,BBOX_KF_COLOR,1)
        
        sel_pred=None
        if valid_kf_preds:
            if self.selection_strategy=="biggest_area": sel_pred=max(valid_kf_preds,key=lambda p:p["area"])
            elif self.selection_strategy=="closest_to_center": sel_pred=min(valid_kf_preds,key=lambda p:p["dist_sq_to_center"])
            else: sel_pred=valid_kf_preds[0] # Default "first"
        
        pub_this_frame = False
        if sel_pred:
            self.bbox_msg.data=sel_pred["box_coords"]; self.override.data=True; pub_this_frame=True
            self.get_logger().info(f"Pub KF ID {sel_pred['id']} Strat:{self.selection_strategy} Box:{sel_pred['box_coords']}")
            if self.show and vis_yolo_kf: cv2.rectangle(vis_yolo_kf,(sel_pred["box_coords"][0],sel_pred["box_coords"][1]),(sel_pred["box_coords"][2],sel_pred["box_coords"][3]),BBOX_SELECTED_KF_COLOR,2)
        if not pub_this_frame: 
            ctr=self.imgsz//2; self.bbox_msg.data=[ctr,ctr,ctr,ctr]; self.override.data=False
        self.bbox_publisher.publish(self.bbox_msg); self.override_publisher.publish(self.override)
        
        for tid_rem in active_ids_to_remove:
            if tid_rem in self.active_trackers: del self.active_trackers[tid_rem]
        if self.show and vis_yolo_kf: cv2.imshow("3. Full YOLO + KF", vis_yolo_kf); cv2.waitKey(1)

    def id_listener_callback(self, msg: Int32):
        rid=msg.data; self.get_logger().info(f"RobotID: {rid}")
        if rid in RED_TEAM: self.target_class=BLUE_CLASS; self.get_logger().info(f"ID {rid} RED. Target BLUE.")
        elif rid in BLUE_TEAM: self.target_class=RED_CLASS; self.get_logger().info(f"ID {rid} BLUE. Target RED.")
        else: self.get_logger().warn(f"ID {rid} unknown. Target: {'BLUE' if self.target_class==BLUE_CLASS else 'RED'}")

    def check_enemy(self, roi):
        if roi.shape[0]==0 or roi.shape[1]==0: return False,None
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        rng=BLUE_RANGES if self.target_class==BLUE_CLASS else RED_RANGES
        mask=np.zeros_like(hsv[:,:,0])
        for r_low,r_high in rng: mask=cv2.bitwise_or(mask,cv2.inRange(hsv,r_low,r_high))
        area=np.count_nonzero(mask)
        thr=roi.shape[0]*roi.shape[1]*0.01 # 1% threshold
        is_e=area>thr
        self.get_logger().debug(f"check_enemy: Tgt:{'B' if self.target_class==BLUE_CLASS else 'R'}. ROI{roi.shape} Area:{area} Thr:{thr:.1f} -> {is_e}")
        return is_e,area

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Node()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info("KI Shutdown.")
    finally:
        node.destroy_node(); 
        if rclpy.ok(): rclpy.shutdown()
        cv2.destroyAllWindows(); node.get_logger().info("Cleanup done.")

if __name__ == "__main__":
    main()