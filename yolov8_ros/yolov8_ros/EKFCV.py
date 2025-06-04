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

# COLOR_THRESHOLD = 200 # This was unused, check_enemy uses dynamic threshold

# ──────────────────────────────────────────────────────────────
# Kalman Filter Tracker Classes
# ──────────────────────────────────────────────────────────────
class ArmorPlateTracker:
    def __init__(self, initial_x, initial_y, width, height, dt=1.0):
        self.kf = cv2.KalmanFilter(6, 2)  # 6 states: (x, y, vx, vy, ax, ay), 2 measurements (x,y)
        self.dt = dt

        self.kf.transitionMatrix = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt*self.dt, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt*self.dt],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)

        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], np.float32)

        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.2
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1

        self.kf.statePost = np.array([
            np.float32(initial_x), np.float32(initial_y), 0, 0, 0, 0
        ], np.float32).reshape(6, 1)

        self.color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        self.width = float(width)
        self.height = float(height)

    def update(self, x, y, width, height):
        measurement = np.array([[np.float32(x)], [np.float32(y)]], np.float32)
        self.kf.correct(measurement)
        self.width = float(width)
        self.height = float(height)

    def predict(self):
        prediction = self.kf.predict()
        return int(prediction[0, 0]), int(prediction[1, 0])

class TrackerWrapper:
    def __init__(self, tracker: ArmorPlateTracker):
        self.tracker = tracker
        self.lost_frames = 0
# ──────────────────────────────────────────────────────────────

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__("yolov8_node")
        self.get_logger().info("Initializing YOLOv8Node with Kalman Filter Integration...")

        self.declare_parameter("model_path", "/home/ultraviolet/CV_Detection/yolov8_ros/yolov8_ros/best.engine")
        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.get_logger().info(f"Model path: {model_path}")

        self.declare_parameter("task", "detect")
        self.declare_parameter("type", "yolo")
        self.declare_parameter("half", True)
        self.declare_parameter("imgsz", 640)
        # CHANGED DEFAULTS FOR EASIER DEBUGGING:
        self.declare_parameter("show", True)
        self.declare_parameter("conf", 0.25) # LOWERED DEFAULT CONFIDENCE
        self.declare_parameter("tracker_timeout", 15) # Frames to wait
        self.declare_parameter("kalman_dt", 1.0/30.0) # Assumed dt for KF (e.g., for 30fps)

        model_type = self.get_parameter("type").get_parameter_value().string_value
        self.task = self.get_parameter("task").get_parameter_value().string_value
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.imgsz = self.get_parameter("imgsz").get_parameter_value().integer_value
        self.show = self.get_parameter("show").get_parameter_value().bool_value
        self.conf = self.get_parameter("conf").get_parameter_value().double_value
        self.tracker_timeout_val = self.get_parameter("tracker_timeout").get_parameter_value().integer_value
        self.kalman_dt_val = self.get_parameter("kalman_dt").get_parameter_value().double_value

        self.get_logger().info(f"Model type: {model_type}, Task: {self.task}")
        self.get_logger().info(f"Confidence threshold: {self.conf}, Img Size: {self.imgsz}, Show: {self.show}")
        self.get_logger().info(f"Tracker Timeout: {self.tracker_timeout_val}, Kalman dt: {self.kalman_dt_val:.4f}")


        self.target_class = BLUE_CLASS # Default target

        if model_type == "yolo":
            self.model = YOLO(model_path, task=self.task)
        elif model_type == "rtdetr":
            self.model = RTDETR(model_path)
        elif model_type == "nas":
            self.model = NAS(model_path)
        else:
            self.get_logger().error(f"Unsupported model type specified: {model_type}")
            raise NotImplementedError("Unsupported model type specified.")
        self.get_logger().info(f"YOLO Model initialized.")

        self.subscription = self.create_subscription(Image, "video_frames", self.listener_callback, 10)
        self.get_logger().info("Image subscriber created on 'video_frames'.")
        self.bridge = CvBridge()

        self.int_subscription = self.create_subscription(Int32, "robot_id", self.id_listener_callback, 10)
        self.get_logger().info("Robot ID subscriber created on 'robot_id'.")

        self.bbox_publisher = self.create_publisher(Int32MultiArray, "bounding_box", 10)
        self.override_publisher = self.create_publisher(Bool, "override", 10)
        self.override = Bool()
        self.bbox_msg = Int32MultiArray()
        # Default to center of self.imgsz (which is also padded_image size)
        default_center = self.imgsz // 2
        self.bbox_msg.data = [default_center, default_center, default_center, default_center]

        # Kalman Filter specific members
        self.active_trackers = {}  # Stores {track_id: TrackerWrapper}

    def reduce_white_regions(self, hsv):
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        hsv[:, :, 2][white_mask > 0] = (hsv[:, :, 2][white_mask > 0] * 0.7).astype(np.uint8)
        return white_mask # Not used by caller, HSV modified in-place

    def create_color_mask(self, hsv, color_ranges):
        mask = np.zeros_like(hsv[:, :, 0])
        for low, high in color_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, low, high))
        return mask

    def enhance_image(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.reduce_white_regions(hsv)
        red_mask = self.create_color_mask(hsv, RED_RANGES)
        blue_mask = self.create_color_mask(hsv, BLUE_RANGES)
        color_mask = cv2.bitwise_or(red_mask, blue_mask)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        enhanced_hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        color_regions = cv2.bitwise_and(enhanced_hsv_bgr, enhanced_hsv_bgr, mask=color_mask)
        result = cv2.addWeighted(enhanced_hsv_bgr, 0.7, color_regions, 0.3, 0)
        result = cv2.bilateralFilter(result, 9, 75, 75)
        return result

    def listener_callback(self, msg):
        self.get_logger().debug("Received image message.")
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        original_height, original_width = frame.shape[:2]
        self.get_logger().debug(f"Original frame shape: {original_height}x{original_width}")

        if self.show: # Show raw input
            cv2.imshow("Raw Input", frame)
            cv2.waitKey(1)

        # --- Image Enhancement (Optional) ---
        # enhanced_frame = self.enhance_image(frame.copy())
        enhanced_frame = frame # Using original frame

        # --- Robust Padding/Resizing (Letterboxing) to self.imgsz ---
        target_h, target_w = self.imgsz, self.imgsz
        scale = min(target_w / original_width, target_h / original_height)
        new_w, new_h = int(original_width * scale), int(original_height * scale)
        
        # Create a blank canvas (e.g., gray) and place the resized image
        # Ultralytics typically uses (114, 114, 114) for letterbox padding
        padded_image = np.full((self.imgsz, self.imgsz, 3), (114, 114, 114), dtype=np.uint8)
        
        if new_w > 0 and new_h > 0: # Ensure dimensions are positive
            resized_image = cv2.resize(enhanced_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2 # Padding on each side
            padded_image[dh:dh+new_h, dw:dw+new_w] = resized_image
        else:
            self.get_logger().warn(f"Cannot resize image to valid new dimensions ({new_w}x{new_h}). Using unpadded/resized original.")
            # Fallback: try to resize directly, may distort aspect ratio if not square
            padded_image = cv2.resize(enhanced_frame, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        self.get_logger().debug(f"Padded image shape for YOLO: {padded_image.shape}")
        if self.show:
            cv2.imshow("Padded Input for YOLO", padded_image)
            cv2.waitKey(1)

        # --- YOLO Model Tracking ---
        self.get_logger().debug(f"Calling model.track() with conf: {self.conf}")
        results = self.model.track(
            padded_image,
            half=self.half,
            imgsz=self.imgsz, # Model expects this size
            show=False,       # We handle showing manually
            conf=self.conf,
            persist=True,     # Persist tracks between frames
            tracker='bytetrack.yaml' # Explicitly specify a common tracker
        )
        
        vis_frame = padded_image.copy() if self.show else None
        current_frame_processed_track_ids = set()
        published_bbox_this_frame = False

        if results and len(results) > 0:
            r = results[0] # Assuming single image result; Ultralytics returns a list
            self.get_logger().debug(f"YOLO results[0] type: {type(r)}")
            
            boxes_data = None
            if hasattr(r, "boxes") and r.boxes is not None: # Standard for detection
                boxes_data = r.boxes
            elif hasattr(r, "obb") and r.obb is not None and self.task=="obb": # For OBB tasks
                 boxes_data = r.obb # Note: .obb might have .xywhr, .xyxy for AABB
            
            if boxes_data is not None and hasattr(boxes_data, 'xyxy') and boxes_data.xyxy is not None:
                bbox_coords_list = boxes_data.xyxy.cpu().int().tolist()
                self.get_logger().debug(f"Number of raw boxes from YOLO: {len(bbox_coords_list)}")

                track_ids_list = []
                if hasattr(boxes_data, 'id') and boxes_data.id is not None:
                    track_ids_list = boxes_data.id.cpu().int().tolist()
                    self.get_logger().debug(f"Track IDs from YOLO: {track_ids_list}")
                else:
                    track_ids_list = [None] * len(bbox_coords_list)
                    self.get_logger().debug("No track IDs provided by YOLO's tracker.")

                for i, bbox_xyxy in enumerate(bbox_coords_list):
                    track_id = track_ids_list[i]
                    x1, y1, x2, y2 = bbox_xyxy
                    self.get_logger().debug(f"Raw detection {i} (ID: {track_id}): Box[{x1},{y1},{x2},{y2}]")

                    if self.show and vis_frame is not None: # Draw raw YOLO detection
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), BBOX_RAW_COLOR, BBOX_LINE_SIZE)
                        if track_id is not None:
                             cv2.putText(vis_frame, f"ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_RAW_COLOR, 1)

                    roi_img = padded_image[max(0,y1):min(y2, self.imgsz), max(0,x1):min(x2, self.imgsz)]
                    if roi_img.shape[0] == 0 or roi_img.shape[1] == 0:
                        self.get_logger().debug(f"Raw detection {i}: ROI is empty, skipping.")
                        continue
                    
                    # cv2.imwrite(f"/tmp/roi_det_{i}_id_{track_id}.png", roi_img) # Uncomment to save ROIs

                    is_enemy, color_area_val = self.check_enemy(roi_img)
                    self.get_logger().debug(f"Raw detection {i} (ID: {track_id}): check_enemy result: {is_enemy} (Target: {'BLUE' if self.target_class == BLUE_CLASS else 'RED'})")

                    if is_enemy and track_id is not None: # Require a track_id to use Kalman filter
                        current_frame_processed_track_ids.add(track_id)
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        width = float(abs(x2 - x1))
                        height = float(abs(y2 - y1))

                        if track_id not in self.active_trackers:
                            self.get_logger().info(f"Creating new Kalman tracker for enemy ID: {track_id}")
                            kf_tracker = ArmorPlateTracker(center_x, center_y, width, height, dt=self.kalman_dt_val)
                            self.active_trackers[track_id] = TrackerWrapper(kf_tracker)
                        else:
                            self.get_logger().debug(f"Updating Kalman tracker for enemy ID: {track_id}")
                            self.active_trackers[track_id].tracker.update(center_x, center_y, width, height)
                        
                        self.active_trackers[track_id].lost_frames = 0
                    elif is_enemy and track_id is None:
                        self.get_logger().debug(f"Raw detection {i} is an enemy but has no track ID. Not Kalman tracking this instance.")
            else:
                self.get_logger().debug("No bounding boxes (boxes_data.xyxy is None or boxes_data is None) in YOLO results[0].")
        else:
            self.get_logger().debug("YOLO model.track() returned None or empty list of results.")

        # --- Update and Publish Kalman Filtered Tracks ---
        active_ids_to_remove = []
        for track_id, tw_wrapper in self.active_trackers.items():
            if track_id not in current_frame_processed_track_ids:
                tw_wrapper.lost_frames += 1
                self.get_logger().debug(f"Track ID {track_id} not seen. Lost frames: {tw_wrapper.lost_frames}")


            if tw_wrapper.lost_frames >= self.tracker_timeout_val:
                active_ids_to_remove.append(track_id)
                self.get_logger().info(f"Removing tracker ID: {track_id} due to timeout ({tw_wrapper.lost_frames} frames).")
                continue

            if not published_bbox_this_frame and tw_wrapper.lost_frames == 0: # Publish first active, recently seen track
                pred_x, pred_y = tw_wrapper.tracker.predict()
                kf_width = tw_wrapper.tracker.width
                kf_height = tw_wrapper.tracker.height

                pub_x1 = int(pred_x - kf_width / 2)
                pub_y1 = int(pred_y - kf_height / 2)
                pub_x2 = int(pred_x + kf_width / 2)
                pub_y2 = int(pred_y + kf_height / 2)
                
                # Clamp to image boundaries (of padded_image, which is self.imgsz x self.imgsz)
                pub_x1 = max(0, min(pub_x1, self.imgsz -1))
                pub_y1 = max(0, min(pub_y1, self.imgsz -1))
                pub_x2 = max(0, min(pub_x2, self.imgsz -1))
                pub_y2 = max(0, min(pub_y2, self.imgsz -1))

                if pub_x1 >= pub_x2 or pub_y1 >= pub_y2: # Check for invalid box
                    self.get_logger().warn(f"KF for ID {track_id} produced invalid box: {[pub_x1, pub_y1, pub_x2, pub_y2]}. Skipping publish for this track.")
                else:
                    self.bbox_msg.data = [pub_x1, pub_y1, pub_x2, pub_y2]
                    self.override.data = True # Valid KF prediction to publish
                    self.bbox_publisher.publish(self.bbox_msg)
                    self.override_publisher.publish(self.override)
                    published_bbox_this_frame = True
                    self.get_logger().info(f"Published KF BBox for ID {track_id}: {[pub_x1, pub_y1, pub_x2, pub_y2]}")

                    if self.show and vis_frame is not None:
                        cv2.rectangle(vis_frame, (pub_x1, pub_y1), (pub_x2, pub_y2), BBOX_KF_COLOR, BBOX_LINE_SIZE)
                        cv2.putText(vis_frame, f"KF_ID:{track_id}", (pub_x1, pub_y1 - 25 if pub_y1 > 25 else pub_y1 + int(kf_height) + 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BBOX_KF_COLOR, 1)
        
        for track_id_rem in active_ids_to_remove:
            if track_id_rem in self.active_trackers:
                del self.active_trackers[track_id_rem]

        if not published_bbox_this_frame:
            default_center = self.imgsz // 2 # Recalculate just in case, though it's const
            self.bbox_msg.data = [default_center, default_center, default_center, default_center]
            self.override.data = False # No valid KF target published
            self.bbox_publisher.publish(self.bbox_msg)
            self.override_publisher.publish(self.override)
            self.get_logger().debug("No KF-tracked enemy published this frame. Sent default bbox.")

        if self.show and vis_frame is not None:
            cv2.imshow("YOLOv8 Kalman Tracking", vis_frame)
            cv2.waitKey(1) # Essential for imshow to update

    def id_listener_callback(self, msg):
        robot_id = msg.data
        self.get_logger().info(f"Received robot_id: {robot_id}")
        if robot_id in RED_TEAM:
            self.target_class = BLUE_CLASS
            self.get_logger().info(f"Robot ID {robot_id} is RED_TEAM. Targeting BLUE_CLASS.")
        elif robot_id in BLUE_TEAM:
            self.target_class = RED_CLASS
            self.get_logger().info(f"Robot ID {robot_id} is BLUE_TEAM. Targeting RED_CLASS.")
        else:
            self.get_logger().warn(f"Robot ID {robot_id} not in known teams. Current target_class: {'BLUE' if self.target_class == BLUE_CLASS else 'RED'}")


    def check_enemy(self, roi_image):
        if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
            self.get_logger().warn("check_enemy: Invalid ROI shape (0 dimension); returning False.")
            return False, None

        hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        target_ranges = BLUE_RANGES if self.target_class == BLUE_CLASS else RED_RANGES
        
        final_mask = np.zeros_like(hsv_roi[:,:,0]) # Ensure it's the same size as one channel of hsv_roi
        for hsv_range_idx, hsv_range in enumerate(target_ranges):
            current_mask = cv2.inRange(hsv_roi, hsv_range[0], hsv_range[1])
            final_mask = cv2.bitwise_or(final_mask, current_mask)
            # self.get_logger().debug(f"check_enemy: TargetClass {'BLUE' if self.target_class == BLUE_CLASS else 'RED'}, Range {hsv_range_idx}: PixelsInRange: {np.count_nonzero(current_mask)}")

        color_area = np.count_nonzero(final_mask)
        
        # Using the 0.01 (1%) threshold from your original code
        min_color_area_threshold = roi_image.shape[0] * roi_image.shape[1] * 0.01 

        is_it_enemy = color_area > min_color_area_threshold
        
        # More detailed log for check_enemy decision
        # This log level can be reduced to DEBUG if too verbose during normal operation
        self.get_logger().debug( 
            f"check_enemy: Target: {'BLUE' if self.target_class == BLUE_CLASS else 'RED'}. "
            f"ROI Shape: {roi_image.shape}, Color Area: {color_area}, Threshold Req: {min_color_area_threshold:.2f}. Result: {is_it_enemy}"
        )
        return is_it_enemy, color_area


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()