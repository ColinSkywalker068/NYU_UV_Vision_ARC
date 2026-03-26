import cv2
import numpy as np
import math

# --- Constants ---
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


def letterbox_frame(frame: cv2.Mat, target_w: int, target_h: int) -> cv2.Mat:
    """
    Resizes and pads the image to fit the target dimensions while maintaining aspect ratio.
    """
    src_h, src_w = frame.shape[:2]
    if src_h == target_h and src_w == target_w:
        return frame

    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    # Use INTER_LINEAR for speed, or INTER_AREA for quality if shrinking significantly
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return np.ascontiguousarray(padded)


def calculate_angles(x1, y1, x2, y2, imgsz_width, imgsz_height, hfov_rad, vfov_rad):
    """
    Calculates yaw and pitch angles from bounding box coordinates.
    """
    center_x = x1 + (x2 - x1) / 2
    center_y = y1 + (y2 - y1) / 2

    # Normalize pixel offset to [-1, 1] based on image dimensions
    pixel_offset_x = (center_x - imgsz_width / 2) / (imgsz_width / 2)
    pixel_offset_y = (center_y - imgsz_height / 2) / (imgsz_height / 2)

    yaw_angle = math.atan(pixel_offset_x * math.tan(hfov_rad / 2))
    pitch_angle = math.atan(pixel_offset_y * math.tan(vfov_rad / 2))

    return yaw_angle, pitch_angle


def check_enemy(roi_image, target_class):
    """
    Checks if the ROI contains the target color (enemy).
    """
    if roi_image.size == 0:
        return False, 0

    hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
    target_ranges = BLUE_RANGES if target_class == BLUE_CLASS else RED_RANGES

    final_mask = np.zeros_like(hsv_roi[:, :, 0])
    for hsv_range in target_ranges:
        current_mask = cv2.inRange(hsv_roi, hsv_range[0], hsv_range[1])
        final_mask = cv2.bitwise_or(final_mask, current_mask)

    color_area = np.count_nonzero(final_mask)
    min_color_area_threshold = roi_image.shape[0] * roi_image.shape[1] * 0.01

    return color_area > min_color_area_threshold, color_area
