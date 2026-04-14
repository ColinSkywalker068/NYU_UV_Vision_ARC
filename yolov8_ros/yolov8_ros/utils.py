"""Utility functions for the YOLOv8 ROS pipeline.

Provides letterbox preprocessing with invertible metadata, angle
calculation, and HSV-based enemy colour verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RED_TEAM = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
BLUE_TEAM = (101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111)

RED_CLASS = 1
BLUE_CLASS = 0

# HSV colour ranges
BLUE_RANGES = [
    [np.array((100, 150, 150), np.uint8), np.array((128, 255, 255), np.uint8)]
]
RED_RANGES = [
    [np.array((0, 150, 150), np.uint8), np.array((10, 255, 255), np.uint8)],
    [np.array((170, 150, 150), np.uint8), np.array((180, 255, 255), np.uint8)],
]


# ---------------------------------------------------------------------------
# Letterbox preprocessing
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LetterboxResult:
    """Metadata produced by :func:`letterbox_frame`.

    Holds enough information to map model-space coordinates back to
    source-image space via :func:`unletterbox_coords`.
    """

    frame: np.ndarray
    scale: float
    pad_x: int
    pad_y: int
    src_w: int
    src_h: int


def letterbox_frame(frame: np.ndarray, target_w: int, target_h: int) -> LetterboxResult:
    """Resize and pad *frame* to (target_w, target_h), preserving aspect ratio.

    Returns a :class:`LetterboxResult` containing the padded frame and
    the transform metadata needed to invert the operation.
    """
    src_h, src_w = frame.shape[:2]

    if src_h == target_h and src_w == target_w:
        return LetterboxResult(
            frame=frame, scale=1.0, pad_x=0, pad_y=0, src_w=src_w, src_h=src_h
        )

    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
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
    return LetterboxResult(
        frame=np.ascontiguousarray(padded),
        scale=scale,
        pad_x=left,
        pad_y=top,
        src_w=src_w,
        src_h=src_h,
    )


def unletterbox_coords(
    x1: int, y1: int, x2: int, y2: int, lb: LetterboxResult
) -> tuple[int, int, int, int]:
    """Convert model-space (padded) bbox coordinates back to source-image space."""
    if lb.scale <= 0:
        return x1, y1, x2, y2

    x1_src = int(max(0, (x1 - lb.pad_x) / lb.scale))
    y1_src = int(max(0, (y1 - lb.pad_y) / lb.scale))
    x2_src = int(min(lb.src_w, (x2 - lb.pad_x) / lb.scale))
    y2_src = int(min(lb.src_h, (y2 - lb.pad_y) / lb.scale))
    return x1_src, y1_src, x2_src, y2_src


# ---------------------------------------------------------------------------
# Angle calculation
# ---------------------------------------------------------------------------
def calculate_angles(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
    hfov_rad: float,
    vfov_rad: float,
) -> tuple[float, float]:
    """Calculate yaw and pitch from a bounding box in image coordinates.

    *img_w* / *img_h* must be the dimensions of the coordinate space the
    box lives in (source-image space after un-letterboxing).
    """
    center_x = x1 + (x2 - x1) / 2
    center_y = y1 + (y2 - y1) / 2

    pixel_offset_x = (center_x - img_w / 2) / (img_w / 2)
    pixel_offset_y = (center_y - img_h / 2) / (img_h / 2)

    yaw = -math.atan(pixel_offset_x * math.tan(hfov_rad / 2))
    pitch = math.atan(pixel_offset_y * math.tan(vfov_rad / 2))
    return yaw, pitch


# ---------------------------------------------------------------------------
# HSV colour verification
# ---------------------------------------------------------------------------
def check_enemy(roi_image: np.ndarray, target_class: int) -> tuple[bool, int]:
    """Check whether *roi_image* contains the target enemy colour.

    Returns ``(is_enemy, colour_pixel_count)``.
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
    min_threshold = roi_image.shape[0] * roi_image.shape[1] * 0.01
    return color_area > min_threshold, color_area
