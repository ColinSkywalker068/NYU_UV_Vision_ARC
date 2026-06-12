#!/usr/bin/env python3
"""Minimal model smoke test for `yolo_cpp/models/latest_best.pt`.

This script keeps the validation narrow: load the model, run one OpenCV-backed
inference pass on either a user-supplied image or a generated test card, and
save an annotated result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_model_path() -> Path:
    return repo_root() / "yolo_cpp" / "models" / "latest_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-frame YOLO smoke test.")
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help="Path to the .pt model (defaults to yolo_cpp/models/latest_best.pt)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional image path to run inference on. If omitted, a synthetic test card is used.",
    )
    parser.add_argument(
        "--task",
        default="obb",
        choices=["detect", "obb", "segment", "pose", "classify"],
        help="Ultralytics task for the model.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/yolo_cpp_model_smoke.jpg"),
        help="Annotated output image path.",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open a window with the annotated result after inference.",
    )
    return parser.parse_args()


def make_test_card(width: int = 1280, height: int = 720) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (18, 18, 18)

    cv2.rectangle(frame, (80, 60), (width - 80, height - 60), (48, 48, 48), 2)
    cv2.line(frame, (width // 2, 0), (width // 2, height), (32, 32, 32), 1)
    cv2.line(frame, (0, height // 2), (width, height // 2), (32, 32, 32), 1)

    armor_like_boxes = [
        ((220, 160), (430, 310), (0, 215, 255)),
        ((760, 210), (970, 360), (0, 128, 255)),
        ((510, 420), (690, 560), (0, 255, 0)),
    ]
    for top_left, bottom_right, color in armor_like_boxes:
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 3)
        center_x = (top_left[0] + bottom_right[0]) // 2
        center_y = (top_left[1] + bottom_right[1]) // 2
        cv2.circle(frame, (center_x, center_y), 8, (255, 255, 255), -1)

    cv2.putText(
        frame,
        "yolo_cpp smoke test",
        (90, height - 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return frame


def load_source_image(source: Optional[Path]) -> np.ndarray:
    if source is None:
        return make_test_card()

    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(f"Failed to read source image: {source}")
    return image


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    frame = load_source_image(args.source)
    model = YOLO(str(args.model), task=args.task)
    results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
    result = results[0]
    annotated = result.plot()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotated):
        raise RuntimeError(f"Failed to write annotated output: {args.output}")

    box_count = len(result.boxes) if result.boxes is not None else 0
    print(f"model={args.model}")
    print(f"source={'synthetic test card' if args.source is None else args.source}")
    print(f"detections={box_count}")
    print(f"saved={args.output}")

    if box_count:
        for i, box in enumerate(result.boxes):
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            xyxy = box.xyxy[0].tolist()
            print(f"  det[{i}] cls={cls_id} conf={conf:.3f} xyxy={xyxy}")

    if args.view:
        cv2.imshow("yolo_cpp model smoke", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
