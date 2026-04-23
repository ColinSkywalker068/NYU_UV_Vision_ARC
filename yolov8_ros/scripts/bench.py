#!/usr/bin/env python3
"""
Jetson TensorRT benchmark — bestmerge.engine (OBB)
Safe optimizations only:
  - High-priority CUDA stream
  - Python GC disabled during bench
  - Pre-resized frame (skip resize every call)
  - No persistent predictor (causes heap corruption with TRT)
  - No pre-uploaded tensor (causes heap corruption with TRT)
"""
import gc
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ASSETS

MODEL  = "/home/ultraviolet/CV_Detection/model_test/bestmerge.engine"
IMGSZ  = 640
CONF   = 0.5
DEVICE = "0"
WARMUP = 30
RUNS   = 300


def sync() -> None:
    torch.cuda.synchronize()


def main() -> None:
    # -- High-priority CUDA stream -------------------------------------------
    stream = torch.cuda.Stream(priority=-1)
    torch.cuda.set_stream(stream)

    # -- Load model ----------------------------------------------------------
    model = YOLO(MODEL, task="obb")

    # -- Load and pre-resize frame once (avoid resize overhead in hot loop) --
    asset_dir = Path(ASSETS)
    img_path = next(
        (p for p in sorted(asset_dir.glob("*"))
         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        None,
    )
    if img_path is None:
        raise RuntimeError(f"No images in {asset_dir}")

    raw = cv2.imread(str(img_path))
    if raw is None:
        raise RuntimeError(f"Could not read {img_path}")

    # Pre-resize to exact input size so Ultralytics does zero work on it
    frame = cv2.resize(raw, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    frame = np.ascontiguousarray(frame)   # ensure contiguous memory layout

    print(f"Model : {MODEL}")
    print(f"Task  : obb")
    print(f"Image : {img_path.name} (pre-resized to {IMGSZ}x{IMGSZ})")
    print(f"imgsz : {IMGSZ}  conf : {CONF}  device : {DEVICE}  half : True")
    print(f"Warmup: {WARMUP}  Runs: {RUNS}\n")

    # -- Flush before bench --------------------------------------------------
    sync()
    torch.cuda.empty_cache()
    time.sleep(1.0)

    latencies: list[float] = []

    # -- Disable GC for hot loop ---------------------------------------------
    gc.disable()

    for i in range(WARMUP + RUNS):
        sync()
        t0 = time.perf_counter()

        results = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            half=True,
            iou=0.45,
            agnostic_nms=True,
            verbose=False,
            stream=False,
        )

        sync()
        t1 = time.perf_counter()

        del results

        if i >= WARMUP:
            latencies.append((t1 - t0) * 1000.0)

    # gc.enable()  -- intentionally omitted: re-enabling GC before TRT cleanup causes heap corruption

    # -- Stats ---------------------------------------------------------------
    arr = np.array(latencies)
    mean_ms = float(arr.mean())

    print("=== Results ===")
    print(f"  mean : {mean_ms:7.2f} ms")
    print(f"  p50  : {float(np.percentile(arr, 50)):7.2f} ms")
    print(f"  p90  : {float(np.percentile(arr, 90)):7.2f} ms")
    print(f"  p99  : {float(np.percentile(arr, 99)):7.2f} ms")
    print(f"  min  : {float(arr.min()):7.2f} ms")
    print(f"  max  : {float(arr.max()):7.2f} ms")
    print(f"  FPS  : {1000.0 / mean_ms:7.2f}")

    sync()
    torch.cuda.empty_cache()
    del model
    torch.cuda.synchronize()
    os._exit(0)


if __name__ == "__main__":
    main()