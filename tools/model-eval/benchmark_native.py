#!/usr/bin/env python3
"""Native ONNX/CoreML replay benchmark, with decoded-output agreement checks.

No camera, serial link or ROS deployment. Timings exclude preprocessing and decoding.
Unlabeled clips measure execution/agreement, never detection accuracy or recall.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(frame, kind, height, width):
    if kind == "aruw-2026":
        # Pinned Project Otto contract: float BGR / 255, stretched, not RGB/letterbox.
        result = cv2.resize(frame.astype(np.float32) / 255, (width, height))
    else:
        scale = min(width / frame.shape[1], height / frame.shape[0])
        nw, nh = round(frame.shape[1] * scale), round(frame.shape[0] * scale)
        result = np.full((height, width, 3), 124, np.uint8)
        x, y = (width - nw) // 2, (height - nh) // 2
        result[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh))
        result = result[..., ::-1].astype(np.float32) / 255
    return np.ascontiguousarray(result.transpose(2, 0, 1)[None])


def decode(raw, kind, threshold):
    detections = []
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite model output")
    if kind == "aruw-2026":
        if raw.shape != (1, 300, 14):
            raise ValueError(f"Unexpected ARUW output shape: {raw.shape}")
        for row in raw[0]:
            if row[4] >= threshold:
                detections.append((int(row[5]), float(row[4]), row[:4], row[6:14]))
    else:
        if raw.shape != (1, 21, 6300):
            raise ValueError(f"Unexpected RobotPilots output shape: {raw.shape}")
        for row in raw[0].T:
            score = float(row[4:13].max())
            if score >= threshold:
                corners = row[13:21].reshape(4, 2)
                box = np.concatenate((corners.min(0), corners.max(0)))
                cls = int(row[:4].argmax()) * 9 + int(row[4:13].argmax())
                detections.append((cls, score, box, row[13:21]))
        if detections:
            boxes = [[float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])]
                     for _, _, b, _ in detections]
            # Class-aware NMS, matching the upstream color/type identity contract.
            keep = []
            for cls in sorted({d[0] for d in detections}):
                indices = [i for i, d in enumerate(detections) if d[0] == cls]
                selected = cv2.dnn.NMSBoxes([boxes[i] for i in indices],
                                           [detections[i][1] for i in indices],
                                           threshold, 0.45)
                keep.extend(indices[int(i)] for i in selected)
            detections = [detections[i] for i in keep]
    return detections


def iou(a, b):
    area = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(area / max(1e-9, np.maximum(0, a[2:]-a[:2]).prod() +
                           np.maximum(0, b[2:]-b[:2]).prod() - area))


def compare(reference, candidate):
    matched, score_errors, corner_errors = 0, [], []
    unused = set(range(len(candidate)))
    for cls, confidence, box, corners in sorted(reference, key=lambda d: -d[1]):
        choices = [(iou(box, candidate[i][2]), i) for i in unused
                   if candidate[i][0] == cls]
        overlap, index = max(choices, default=(0, -1))
        if overlap < 0.5:
            continue
        unused.remove(index)
        matched += 1
        score_errors.append(abs(confidence - candidate[index][1]))
        corner_errors.extend(np.linalg.norm(
            (corners - candidate[index][3]).reshape(4, 2), axis=1).tolist())
    return matched, len(reference) - matched, len(candidate) - matched, score_errors, corner_errors


def profile_summary(path):
    events = json.loads(Path(path).read_text())
    providers = Counter()
    operators = {}
    for event in events:
        provider = event.get("args", {}).get("provider")
        if provider and event.get("name", "").endswith("_kernel_time"):
            providers[provider] += 1
            operators.setdefault(provider, set()).add(event["args"].get("op_name", "?"))
    return {"kernel_events": dict(providers),
            "operators": {p: sorted(ops) for p, ops in operators.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--kind", choices=["aruw-2026", "robotpilots-2026"], default="aruw-2026")
    parser.add_argument("--clip", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-clip", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--modes", nargs="+", default=["cpu", "ALL", "CPUAndGPU", "CPUAndNeuralEngine"],
                        choices=["cpu", "ALL", "CPUAndGPU", "CPUAndNeuralEngine", "CPUOnly"])
    args = parser.parse_args()
    if args.frames_per_clip < 1 or args.warmup < 1 or args.cpu_threads < 1:
        parser.error("Frame, warmup and thread counts must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    manifest = json.loads(Path(__file__).with_name("model-manifest.json").read_text())
    entry = next(m for m in manifest if m["name"].startswith(args.kind + "/"))
    if sha != entry["sha256"]:
        parser.error("Model hash differs from the pinned candidate; review the manifest first")
    height, width = (384, 640) if args.kind == "aruw-2026" else (480, 640)
    tensors, frame_ids = [], []
    for clip in args.clip:
        # Use the same file decoder on macOS/Linux; AVFoundation can stall on AVI.
        cap = cv2.VideoCapture(str(clip), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ValueError(f"Cannot read clip: {clip}")
        # Decode sequentially: some upstream AVI headers report zero frames and
        # random seeking silently repeats frame zero. Seeded reservoir sampling
        # is bounded in memory and samples actual decoded frames uniformly.
        rng = np.random.default_rng(20260930)
        samples = []
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index = count if count < args.frames_per_clip else int(rng.integers(count + 1))
            if index < args.frames_per_clip:
                item = (count, preprocess(frame, args.kind, height, width))
                if index == len(samples):
                    samples.append(item)
                else:
                    samples[index] = item
            count += 1
        if not samples:
            raise ValueError(f"No decodable frames: {clip}")
        for index, tensor in sorted(samples):
            tensors.append(tensor)
            frame_ids.append({"clip": clip.name, "frame": index, "decoded_clip_frames": count})
        cap.release()
        print(f"Loaded {len(samples)} distinct frames from {clip.name} ({count} decoded)", flush=True)
    # Blank is a separate diagnostic and not included in replay timing statistics.
    blank = preprocess(np.zeros((480, 640, 3), np.uint8), args.kind, height, width)
    report = {"scope": "Native inference latency and output agreement; unlabeled, not accuracy or Jetson speed",
              "model": entry["name"], "model_sha256": sha,
              "platform": platform.platform(), "machine": platform.machine(),
              "python": platform.python_version(), "onnxruntime": ort.__version__,
              "numpy": np.__version__, "opencv": cv2.__version__,
              "available_providers": ort.get_available_providers(), "cpu_threads": args.cpu_threads,
              "threshold": args.threshold, "frames": frame_ids, "results": []}
    if platform.system() == "Darwin":
        report["chip"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    reference = None
    # Always establish native CPU reference, even when selecting one CoreML mode.
    for mode in dict.fromkeys(["cpu"] + args.modes):
        result = {"mode": mode}
        report["results"].append(result)
        try:
            if mode != "cpu" and "CoreMLExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("This ONNX Runtime build does not include CoreML")
            providers = ["CPUExecutionProvider"] if mode == "cpu" else [
                ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": mode,
                    "RequireStaticInputShapes": "1", "EnableOnSubgraphs": "0",
                    "ModelCacheDirectory": str((args.output / "cache" / sha / mode).resolve())}),
                "CPUExecutionProvider"]
            def make_session(profile=False):
                options = ort.SessionOptions()
                options.intra_op_num_threads = args.cpu_threads
                options.inter_op_num_threads = 1
                options.enable_profiling = profile
                options.profile_file_prefix = str(args.output / ("profile-" + mode))
                return ort.InferenceSession(str(args.model), options, providers=providers)
            started = time.perf_counter()
            session = make_session()
            result["session_initialization_ms"] = (time.perf_counter() - started) * 1000
            result["registered_providers"] = session.get_providers()
            info = session.get_inputs()[0]
            if info.shape != [1, 3, height, width] or info.type != "tensor(float)":
                raise ValueError(f"Unexpected model input {info.shape}/{info.type}")
            for i in range(args.warmup):
                session.run(None, {info.name: tensors[i % len(tensors)]})
            timings, decoded = [], []
            for tensor in tensors:
                started = time.perf_counter()
                raw = session.run(None, {info.name: tensor})[0]
                timings.append((time.perf_counter() - started) * 1000)
                decoded.append(decode(raw, args.kind, args.threshold))
            result.update({"frames": len(tensors), "median_inference_ms": float(np.median(timings)),
                           "p95_inference_ms": float(np.percentile(timings, 95)),
                           "frames_with_detections": sum(bool(d) for d in decoded),
                           "detection_count": sum(map(len, decoded)),
                           "blank_detections": len(decode(session.run(None, {info.name: blank})[0], args.kind, args.threshold))})
            if reference is None:
                reference = decoded
            else:
                matches = [compare(a, b) for a, b in zip(reference, decoded)]
                scores = [v for m in matches for v in m[3]]
                corners = [v for m in matches for v in m[4]]
                result["cpu_agreement"] = {
                    "matched_detections": sum(m[0] for m in matches),
                    "unmatched_cpu": sum(m[1] for m in matches),
                    "unmatched_accelerated": sum(m[2] for m in matches),
                    "max_score_difference": max(scores, default=None),
                    "max_corner_distance_model_pixels": max(corners, default=None),
                    "p95_corner_distance_model_pixels": float(np.percentile(corners, 95)) if corners else None,
                    "matching": "same class and box IoU >= 0.5; output rows may reorder"}
            del session
            # Profile separately so per-node tracing does not inflate timing measurements.
            session = make_session(profile=True)
            session.run(None, {info.name: tensors[0]})
            result["execution_profile"] = profile_summary(session.end_profiling())
            del session
            result["status"] = "completed"
        except Exception as error:
            result.update(status="error", error=str(error))
        print(json.dumps(result), flush=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(r["status"] == "error" for r in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
