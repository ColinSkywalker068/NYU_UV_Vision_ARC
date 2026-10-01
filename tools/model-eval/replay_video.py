#!/usr/bin/env python3
"""Native ARUW inference and video overlays for the production C++ replay tool.

Stages: infer -> armor_replay (built with ROS tests) -> render.
Only pixels are inferred here. No guessed calibration, motor command or fire output.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import onnxruntime as ort

from benchmark_native import preprocess


def infer(args):
    args.output.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    manifest = json.loads(Path(__file__).with_name("model-manifest.json").read_text())
    expected = next(m for m in manifest if m["name"].startswith("aruw-2026/"))
    if sha != expected["sha256"]:
        raise ValueError("Use the verified pinned ARUW model from model-manifest.json")
    providers = ["CPUExecutionProvider"]
    if args.provider == "coreml":
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CoreML unavailable; select --provider cpu explicitly")
        providers.insert(0, ("CoreMLExecutionProvider", {
            "ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "1",
            "ModelCacheDirectory": str((args.output / "coreml-cache" / sha).resolve())}))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), options, providers=providers)
    info = session.get_inputs()[0]
    if info.shape != [1, 3, 384, 640] or info.type != "tensor(float)":
        raise ValueError("ARUW input contract changed")
    cap = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Cannot read video or determine frame rate")
    count = 0
    timings = []
    with (args.output / "tensors.txt").open("w") as output:
        while not args.max_frames or count < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            tensor = preprocess(frame, "aruw-2026", 384, 640)
            started = time.perf_counter()
            raw = session.run(None, {info.name: tensor})[0]
            timings.append((time.perf_counter() - started) * 1000)
            if raw.shape != (1, 300, 14) or not np.isfinite(raw).all():
                raise ValueError("Invalid output tensor")
            # C++ owns validation, class mapping, corner ordering and tracking.
            # Drop only rows under threshold to keep the intermediate file small.
            rows = raw[0][raw[0, :, 4] >= args.threshold]
            h, w = frame.shape[:2]
            output.write(f"{count} {count/fps:.9f} {w} {h} {len(rows)}")
            if len(rows):
                output.write(" " + " ".join(f"{float(v):.9g}" for v in rows.flat))
            output.write("\n")
            count += 1
            if count % 300 == 0:
                print(f"Inferred {count} frames", flush=True)
    cap.release()
    if count == 0:
        raise ValueError("No frames decoded")
    metadata = {"video": str(args.video.resolve()), "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
                "model_sha256": sha, "fps": fps, "frames": count, "threshold": args.threshold,
                "provider": args.provider, "registered_providers": session.get_providers(),
                "onnxruntime": ort.__version__, "median_inference_ms": float(np.median(timings[1:] or timings)),
                "p95_inference_ms": float(np.percentile(timings[1:] or timings, 95)),
                "scope": "Unlabeled prerecorded imagery; image-space tracking only; no 3D commands"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


def render(args):
    meta = json.loads((args.output / "metadata.json").read_text())
    video = Path(meta["video"])
    if hashlib.sha256(video.read_bytes()).hexdigest() != meta["video_sha256"]:
        raise ValueError("Replay input changed since inference")
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    tracks = [json.loads(line) for line in (args.output / "tracks.jsonl").read_text().splitlines()]
    if len(tracks) != meta["frames"]:
        raise ValueError("Tracking output frame count differs from inference")
    writer = None
    trails = defaultdict(lambda: deque(maxlen=25))
    seen_ids = set()
    frames_detected = frames_selected = switches = coasts = 0
    last_selected = -1
    representative = None
    representative_score = -1
    for index, result in enumerate(tracks):
        ok, frame = cap.read()
        if not ok or result["frame"] != index:
            raise ValueError("Video/tracking alignment failed")
        # Fit camera view above a separate diagnostics panel; preserve image aspect ratio.
        scale = min(1.0, 1280 / frame.shape[1])
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        h, w = frame.shape[:2]
        canvas = np.zeros((h + 150, w, 3), np.uint8)
        canvas[:h] = frame
        center = (w // 2, h // 2)
        cv2.drawMarker(canvas, center, (230, 230, 230), cv2.MARKER_CROSS, 24, 1)
        selected = result["selected_id"]
        frames_detected += result["detections"] > 0
        frames_selected += selected >= 0
        if selected >= 0 and last_selected >= 0 and selected != last_selected:
            switches += 1
        if selected >= 0:
            last_selected = selected
        selection_text = "No confirmed selected plate"
        visible = []
        for track in result["tracks"]:
            seen_ids.add(track["id"])
            # Lost tracks beyond publication coast limit remain internal only.
            if track["lost_frames"] > 1:
                continue
            visible.append(track)
            coast = track["lost_frames"] > 0
            coasts += coast
            color = (0, 190, 255) if coast else ((85, 100, 255) if track["class_id"] == 1 else (255, 170, 65))
            x, y = (int(v * scale) for v in track["center"])
            predicted = tuple(int(v * scale) for v in track["prediction"])
            if not coast:
                corners = np.rint(np.asarray(track["corners"]) * scale).astype(np.int32)
                cv2.polylines(canvas, [corners], True, color, 2, cv2.LINE_AA)
                for k, point in enumerate(corners):
                    cv2.circle(canvas, tuple(point), 3, color, -1)
                    cv2.putText(canvas, str(k), tuple(point + (4, -4)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)
            trails[track["id"]].append((x, y))
            trail = np.asarray(trails[track["id"]], np.int32)
            if len(trail) > 1:
                cv2.polylines(canvas, [trail], False, color, 1, cv2.LINE_AA)
            cv2.drawMarker(canvas, predicted, color, cv2.MARKER_DIAMOND, 12, 1)
            label = f"ID {track['id']} | {'R' if track['class_id'] else 'B'}{track['number']} | {track['confidence']:.2f}"
            if coast:
                label += " COAST"
            elif track["state"] == 0:
                label += " NEW"
            cv2.putText(canvas, label, (max(2, x - 60), max(20, y - 18)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
            if track["id"] == selected:
                cv2.arrowedLine(canvas, center, (x, y), (70, 240, 120), 2, cv2.LINE_AA, tipLength=.08)
                dx, dy = track["center"][0] - w / scale / 2, track["center"][1] - h / scale / 2
                selection_text = f"Selected {selected} | error: dx={dx:+.1f}px  dy={dy:+.1f}px | normalized ({dx/(w/scale/2):+.3f}, {dy/(h/scale/2):+.3f})"
        lines = [
            f"ARUW 2026 + production C++ decoder / tracker | {index/meta['fps']:.2f}s | frame {index} | replay {args.playback_speed:g}x",
            selection_text,
            f"Visible tracks {len(visible)} | selected-ID changes {switches} | inference median {meta['median_inference_ms']:.1f}ms",
            "Arrow = image centering error. Diamond = tracker prediction. Yellow = coast.",
            "REPLAY ONLY | no calibration/attitude: no 3D, motor commands or fire | both colors shown",
        ]
        for line_no, line in enumerate(lines):
            cv2.putText(canvas, line, (12, h + 25 + 26 * line_no), cv2.FONT_HERSHEY_SIMPLEX,
                        .48 if w >= 900 else .34, (225, 230, 235), 1, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(str(args.output / "overlay-mjpeg.avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                     meta["fps"] * args.playback_speed, (w, h + 150))
            if not writer.isOpened():
                raise RuntimeError("Cannot create video writer")
        writer.write(canvas)
        score = len(visible) + (selected >= 0)
        if index > min(30, len(tracks) // 4) and score > representative_score:
            representative, representative_score = canvas.copy(), score
    cap.release()
    writer.release()
    if representative is not None:
        cv2.imwrite(str(args.output / "preview.jpg"), representative)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(args.output / "overlay-mjpeg.avi"),
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(args.output / "overlay.mp4")], check=True)
        (args.output / "overlay-mjpeg.avi").unlink()
    metrics = {"frames": len(tracks), "frames_with_detections": frames_detected,
               "frames_with_selection": frames_selected, "distinct_plate_track_ids": len(seen_ids),
               "selected_id_changes": switches, "coasting_track_frames": coasts,
               "scope": "No ground truth: ID changes include legitimate plate switches and cannot be called identity errors"}
    (args.output / "tracking-summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["infer", "render"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--provider", choices=["cpu", "coreml"], default="coreml")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--playback-speed", type=float, default=1.0,
                        help="Video playback speed only; tracking always uses original frame timing")
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.max_frames < 0:
        parser.error("Threshold must be in (0,1), max-frames nonnegative")
    if not np.isfinite(args.playback_speed) or not 0.1 <= args.playback_speed <= 2:
        parser.error("Playback speed must be between 0.1 and 2")
    if args.stage == "infer":
        if args.video is None or args.model is None:
            parser.error("infer requires --video and --model")
        infer(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
