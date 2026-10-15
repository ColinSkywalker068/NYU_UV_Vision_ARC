#!/usr/bin/env python3
"""Offline camera-motion consistency check. Never connects to ROS or writes config.

Fits a focal-length multiplier and attitude time shift under the explicit
assumption of a fixed target and fixed camera pivot. This is NOT intrinsic
calibration: target/chassis motion, distortion, translation, and encoder scale
can confound the result. Camera principal point and aspect are held fixed.
Requires NumPy. Supply multiple independent stationary-target windows.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def log_lines(path):
    with Path(path).open(errors="replace") as stream:
        yield from stream


def read_targets(path, start_ns):
    records = {}
    for line in log_lines(path):
        if "kind=tracking_target " not in line:
            continue
        row = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
        if any(row.get(key) != "1" for key in ("has_target", "camera_pose_valid")):
            continue
        if row.get("coasting") != "0":
            continue
        try:
            stamp = int(row["frame_ns"])
            values = [(stamp - start_ns) * 1e-9, float(row["center_x"]),
                      float(row["center_y"])]
            values += [float(row["camera_q" + key]) for key in "xyzw"]
            values += [int(row["track_id"])]
        except (KeyError, ValueError):
            continue
        if np.isfinite(values).all() and np.linalg.norm(values[3:7]) > 1e-6:
            records[stamp] = values
    return np.array([records[key] for key in sorted(records)], dtype=float)


def bearings(data, focal, center, multiplier=1.0, shift_s=0.0):
    """Positive shift pairs pixels at t with camera orientation at t+shift."""
    t = data[:, 0]
    q = data[:, 3:7].copy()
    # Prevent sign-equivalent quaternion samples interpolating through zero.
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1
    q = np.column_stack([np.interp(t + shift_s, t, q[:, i]) for i in range(4)])
    q /= np.linalg.norm(q, axis=1)[:, None]
    ray = np.column_stack(((data[:, 1] - center[0]) / (focal[0] * multiplier),
                           (data[:, 2] - center[1]) / (focal[1] * multiplier),
                           np.ones(len(data))))
    uv = np.cross(q[:, :3], ray)
    world = ray + 2 * (q[:, 3:4] * uv + np.cross(q[:, :3], uv))
    return np.column_stack((np.unwrap(np.arctan2(world[:, 0], world[:, 2])),
                            np.arctan2(-world[:, 1], np.hypot(world[:, 0], world[:, 2]))))


def fit_window(data, focal, center, axis, scales, shifts):
    if len(data) < 40 or data[-1, 0] - data[0, 0] < 1:
        raise ValueError("Need at least 40 observations over one second")
    if len(np.unique(data[:, 7])) != 1:
        raise ValueError("Window spans multiple target identities")
    # Use identical interior samples for all shifts; no interpolation beyond edges.
    margin = max(abs(float(x)) for x in shifts) + .02
    interior = (data[:, 0] > data[0, 0] + margin) & (data[:, 0] < data[-1, 0] - margin)
    block = np.floor((data[:, 0] - data[0, 0]) / .5).astype(int)
    train, test = interior & (block % 2 == 0), interior & (block % 2 == 1)
    if min(train.sum(), test.sum()) < 10:
        raise ValueError("Insufficient held-out samples")
    index = 0 if axis == "yaw" else 1

    def score(scale, shift):
        angle = bearings(data, focal, center, scale, shift)[:, index]
        anchor = np.mean(angle[train])
        return (float(np.sqrt(np.mean((angle[train] - anchor) ** 2))),
                float(np.sqrt(np.mean((angle[test] - anchor) ** 2))))

    def fit(candidates):
        best = min(((*score(s, dt), float(s), float(dt)) for s, dt in candidates),
                   key=lambda item: item[0])
        return {"multiplier": best[2], "attitude_shift_ms": 1000 * best[3],
                "train_rms_deg": float(np.degrees(best[0])),
                "held_out_rms_deg": float(np.degrees(best[1]))}

    result = {"axis": axis, "samples": len(data),
              "baseline": fit([(1, 0)]),
              "time_shift_only": fit((1, dt) for dt in shifts),
              "focal_only": fit((s, 0) for s in scales),
              "focal_and_time_shift": fit((s, dt) for s in scales for dt in shifts)}
    best = result["focal_and_time_shift"]
    best["at_search_boundary"] = (best["multiplier"] in (min(scales), max(scales)) or
                                   best["attitude_shift_ms"] in (1000 * min(shifts), 1000 * max(shifts)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--start-ns", type=int, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--window", action="append", required=True, help="start:end:yaw|pitch in seconds since start-ns")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.fx, args.fy) <= 0:
        parser.error("Focal lengths must be positive")
    data = read_targets(args.log, args.start_ns)
    if data.size == 0:
        parser.error("No usable target records")
    results = []
    for value in args.window:
        lo, hi, axis = value.split(":")
        if axis not in ("yaw", "pitch"):
            parser.error("axis must be yaw or pitch")
        selected = data[(data[:, 0] >= float(lo)) & (data[:, 0] <= float(hi))]
        result = fit_window(selected, (args.fx, args.fy), (args.cx, args.cy), axis,
                            np.round(np.linspace(.5, 5, 91), 5),
                            np.round(np.linspace(-.15, .15, 31), 5))
        result["window"] = value
        results.append(result)
    report = {"warning": "Conditional fixed-target consistency fit, NOT camera calibration. Do not automatically deploy. Positive shifts pair image t with orientation t+shift. Host timestamps do not establish MCU source synchronization.",
              "source_log": str(args.log.resolve()), "windows": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
