#!/usr/bin/env python3
"""Summarize recorded tracking/attitude alignment; never opens ROS or a device."""
import argparse
import glob
import json
import math
import shlex
from pathlib import Path


def rows(paths):
    for path in paths:
        with path.open(errors="replace") as stream:
            for line in stream:
                try:
                    yield dict(token.split("=", 1) for token in shlex.split(line) if "=" in token)
                except ValueError:
                    continue  # A power loss can leave one partial trailing record.


def summarize(paths):
    targets = {}
    for row in rows(paths):
        if row.get("kind") == "tracking_target":
            targets[row.get("frame_ns")] = (row.get("track_id"), row.get("coasting"))
    result = {"alignment_samples": 0, "rejected_commands": 0, "stale_attitude_samples": 0,
              "tx_enabled_samples": 0, "serial_write_successes": 0, "serial_write_failures": 0,
              "coasting_samples": 0, "matched_target_samples": 0, "by_target": {}}
    for row in rows(paths):
        if row.get("kind") == "serial_tx":
            key = "serial_write_successes" if row.get("ok") == "1" else "serial_write_failures"
            result[key] += 1
        if row.get("kind") != "tracking_alignment":
            continue
        result["alignment_samples"] += 1
        result["tx_enabled_samples"] += row.get("tx_enabled") == "1"
        result["stale_attitude_samples"] += row.get("attitude_fresh") != "1"
        if row.get("accepted") != "1":
            result["rejected_commands"] += 1
            continue
        identity, coast = targets.get(row.get("frame_ns"), ("unmatched", "0"))
        result["matched_target_samples"] += identity != "unmatched"
        result["coasting_samples"] += coast == "1"
        if row.get("attitude_fresh") != "1":
            continue
        try:
            values = {key: float(row[key]) for key in
                      ("yaw_error", "pitch_error", "frame_age_s", "measurement_age_s", "attitude_age_s")}
        except (KeyError, ValueError):
            continue
        if not all(math.isfinite(value) for value in values.values()):
            continue
        stat = result["by_target"].setdefault(identity, {"samples": 0, "yaw_error_sq": 0,
            "pitch_error_sq": 0, "max_abs_yaw_error_deg": 0, "max_abs_pitch_error_deg": 0,
            "max_frame_age_ms": 0, "max_measurement_age_ms": 0, "max_attitude_age_ms": 0})
        stat["samples"] += 1
        for axis in ("yaw", "pitch"):
            error = math.degrees(values[axis + "_error"])
            stat[axis + "_error_sq"] += error * error
            key = "max_abs_" + axis + "_error_deg"
            stat[key] = max(stat[key], abs(error))
        for kind in ("frame", "measurement", "attitude"):
            key = "max_" + kind + "_age_ms"
            stat[key] = max(stat[key], 1000 * values[kind + "_age_s"])
    for stat in result["by_target"].values():
        for axis in ("yaw", "pitch"):
            stat[axis + "_error_rms_deg"] = math.sqrt(stat.pop(axis + "_error_sq") / stat["samples"])
    result["interpretation"] = "Host-time diagnostics only. Serial write success is not MCU acceptance or motor response. Inspect time series to distinguish command motion from measured motion."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Log files or quoted glob patterns; include rotated .log.* files")
    args = parser.parse_args()
    paths = sorted({Path(path) for pattern in args.logs for path in glob.glob(pattern)})
    if not paths:
        parser.error("no log files matched")
    print(json.dumps(summarize(paths), indent=2))
