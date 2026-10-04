#!/usr/bin/env python3
"""Bounded capture of existing CV logs and preview stats; no serial/ROS control.

Run on the Jetson. Optional --typec-repo also records J-Link RTT without resetting.
"""
import argparse
from datetime import datetime, timezone
import json
import os
import math
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import threading
import urllib.request
import uuid


def video_command(executable, port, seconds, max_mib, output):
    # Stream-copy the already-rendered JPEGs. No second camera client, inference,
    # encoder, or controller access. Wall time preserves irregular arrival spacing.
    return [executable, "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-use_wallclock_as_timestamps", "1", "-analyzeduration", "1000000",
            "-probesize", "1048576", "-f", "mjpeg", "-i", "pipe:0",
            "-map", "0:v:0", "-c:v", "copy", "-t", str(seconds),
            "-fs", str(max_mib * 1024 * 1024), "-f", "matroska", str(output)]


def feed_video(process, port, output, stop, state):
    """Jetson's NVIDIA FFmpeg has no HTTP protocol; demux multipart in Python."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{port}/stream.mjpg", timeout=3) as stream, output.open("w") as index:
            while not stop.is_set():
                line = stream.readline(8192)
                if not line:
                    raise RuntimeError("Preview stream ended")
                if not line.startswith(b"--"):
                    continue
                length = None
                for _ in range(32):
                    header = stream.readline(8192)
                    if header in (b"\r\n", b"\n"):
                        break
                    if header.lower().startswith(b"content-length:"):
                        length = int(header.split(b":", 1)[1])
                if length is None or not 0 < length <= 10 * 1024 * 1024:
                    raise RuntimeError("Invalid preview JPEG length")
                jpeg = stream.read(length)
                if len(jpeg) != length or not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
                    raise RuntimeError("Incomplete preview JPEG")
                stamp = {"frame": state["frames"], "utc_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}
                process.stdin.write(jpeg)
                process.stdin.flush()
                index.write(json.dumps(stamp) + "\n")
                index.flush()
                state["frames"] += 1
    except Exception as exc:
        if not stop.is_set() and process.poll() is None:
            state["error"] = str(exc)
    finally:
        try:
            process.stdin.close()
        except (OSError, ValueError):
            pass


def stop_child(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--max-mib", type=int, default=128)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--preview-port", type=int, default=8080)
    ap.add_argument("--typec-repo", type=Path, help="Also record Type C RTT through J-Link/OpenOCD")
    ap.add_argument("--record-video", action="store_true", help="Save the existing overlay stream as overlay.mkv (requires ffmpeg)")
    ap.add_argument("--video-max-mib", type=int, default=1024, help="Separate video size cap; logs have their own --max-mib cap")
    args = ap.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or min(args.max_mib, args.video_max_mib) < 1 or not 1 <= args.preview_port <= 65535:
        ap.error("positive duration/size and valid port required")
    if args.typec_repo and not (args.typec_repo / "scripts/logger/typec_diagnostics.py").is_file():
        ap.error("Type C diagnostics script is missing from --typec-repo")
    ffmpeg = shutil.which("ffmpeg") if args.record_video else None
    if args.record_video and not ffmpeg:
        ap.error("--record-video needs ffmpeg; install it before starting the session")
    repo = args.repo.resolve()
    out = repo / "vision_logs" / "sessions" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    def git(*command):
        result = subprocess.run(["git", "-C", str(repo), *command], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    def clock():
        return {"utc_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}
    manifest = {"start": clock(), "revision": git("rev-parse", "HEAD"),
                "working_tree": git("status", "--short"), "duration_requested_s": args.seconds,
                "clock_note": "Host clocks are not synchronized MCU source timestamps.",
                "scope": "New bytes appended to existing bridge logs plus HTTP stats; no configuration copied."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Key by inode so rotations do not duplicate previously recorded bytes.
    offsets = {}
    for path in (repo / "vision_logs").glob("vision_serial_runtime_*.log*"):
        try:
            stat = path.stat()
            offsets[(stat.st_dev, stat.st_ino)] = stat.st_size
        except FileNotFoundError:
            continue
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.monotonic()
    limit = args.max_mib * 1024 * 1024
    copied = 0
    next_preview = start
    reason = "duration_complete"
    typec_process = None
    typec_console = None
    video_process = None
    video_console = None
    video_thread = None
    video_stop = threading.Event()
    video_state = {"frames": 0}
    print(f"Recording existing logs only: {out}", flush=True)
    try:
        if args.record_video:
            video_console = (out / "video-console.log").open("w")
            manifest["video_start"] = clock()
            manifest["video_note"] = "Rendered image overlays only; web-page side panels are in preview.jsonl. Video times reflect host arrival, not exposure/MCU synchronization. MJPEG in Matroska; play with VLC."
            video_process = subprocess.Popen(
                video_command(ffmpeg, args.preview_port, args.seconds, args.video_max_mib, out / "overlay.mkv"),
                stdin=subprocess.PIPE, stdout=video_console, stderr=subprocess.STDOUT, start_new_session=True)
            video_thread = threading.Thread(target=feed_video, args=(video_process, args.preview_port,
                out / "video-frames.jsonl", video_stop, video_state), daemon=True)
            video_thread.start()
            print("Recording overlay video: overlay.mkv (stream copy, no camera/config changes)", flush=True)
        if args.typec_repo:
            typec_console = (out / "typec-console.log").open("w")
            typec_process = subprocess.Popen(
                [sys.executable, str(args.typec_repo.resolve() / "scripts/logger/typec_diagnostics.py"),
                 "--openocd", "--seconds", str(args.seconds), "--out", str(out / "typec")],
                stdout=typec_console, stderr=subprocess.STDOUT, start_new_session=True)
            print("Type C RTT capture requested; status in typec-console.log. No reset/flash or control packets.", flush=True)
        with (out / "bridge.log").open("wb") as bridge, (out / "preview.jsonl").open("w") as preview:
            while time.monotonic() - start < args.seconds and copied < limit:
                if typec_process and typec_process.poll() not in (None, 0):
                    raise RuntimeError("Type C capture failed; inspect typec-console.log and typec/*/openocd.log")
                if video_process and video_process.poll() not in (None, 0):
                    raise RuntimeError("Overlay video capture failed; inspect video-console.log")
                if "error" in video_state:
                    raise RuntimeError("Overlay input failed: " + video_state["error"])
                if video_process and video_process.poll() == 0 and "video_finished" not in manifest:
                    manifest["video_finished"] = clock()
                    manifest["video_ended_early"] = True
                    print("Video stopped at its size/duration limit or stream end; log capture continues.", flush=True)
                for path in sorted((repo / "vision_logs").glob("vision_serial_runtime_*.log*")):
                    try:
                        with path.open("rb") as source:
                            stat = os.fstat(source.fileno())
                            key = (stat.st_dev, stat.st_ino)
                            offset = offsets.get(key, 0)
                            if offset > stat.st_size:
                                offset = 0
                            source.seek(offset)
                            data = source.read(min(1024 * 1024, limit - copied))
                            bridge.write(data)
                            copied += len(data)
                            offsets[key] = source.tell()
                    except FileNotFoundError:
                        continue  # Rotation raced the directory listing.
                bridge.flush()
                if time.monotonic() >= next_preview:
                    row = clock()
                    try:
                        with opener.open(f"http://127.0.0.1:{args.preview_port}/stats.json", timeout=.5) as response:
                            row["stats"] = json.loads(response.read(65536))
                    except (OSError, ValueError) as exc:
                        row["error"] = str(exc)
                    row["response_monotonic_ns"] = time.monotonic_ns()
                    preview.write(json.dumps(row) + "\n")
                    preview.flush()
                    next_preview = time.monotonic() + 1
                time.sleep(.2)
        if copied >= limit:
            reason = "bridge_log_size_limit"
    except KeyboardInterrupt:
        reason = "stopped_by_user"
    except Exception as exc:
        reason = "capture_error"
        manifest["error"] = str(exc)
        raise
    finally:
        if typec_process:
            stop_child(typec_process)
            manifest["typec_exit_code"] = typec_process.returncode
        if typec_console:
            typec_console.close()
        if video_process:
            video_stop.set()
            stop_child(video_process)
            video_thread.join(timeout=4)
            manifest["video_input"] = video_state
            manifest["video_exit_code"] = video_process.returncode
            video = out / "overlay.mkv"
            manifest["video_bytes"] = video.stat().st_size if video.exists() else 0
            manifest["video_end"] = clock()
        if video_console:
            video_console.close()
        manifest.update(end=clock(), bridge_bytes=copied, stop_reason=reason)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {copied} bridge-log bytes; {reason}. Share {out}")
    return 0 if copied and (typec_process is None or typec_process.returncode == 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
