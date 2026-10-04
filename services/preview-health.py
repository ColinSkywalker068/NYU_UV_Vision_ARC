#!/usr/bin/env python3
"""Check HTTP preview and advancing frames; never start/restart CV or change TX."""
import argparse
import json
import os
import shlex
import time
import urllib.request


def preview_port():
    port = 8080
    for arg in shlex.split(os.environ.get("CV_STM_EXTRA_LAUNCH_ARGS", "")):
        if arg.startswith("preview_http_port:="):
            port = int(arg.split(":=", 1)[1])
    if not 1 <= port <= 65535:
        raise ValueError("invalid preview_http_port")
    return port


def check(base, timeout):
    deadline = time.monotonic() + timeout
    previous = None
    error = "no advancing frames"
    # Ignore proxy settings for the local camera endpoint.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open(base + "/stats.json", timeout=min(2, max(.1, deadline - time.monotonic()))) as response:
                stats = json.load(response)
            count = int(stats["frame_count"])
            if previous is not None and count > previous:
                with opener.open(base + "/snapshot.jpg", timeout=2) as response:
                    if response.read(2) != b"\xff\xd8":
                        raise ValueError("snapshot is not JPEG")
                print(f"[preview] LIVE: frame {previous} -> {count}; Jetson URL: {base}/")
                return 0
            previous = count
            error = f"frame count is not advancing (last={count})"
        except (OSError, ValueError, KeyError) as exc:
            error = str(exc)
        time.sleep(.25)
    print(f"[preview] NOT VERIFIED: {error}. CV was not changed.")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timeout", type=float, default=10)
    args = ap.parse_args()
    mode = os.environ.get("CV_STM_SHOW", "false").lower()
    if mode != "web":
        print(f"[preview] Saved mode is {mode!r}; HTTP preview is not requested. Set CV_STM_SHOW=web before your next restart.")
        return 0
    port = preview_port()
    print("[preview] Host browser: http://127.0.0.1:18080/ (SSH forwarding required)")
    print(f"[preview] On your host device: ssh -N -L 127.0.0.1:18080:127.0.0.1:{port} ultraviolet@192.168.55.1")
    print("[preview] For Wi-Fi, replace the USB IP with this Jetson's trusted hostname/IP.")
    return check(f"http://127.0.0.1:{port}", args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
