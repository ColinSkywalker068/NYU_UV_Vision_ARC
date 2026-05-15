#!/usr/bin/env python3
"""Check whether a Basler camera can be opened and can stream via pypylon."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


PROCESS_SEARCH_TERMS = (
    "armor_detector_node",
    "camera_node",
    "pylonviewer",
    "pylonsrc",
    "pypylon",
    "gst-launch",
    "ros2 launch",
    "stm.launch.py",
)


class CheckFailure(RuntimeError):
    """Expected health-check failure with a user-facing message."""


def print_status(level: str, label: str, detail: str = "") -> None:
    if detail:
        print(f"[{level}] {label}: {detail}")
    else:
        print(f"[{level}] {label}")


def run_command(command: list[str], timeout_seconds: float = 3.0) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return 127, "", f"{command[0]} not found"
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return 124, stdout, stderr or "command timed out"
    return completed.returncode, completed.stdout, completed.stderr


def short_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if not message:
        message = repr(exc)
    return f"{type(exc).__name__}: {message}"


def print_exception_hints(exc: BaseException) -> None:
    text = str(exc).lower()
    hints: list[str] = []
    if any(
        needle in text
        for needle in (
            "already",
            "access denied",
            "busy",
            "exclusively",
            "in use",
            "controlled by another application",
        )
    ):
        hints.append(
            "The camera looks grabbed. Stop ROS/CV containers, Pylon Viewer, gst-launch, "
            "or any other pypylon process, then replug the camera."
        )
    if "no device" in text or "not found" in text or "could not find" in text:
        hints.append(
            "Pylon cannot see a usable device even if lsusb can. Restart the container "
            "after replugging and confirm USB/GigE device permissions."
        )
    if "timeout" in text or "transfer" in text or "usb" in text:
        hints.append(
            "This smells like transport trouble: USB3 cable/port/power, hub stability, "
            "or a stale container device view."
        )
    if "not writable" in text or "accessexception" in text or "genicam" in text:
        hints.append(
            "A GenICam setting may be rejected by this camera state/model. Re-run with "
            "--config-mode none to test open/grab only."
        )
    for hint in hints:
        print(f"      hint: {hint}")


def print_environment_snapshot() -> None:
    print("\nEnvironment")
    pylon_root = os.environ.get("PYLON_ROOT", "")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    gst_plugin_path = os.environ.get("GST_PLUGIN_PATH", "")
    print_status("INFO", "PYLON_ROOT", pylon_root or "(unset)")
    print_status(
        "INFO",
        "/opt/pylon",
        "present" if Path("/opt/pylon").exists() else "missing",
    )
    print_status(
        "INFO",
        "LD_LIBRARY_PATH contains pylon",
        "yes" if "pylon" in ld_library_path.lower() else "no",
    )
    print_status(
        "INFO",
        "GST_PLUGIN_PATH contains pylon",
        "yes" if "pylon" in gst_plugin_path.lower() else "no",
    )


def check_usb_listing() -> None:
    print("\nUSB visibility")
    if shutil.which("lsusb") is None:
        print_status("WARN", "lsusb", "not installed; skipping USB enumeration")
        return

    code, stdout, stderr = run_command(["lsusb"])
    if code != 0:
        print_status("WARN", "lsusb failed", stderr.strip() or f"exit code {code}")
        return

    matches = [
        line
        for line in stdout.splitlines()
        if "basler" in line.lower() or "2676:" in line.lower()
    ]
    if matches:
        print_status("PASS", "Basler-like USB device visible")
        for line in matches:
            print(f"      {line}")
    else:
        print_status(
            "WARN",
            "No obvious Basler line in lsusb",
            "Pylon may still see GigE cameras or devices with generic USB labels.",
        )


def check_gstreamer_pylonsrc() -> None:
    print("\nGStreamer Basler plugin")
    if shutil.which("gst-inspect-1.0") is None:
        print_status("WARN", "gst-inspect-1.0", "not installed; skipping pylonsrc check")
        return
    code, stdout, stderr = run_command(["gst-inspect-1.0", "pylonsrc"], 5.0)
    if code == 0:
        first_line = stdout.splitlines()[0] if stdout.splitlines() else "available"
        print_status("PASS", "pylonsrc", first_line)
    else:
        print_status(
            "WARN",
            "pylonsrc unavailable",
            (stderr or stdout).strip() or f"exit code {code}",
        )


def check_possible_grabbers() -> None:
    print("\nPossible camera owners")
    command = ["ps", "-eo", "pid,comm,args"]
    code, stdout, stderr = run_command(command)
    if code != 0:
        code, stdout, stderr = run_command(["ps", "aux"])
    if code != 0:
        print_status("WARN", "process scan failed", stderr.strip() or f"exit code {code}")
        return

    current_pid = str(os.getpid())
    hits: list[str] = []
    for line in stdout.splitlines()[1:]:
        lower = line.lower()
        if line.strip().startswith(current_pid):
            continue
        if "check_basler.py" in lower:
            continue
        if any(term in lower for term in PROCESS_SEARCH_TERMS):
            hits.append(line.strip())

    if not hits:
        print_status("PASS", "No obvious ROS/Pylon grabber processes found")
        return

    print_status(
        "WARN",
        "Possible grabbers found",
        "stop these before testing if open fails as busy/access denied",
    )
    for line in hits[:12]:
        print(f"      {line}")
    if len(hits) > 12:
        print(f"      ... {len(hits) - 12} more")


def load_pypylon() -> Any:
    print("\npypylon")
    try:
        import pypylon  # type: ignore[import-not-found]
        from pypylon import pylon  # type: ignore[import-not-found]
    except Exception as exc:
        print_status("FAIL", "import pypylon", short_exception(exc))
        print("      install or run in an environment with: python3 -m pip install pypylon")
        raise CheckFailure("pypylon import failed") from exc

    version = getattr(pypylon, "__version__", "unknown")
    print_status("PASS", "import pypylon", f"version {version}")
    return pylon


def safe_device_value(device_info: Any, method_name: str) -> str:
    try:
        method = getattr(device_info, method_name)
    except AttributeError:
        return ""
    try:
        value = method()
    except Exception:
        return ""
    return str(value)


def describe_device(device_info: Any) -> str:
    fields = [
        ("model", safe_device_value(device_info, "GetModelName")),
        ("serial", safe_device_value(device_info, "GetSerialNumber")),
        ("class", safe_device_value(device_info, "GetDeviceClass")),
        ("vendor", safe_device_value(device_info, "GetVendorName")),
        ("friendly", safe_device_value(device_info, "GetFriendlyName")),
        ("full", safe_device_value(device_info, "GetFullName")),
        ("ip", safe_device_value(device_info, "GetIpAddress")),
    ]
    return ", ".join(f"{name}={value}" for name, value in fields if value) or "(no info)"


def select_device(pylon: Any, index: int, serial: str) -> Any:
    factory = pylon.TlFactory.GetInstance()
    devices = list(factory.EnumerateDevices())
    if not devices:
        print_status("FAIL", "Pylon enumeration", "no Basler devices found")
        raise CheckFailure("no Pylon devices found")

    print_status("PASS", "Pylon enumeration", f"{len(devices)} device(s)")
    for device_index, device_info in enumerate(devices):
        print(f"      [{device_index}] {describe_device(device_info)}")

    if serial:
        for device_info in devices:
            if safe_device_value(device_info, "GetSerialNumber") == serial:
                return device_info
        raise CheckFailure(f"serial {serial!r} was not found in Pylon enumeration")

    if index < 0 or index >= len(devices):
        raise CheckFailure(f"device index {index} out of range for {len(devices)} device(s)")
    return devices[index]


def node_readable_value(camera: Any, name: str) -> str:
    try:
        node = getattr(camera, name)
    except Exception:
        return "(missing)"
    try:
        if hasattr(node, "IsReadable") and not node.IsReadable():
            return "(not readable)"
    except Exception:
        pass
    try:
        return str(node.GetValue())
    except Exception as exc:
        return f"({short_exception(exc)})"


def print_camera_nodes(camera: Any) -> None:
    print("\nCamera nodes")
    for name in (
        "Width",
        "Height",
        "PixelFormat",
        "ExposureAuto",
        "ExposureTime",
        "Gain",
        "AcquisitionFrameRateEnable",
        "AcquisitionFrameRate",
    ):
        print_status("INFO", name, node_readable_value(camera, name))


def set_node(camera: Any, name: str, value: object) -> tuple[bool, str]:
    try:
        node = getattr(camera, name)
    except Exception:
        return False, "missing"
    try:
        if hasattr(node, "IsWritable") and not node.IsWritable():
            return False, "not writable"
    except Exception:
        pass
    try:
        node.SetValue(value)
    except Exception as exc:
        return False, short_exception(exc)
    return True, str(value)


def apply_camera_config(camera: Any, args: argparse.Namespace) -> bool:
    print("\nCamera config")
    if args.config_mode == "none":
        print_status("PASS", "config mode", "none; no GenICam writes attempted")
        return True

    if args.config_mode == "safe":
        fps = args.fps if args.fps is not None else 30
        writes: list[tuple[str, object]] = [
            ("AcquisitionFrameRateEnable", True),
            ("AcquisitionFrameRate", float(fps)),
        ]
    else:
        fps = args.fps if args.fps is not None else 90
        exposure_us = args.exposure_us if args.exposure_us is not None else 5000.0
        gain_db = args.gain_db if args.gain_db is not None else 5.0
        uncapped = not args.stm_capped_fps
        writes = [
            ("AcquisitionFrameRateEnable", not uncapped),
            ("ExposureAuto", "Off"),
            ("ExposureTime", float(exposure_us)),
            ("Gain", float(gain_db)),
        ]
        if not uncapped:
            writes.insert(1, ("AcquisitionFrameRate", float(fps)))

    all_ok = True
    for name, value in writes:
        ok, detail = set_node(camera, name, value)
        all_ok = all_ok and ok
        print_status("PASS" if ok else "WARN", f"set {name}", detail)
    if args.strict_config and not all_ok:
        raise CheckFailure("one or more requested GenICam writes failed")
    return all_ok


def save_frame(path: Path, frame_bgr: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:
        print_status("WARN", "save frame skipped", f"OpenCV import failed: {short_exception(exc)}")
        return
    if not cv2.imwrite(str(path), frame_bgr):
        print_status("WARN", "save frame failed", str(path))
        return
    print_status("PASS", "saved frame", str(path))


def grab_frames(pylon: Any, camera: Any, args: argparse.Namespace) -> int:
    print("\nFrame grab")
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    grabbed = 0
    saved = False
    start_time = time.monotonic()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    try:
        for frame_index in range(args.grab_count):
            result = camera.RetrieveResult(
                args.timeout_ms,
                pylon.TimeoutHandling_Return,
            )
            if result is None:
                print_status("WARN", f"frame {frame_index}", "RetrieveResult returned None")
                continue
            try:
                if not result.GrabSucceeded():
                    error_code = getattr(result, "GetErrorCode", lambda: "unknown")()
                    error_text = getattr(result, "GetErrorDescription", lambda: "unknown")()
                    print_status(
                        "WARN",
                        f"frame {frame_index}",
                        f"grab failed code={error_code} description={error_text}",
                    )
                    continue
                converted = converter.Convert(result)
                frame_bgr = converted.GetArray()
                grabbed += 1
                print_status(
                    "PASS",
                    f"frame {frame_index}",
                    f"{result.GetWidth()}x{result.GetHeight()} array={frame_bgr.shape}",
                )
                if args.save_frame and not saved:
                    save_frame(Path(args.save_frame), frame_bgr)
                    saved = True
            finally:
                result.Release()
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()

    elapsed = max(1e-6, time.monotonic() - start_time)
    if grabbed:
        print_status(
            "PASS",
            "grab summary",
            f"{grabbed}/{args.grab_count} frames in {elapsed:.2f}s",
        )
    else:
        print_status("FAIL", "grab summary", "no frames grabbed")
    return grabbed


def open_and_test_camera(pylon: Any, device_info: Any, args: argparse.Namespace) -> int:
    factory = pylon.TlFactory.GetInstance()
    camera = pylon.InstantCamera(factory.CreateDevice(device_info))
    try:
        print("\nOpen camera")
        camera.Open()
        print_status("PASS", "open", describe_device(camera.GetDeviceInfo()))
        print_camera_nodes(camera)
        apply_camera_config(camera, args)
        grabbed = grab_frames(pylon, camera, args)
        return 0 if grabbed > 0 else 1
    except Exception as exc:
        print_status("FAIL", "camera test", short_exception(exc))
        print_exception_hints(exc)
        return 1
    finally:
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception:
            pass
        try:
            if camera.IsOpen():
                camera.Close()
        except Exception:
            pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Basler/Pylon health check: enumerate, open, optionally write a few "
            "GenICam settings, grab frames, and close."
        )
    )
    parser.add_argument("--index", type=int, default=0, help="Pylon device index to test.")
    parser.add_argument("--serial", default="", help="Pylon serial number to test.")
    parser.add_argument("--grab-count", type=int, default=3, help="Number of frames to grab.")
    parser.add_argument("--timeout-ms", type=int, default=1000, help="Per-frame grab timeout.")
    parser.add_argument(
        "--config-mode",
        choices=("none", "safe", "stm"),
        default="none",
        help=(
            "none: no camera writes; safe: cap FPS gently; stm: mimic stm.launch.py "
            "Basler defaults."
        ),
    )
    parser.add_argument("--fps", type=int, default=None, help="FPS for safe/stm config modes.")
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=None,
        help="ExposureTime value for --config-mode stm.",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        default=None,
        help="Gain value for --config-mode stm.",
    )
    parser.add_argument(
        "--stm-capped-fps",
        action="store_true",
        help=(
            "With --config-mode stm, enable AcquisitionFrameRate instead of "
            "launch default uncapped FPS."
        ),
    )
    parser.add_argument(
        "--strict-config",
        action="store_true",
        help="Fail when any requested GenICam config write fails.",
    )
    parser.add_argument(
        "--save-frame",
        default="",
        help="Optional path for the first grabbed BGR frame. Requires OpenCV.",
    )
    parser.add_argument(
        "--skip-system-checks",
        action="store_true",
        help="Skip lsusb/process/GStreamer diagnostics and test pypylon only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.grab_count <= 0:
        raise SystemExit("--grab-count must be positive")
    if args.timeout_ms <= 0:
        raise SystemExit("--timeout-ms must be positive")

    print("Basler camera health check")
    if not args.skip_system_checks:
        print_environment_snapshot()
        check_usb_listing()
        check_possible_grabbers()
        check_gstreamer_pylonsrc()

    try:
        pylon = load_pypylon()
        device_info = select_device(pylon, args.index, args.serial)
    except CheckFailure as exc:
        print_status("FAIL", "health check", str(exc))
        return 1
    except Exception as exc:
        print_status("FAIL", "unexpected setup error", short_exception(exc))
        print_exception_hints(exc)
        return 1

    return open_and_test_camera(pylon, device_info, args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
