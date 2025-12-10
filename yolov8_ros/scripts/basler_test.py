#!/usr/bin/env python3
"""
Test Basler FPS by minimizing exposure and measuring average FPS.
Run: python test_basler_fps.py
"""

from time import perf_counter
from pypylon import pylon

DURATION_SECONDS = 5.0  # how long to measure


def configure_for_max_fps(camera: pylon.InstantCamera) -> None:
    node_map = camera.GetNodeMap()

    # Disable auto features to avoid overhead
    if "ExposureAuto" in node_map:
        camera.ExposureAuto.SetValue("Off")
    if "GainAuto" in node_map:
        camera.GainAuto.SetValue("Off")

    # Minimize exposure to reduce frame time
    if "ExposureTime" in node_map:
        try:
            camera.ExposureTime.SetValue(camera.ExposureTime.GetMin())
        except Exception:
            pass

    # Use a simple, fast pixel format
    if "PixelFormat" in node_map and "Mono8" in camera.PixelFormat.GetSymbolics():
        camera.PixelFormat.SetValue("Mono8")

    # Enable and set max frame rate if supported
    if "AcquisitionFrameRateEnable" in node_map:
        try:
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(camera.AcquisitionFrameRate.GetMax())
        except Exception:
            pass

    # Use latest-image-only to avoid backlog
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


def main() -> None:
    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise RuntimeError("No Basler camera found")

    camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))

    try:
        camera.Open()
        configure_for_max_fps(camera)

        start = perf_counter()
        frames = 0

        while (perf_counter() - start) < DURATION_SECONDS and camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab.GrabSucceeded():
                frames += 1
            grab.Release()

        elapsed = perf_counter() - start
        avg_fps = frames / elapsed if elapsed > 0 else 0.0
        print(f"Frames: {frames}, Elapsed: {elapsed:.3f}s, Avg FPS: {avg_fps:.2f}")

    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()


if __name__ == "__main__":
    main()
