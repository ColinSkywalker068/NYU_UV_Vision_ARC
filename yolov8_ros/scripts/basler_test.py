#!/usr/bin/env python3
from time import perf_counter
from pypylon import pylon, genicam

DURATION_SECONDS = 5.0  # measurement window


def set_enum(node_map: genicam.INodeMap, name: str, value: str) -> None:
    node = node_map.GetNode(name)
    if node and genicam.IsWritable(node):
        pylon.CEnumParameter(node_map, name).SetValue(value)


def set_float_min(node_map: genicam.INodeMap, name: str) -> None:
    node = node_map.GetNode(name)
    if node and genicam.IsWritable(node):
        param = pylon.CFloatParameter(node_map, name)
        param.SetValue(param.GetMin())


def set_float_max(node_map: genicam.INodeMap, name: str) -> None:
    node = node_map.GetNode(name)
    if node and genicam.IsWritable(node):
        param = pylon.CFloatParameter(node_map, name)
        param.SetValue(param.GetMax())


def configure_for_max_fps(camera: pylon.InstantCamera) -> None:
    node_map = camera.GetNodeMap()

    set_enum(node_map, "ExposureAuto", "Off")
    set_enum(node_map, "GainAuto", "Off")
    set_float_min(node_map, "ExposureTime")

    pixel_node = node_map.GetNode("PixelFormat")
    if pixel_node and genicam.IsWritable(pixel_node):
        symbols = pylon.CEnumParameter(node_map, "PixelFormat").GetSymbolics()
        if "Mono8" in symbols:
            set_enum(node_map, "PixelFormat", "Mono8")

    set_enum(node_map, "AcquisitionFrameRateEnable", "True")
    set_float_max(node_map, "AcquisitionFrameRate")

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
        avg_fps = frames / elapsed if elapsed else 0.0
        print(f"Frames: {frames}, Elapsed: {elapsed:.3f}s, Avg FPS: {avg_fps:.2f}")

    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()


if __name__ == "__main__":
    main()
