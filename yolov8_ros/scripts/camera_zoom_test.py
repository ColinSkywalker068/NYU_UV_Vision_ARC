#!/usr/bin/env python3
from pypylon import pylon

try:
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    print("=" * 50)
    print("CAMERA DIAGNOSTICS")
    print("=" * 50)
    print(f"Camera Model: {camera.GetDeviceInfo().GetModelName()}")
    print(
        f"\nSensor Maximum: {camera.WidthMax.GetValue()}x{camera.HeightMax.GetValue()}"
    )
    print(f"Current Resolution: {camera.Width.GetValue()}x{camera.Height.GetValue()}")
    print(
        f"Current Offset: X={camera.OffsetX.GetValue()}, Y={camera.OffsetY.GetValue()}"
    )

    # Check if cropped
    if camera.Width.GetValue() < camera.WidthMax.GetValue():
        print("\n⚠ WARNING: Camera is CROPPED (zoomed in)!")
        print(
            f"   Using {camera.Width.GetValue()}/{camera.WidthMax.GetValue()} of sensor width"
        )

    # Check binning support
    try:
        bin_h = camera.BinningHorizontal.GetValue()
        bin_v = camera.BinningVertical.GetValue()
        print(f"\nBinning: {bin_h}x{bin_v} (Supported)")
    except:
        print("\nBinning: Not available")

    # Check decimation support
    try:
        dec_h = camera.DecimationHorizontal.GetValue()
        dec_v = camera.DecimationVertical.GetValue()
        print(f"Decimation: {dec_h}x{dec_v} (Supported)")
    except:
        print("Decimation: Not available")

    print("=" * 50)

    camera.Close()

except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure:")
    print("1. Camera is connected")
    print("2. No other program is using it")
    print("3. You have permissions (try: sudo usermod -a -G dialout $USER)")
