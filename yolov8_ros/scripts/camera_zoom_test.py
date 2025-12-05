from pypylon import pylon

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

print(f"Sensor Max: {camera.WidthMax.GetValue()}x{camera.HeightMax.GetValue()}")
print(f"Current: {camera.Width.GetValue()}x{camera.Height.GetValue()}")
print(f"Offset: {camera.OffsetX.GetValue()}, {camera.OffsetY.GetValue()}")

# Try binning
try:
    print(f"Binning available: H={camera.BinningHorizontal.GetValue()}, V={camera.BinningVertical.GetValue()}")
except:
    print("Binning not available")

camera.Close()