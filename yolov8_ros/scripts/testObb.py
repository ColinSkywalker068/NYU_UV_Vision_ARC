Here's the full diagnosis before I touch any code — this is deep, so read through it all first.

---

## The Real Bottlenecks, Ranked by Impact

### 1. ExposureAuto is almost certainly still ON — this is the primary FPS killer

Your `open_default_camera` never calls `ExposureAuto.SetValue("Off")`. When ExposureAuto is in `Continuous` mode (the Basler default from factory), the camera adjusts exposure dynamically based on scene brightness. When an object appears, the sensor gets darker signal → camera increases exposure time → **FPS hard-ceiling drops immediately**. This is exactly the 100→21fps pattern you described.

If your exposure time is 20ms you will never get an acquisition rate above 50Hz. At 33ms exposure you're capped at 30fps. At 50ms you're at 20fps. The camera enforces this at the hardware level — no software trick bypasses it.

### 2. GainAuto is also likely ON — compounds the exposure problem

Basler defaults ship with both `ExposureAuto` and `GainAuto` set to `Continuous`. When an object appears, both auto functions recalculate simultaneously. GainAuto drives ExposureAuto's decisions — they fight each other on every frame, inflating exposure even further. Must be disabled together.

### 3. AOI is running at full sensor resolution — you're transferring far more pixels than YOLO needs

Your code never sets `Width`/`Height` on the camera. If your sensor is e.g. 1920×1200 or 2448×2048, the camera is transferring the full frame over USB every frame. Use a smaller AOI. Decreasing the AOI means that the camera has less data to transmit. Critically: it is typically the ROI height that affects the frame rate, not the width, because scanline timing drives the sensor's internal clock. Halving the height can nearly double achievable FPS at the hardware level — the sensor completes each frame faster.

### 4. USB Stream Grabber parameters are at conservative defaults

`MaxTransferSize` specifies the maximum USB data transfer size in bytes. Increase the value to lower the CPU load. `NumMaxQueuedUrbs` specifies the maximum number of USB request blocks to be enqueued simultaneously. Increasing this value may improve stability and reduce jitter. On Jetson's USB host controller these defaults are tuned for compatibility, not throughput.

### 5. `ResultingFrameRate` is never read — you have no visibility into the camera's own ceiling

Basler cameras expose a read-only `ResultingFrameRate` register that reports the maximum FPS achievable given the current combination of exposure time, AOI, pixel format, and bandwidth. Your code never reads this, so you have no idea whether the camera is constrained at 21fps or 100fps before a single Python frame is grabbed.

### 6. `MaxNumBuffer` is at default (10) — grab thread can starve on Jetson under memory pressure

The default internal buffer count means the camera has only 10 slots to queue frames into. If the grab thread is delayed (CUDA memory pressure, OS scheduling) and the camera fills all 10 buffers, it starts dropping frames at the transport layer — silently, with `GrabStrategy_LatestImageOnly` you'd never know.

---

## What's NOT the problem

- `GrabStrategy_LatestImageOnly` — correct
- The threaded grab architecture — correct  
- BGR8 pixel format — correct, already set
- `np.copyto` pre-allocated buffer — already fixed

---

## Implementation

Now applying all of these:Here's the full summary of what was added and the env vars to tune with:

**`ExposureAuto=Off` + `GainAuto=Off`** — most impactful single change. These were almost certainly the cause of the 100→21 drop when an object appears. The camera was reading a darker scene → increasing exposure → hard FPS ceiling collapses.

**`BASLER_EXPOSURE_US=4000`** (default 4ms → 250fps ceiling). You can tune this: brighter light = lower value = higher ceiling. If your scene is dim and you need longer exposure for detection quality, find the minimum that still gives clean detections. `BASLER_EXPOSURE_US=8000` gives a 125fps ceiling, `BASLER_EXPOSURE_US=16000` gives 62fps.

**`BASLER_AOI_W` / `BASLER_AOI_H`** — set these to e.g. `BASLER_AOI_H=720` to cut the height your sensor transfers. Since height drives scanline timing, this can meaningfully raise the ResultingFrameRate the camera reports. Start with `BASLER_AOI_H=720` and see what `ResultingFrameRate` logs.

**`ResultingFrameRate`** — now printed on startup. This is the ground truth. If it prints `21.0`, the bottleneck is still in the camera settings (exposure/AOI), not your code.

**Stream grabber** (`MaxTransferSize=65536`, `NumMaxQueuedUrbs=32`, `MaxNumBuffer=32`) — reduces USB CPU overhead and gives the grab thread more buffer slots to absorb Jetson scheduling hiccups.