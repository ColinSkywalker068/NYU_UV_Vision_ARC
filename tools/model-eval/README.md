# Native Mac inference and production-tracker replay

These tools evaluate prerecorded video without opening a camera, serial port or
robot connection. Apple acceleration runs natively; Docker runs the same C++
decoder/tracker compiled for the ROS pipeline. Neither tool starts the live stack.

## Install and fetch

Use an Apple Silicon Python 3.12 environment (not Python inside Docker):

```bash
python3.12 -m venv /tmp/armor-eval-venv
source /tmp/armor-eval-venv/bin/activate
pip install -r tools/model-eval/requirements-native.txt
python tools/model-eval/fetch_candidates.py --output /tmp/armor-eval --candidate aruw-2026
python tools/model-eval/fetch_replays.py --output /tmp/armor-eval/clips
tools/run-local-tracking-tests.sh
```

The five clips total about 284 MB. Use repeated `--clip` arguments to fetch only
selected files. Downloads are pinned and verified against their repository blob
hash. The generated manifest records URLs, revisions and SHA256. Keep source
videos, weights, caches and derived videos outside git.

| Recording | Reason to test it |
| --- | --- |
| `tongji-demo.avi` | Real moving camera, low light, partial visibility and multiple plates |
| `robotpilots-armor.mp4` | Small general armor sample |
| `robotpilots-medium-spin.avi` | Plate appearance/disappearance during rotation |
| `robotpilots-near-height-spin.avi` | Rotation plus differing plate heights |
| `robotpilots-far-translation.avi` | Small, translating armor targets |

Sources: [Tongji demo](https://github.com/TongjiSuperPower/sp_vision_25/tree/bd9f5e798fa3c6dd3b483ae6627796afb41c608d/assets/demo),
[RobotPilots test videos](https://github.com/SZURPVision/26_NNDeployment_Lib_and_Detection_Models/tree/ccb5dfe09696edeea445ffb8f99adaa7c383ec3a/%E6%B5%8B%E8%AF%95%E8%A7%86%E9%A2%91).
These are unlabeled public demonstrations, possibly overlapping training data.
Proposal counts are not recall, precision or an accuracy ranking.

## Benchmark acceleration and output agreement

```bash
python tools/model-eval/benchmark_native.py \
  --model /tmp/armor-eval/downloads/aruw-2026/5_12_icon_detector.onnx \
  --clip /tmp/armor-eval/clips/tongji-demo.avi \
  --clip /tmp/armor-eval/clips/robotpilots-armor.mp4 \
  --output /tmp/armor-eval/benchmark
```

The runner checks native CPU, Core ML ALL, GPU-enabled, and Neural-Engine-enabled
modes; `--modes CPUOnly` adds a Core ML CPU control. It matches detections by class
and box overlap before comparing corners, because output rows can reorder.
Inference timings exclude preprocessing, video decoding and overlays. Separate
profiling reports CPU fallback rather than assuming all operations are accelerated.
Clip sampling decodes sequentially with a fixed-seed reservoir: some AVI files
report zero frame count or cannot seek accurately.

On the M5 test host, Core ML ALL took 5.09 ms median versus native ORT CPU 8.96 ms
and Core ML CPU 8.05 ms. All 71 above-threshold outputs in 48 sampled frames matched
CPU with maximum corner displacement below 0.0001 model pixels. The compute plan
assigned 397 operations to GPU and 8 to CPU inside Core ML; another 10 operations
remained in ORT CPU. Neural-Engine-enabled mode was slower and did not prove ANE
usage. This is not Jetson performance. [Recorded results](../../docs/validation/model-native-mac-2026-09-30.json).

## Produce a video with real C++ tracking overlays

First run native inference:

```bash
python tools/model-eval/replay_video.py infer \
  --model /tmp/armor-eval/downloads/aruw-2026/5_12_icon_detector.onnx \
  --video /tmp/armor-eval/clips/robotpilots-medium-spin.avi \
  --output /tmp/armor-eval/replay
```

Then process its tensors with the production decoder and tracker, using the test
image/build produced above:

```bash
docker run --rm --network none \
  -v "$PWD:/ws/src/CV_Detection:ro" \
  -v "$PWD/.local-test-output:/ws/out:ro" \
  -v /tmp/armor-eval/replay:/replay \
  cv-detection-local-tests:humble bash -lc \
  'source /opt/ros/humble/setup.bash && source /ws/out/install/local_setup.bash && /ws/out/install/yolo_cpp/lib/yolo_cpp/armor_replay /replay/tensors.txt /replay/tracks.jsonl 0.5'
```

Use your `CV_TEST_IMAGE`/`CV_TEST_OUTPUT` values if overridden. The confidence
argument must match the inference threshold, which defaults to 0.5. Finally:

```bash
python tools/model-eval/replay_video.py render --output /tmp/armor-eval/replay
open /tmp/armor-eval/replay/overlay.mp4
```

With FFmpeg installed, rendering produces H.264 `overlay.mp4`; otherwise it keeps
`overlay-mjpeg.avi`, which can be opened in a compatible native video player.
It also produces `preview.jpg`, raw tensors, `tracks.jsonl`, inference metadata and
`tracking-summary.json`. `--max-frames 600` limits an inference run for a quick check.
Add `--playback-speed 0.25` to render fast spin recordings at quarter speed; the
tracker still uses the recording's original frame timing.

The overlay includes ordered corners, plate IDs/color/number/confidence, trails,
tracker prediction, one-frame coasting and selected plate pixel error. It uses
the production default tracker settings, with both colors selectable for offline
inspection. A plate ID is not a robot identity: legitimate spinning-plate changes,
label changes and association failures can all increase it. Counts have no
ground-truth identity labels and must not be interpreted as identity-error rates.

The pixel arrow is **not an absolute yaw/pitch command**. These clips do not supply
a validated calibration/attitude contract for our camera. Image motion includes
camera motion. 3D pose, robot-center/rotation estimates and controller commands
must be tested with calibrated synchronized recordings or the synthetic production
pipeline. TX and fire are absent from this replay executable.

## Parser contract

ARUW's pinned Project Otto parser uses float BGR/255 stretched to 640×384, restores
source X/Y scales separately, orders four keypoints BL/TL/TR/BR, and derives the
tracked rectangle from those keypoints. Our explicit C++ adapter follows these
geometry conventions. Its class mapping follows the model metadata rather than
upstream competition-specific remapping of infantry labels to sentry.

The public spin replay exposed duplicate rows for the same plate even with the
end-to-end head. The adapter suppresses same-color quadrilaterals only when their
intersection-over-union exceeds 0.85, keeping the highest confidence. A regression
test checks duplicates disappear while neighboring plates remain separate.
The [four-clip replay report](../../docs/validation/aruw-replay-2026-09-30.json)
preserves both before/after counts, including cases where the number of track IDs
increased. Spin identity churn remains unresolved; fewer selected-ID changes are
not proof of more accurate association.

Reference: [ARUW parser](https://gitlab.com/aruw/vision/project-otto/-/blob/154035fc34ffb34f99120ffa8e8a2bb098bc6306/project_otto/target_detector/_target_detector.py),
[Core ML execution provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).
Model/source licenses remain as recorded in the [upgrade report](../../docs/validation/autoaim-local-upgrade.md#upstream-references-and-licenses).
