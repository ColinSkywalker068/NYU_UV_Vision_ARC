# Development Notes

Related pages: [Overview](./overview.md) | [Pipeline](./pipeline.md) | [Nodes](./nodes.md) | [Interfaces](./interfaces.md) | [Legacy Pipeline Notes](./legacy-pipeline-notes.md) | [Launch and Config](./launch-and-config.md)

## Where to Extend the Package

If you are changing model execution or backend behavior, start in the detector backend and detector node code. If you are changing target stability or selection, start in the tracker library and `ArmorTrackerNode`. If you are changing controller-facing outputs, start in `AimSolverNode` or the STM bridge.

The package is easier to work with when you keep those responsibilities separate. A small change in one stage is usually easier to validate than editing the combined perception path first.

## Useful Entry Points

- detector runtime: [`../src/armor_detector_node.cpp`](../src/armor_detector_node.cpp)
- tracker runtime: [`../src/armor_tracker_node.cpp`](../src/armor_tracker_node.cpp)
- solver runtime: [`../src/aim_solver_node.cpp`](../src/aim_solver_node.cpp)
- combined reference path: [`../src/perception_node.cpp`](../src/perception_node.cpp)
- FPS isolation harness: [`../src/fps_benchmark.cpp`](../src/fps_benchmark.cpp)
- tracking tests: [`../test/test_tracking_core.cpp`](../test/test_tracking_core.cpp)

## Verification Flow

When you touch behavior in this package, a good minimum loop is:

1. Build `yolo_cpp` by itself.
2. Run the launch file closest to the path you changed.
3. Check that the expected topics are present and publishing.
4. Revisit the tracker test if the change touched association or target persistence.

## Jetson FPS Isolation

Use the benchmark harness on the Jetson AGX Orin when camera FPS and model FPS need to be separated:

```bash
./services/fps-benchmark.sh --mode all --duration 10 --warmup 2
```

The harness uses the STM model path by default, or `CV_STM_MODEL_PATH` when set. `inference` mode runs the same TensorRT engine on synthetic frames without a camera, `camera` mode pulls frames through `CameraBackend` without inference, and `deepstream` mode measures the Basler + DeepStream result cadence.

For full-FOV Basler downsampling tests, use the DeepStream pre-mux resize switch. This keeps the camera at 1920x1200 full FOV, resizes to the model-aspect box before `nvstreammux`, then lets `nvstreammux` pad to 640x640:

```bash
./services/fps-benchmark.sh --mode deepstream --duration 10 --warmup 2 --pre-mux-resize true
```

The observed `a2A1920-160ucBAS` / serial `41923000` GenICam node map did not expose `BinningSelector`, `BinningHorizontal`, `BinningVertical`, `BinningHorizontalMode`, `BinningVerticalMode`, `DecimationHorizontal`, `DecimationVertical`, `DecimationHorizontalMode`, `DecimationVerticalMode`, `ScalingHorizontal`, or `ScalingVertical`. It did expose writable `PixelFormat`, `Width`, `Height`, `OffsetX`, and `OffsetY`; those size controls are ROI/crop controls, not full-FOV binning.

For stage timing around the remaining conversion path, enable the DeepStream pad profiler for a short run:

```bash
./services/fps-benchmark.sh --mode deepstream --duration 10 --warmup 2 --pre-mux-resize true --basler-fps 110 --deepstream-profile-stages true
```

The profile log reports average milliseconds for source-to-Bayer conversion, queue handoff, `videoconvert`, `nvvideoconvert`, streammux handoff, and `nvinfer`.

## Related Repo Docs

- build and container workflow: [CPP_CONTAINER_RUNBOOK.md](../../CPP_CONTAINER_RUNBOOK.md)
- longer-term parity roadmap: [RM_AUTO_AIM_PARITY_NEXT_STEPS.md](../../docs/RM_AUTO_AIM_PARITY_NEXT_STEPS.md)
