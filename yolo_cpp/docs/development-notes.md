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
- tracking tests: [`../test/test_tracking_core.cpp`](../test/test_tracking_core.cpp)

## Verification Flow

When you touch behavior in this package, a good minimum loop is:

1. Build `yolo_cpp` by itself.
2. Run the launch file closest to the path you changed.
3. Check that the expected topics are present and publishing.
4. Revisit the tracker test if the change touched association or target persistence.

## Related Repo Docs

- build and container workflow: [CPP_CONTAINER_RUNBOOK.md](../../CPP_CONTAINER_RUNBOOK.md)
- longer-term parity roadmap: [RM_AUTO_AIM_PARITY_NEXT_STEPS.md](../../docs/RM_AUTO_AIM_PARITY_NEXT_STEPS.md)
