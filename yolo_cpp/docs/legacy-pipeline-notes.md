# Legacy Pipeline Notes

Related pages: [Overview](./overview.md) | [Pipeline](./pipeline.md) | [Nodes](./nodes.md) | [Interfaces](./interfaces.md) | [Launch and Config](./launch-and-config.md) | [Development Notes](./development-notes.md)

This page keeps the useful narrative from the older top-level pipeline writeup, but places it next to the current `yolo_cpp` docs so it is easier to compare the old mental model with the live modular stack.

## The Old Seven-Step Story

The older documentation described the vision system as a seven-step flow:

1. capture image data
2. preprocess the frame
3. run the model
4. extract useful target features
5. choose the best armor plate
6. compute yaw and pitch
7. hand the result to the controller

That framing is still helpful. The newer package just expresses those steps through ROS nodes and messages instead of one long explanation.

## What Still Matches Today

The older doc emphasized a few ideas that still hold:

- camera frames are normalized before inference
- the model can return multiple armor candidates per frame
- target choice should prefer stable, meaningful candidates instead of the noisiest box
- controller output is not raw detections, but reduced aiming information

Even though the current implementation is more modular, that older intuition is still a good way to explain the pipeline to someone new.

## What Changed

The biggest shift is structure. The older writeup read like one continuous pipeline, while the active stack is now split across:

- detector publication on `detector/armors`
- tracker publication on `tracker/tracked_armors`
- selected target publication on `tracker/target`
- solver output on `bounding_box`, `bounding_box_angles`, `override`, and `obb_angle`

That split makes the runtime easier to inspect, test, and swap piece by piece.

## Archived Original

The original illustrated version was moved to [legacy/docs/Pipeline/Pipeline.md](../../legacy/docs/Pipeline/Pipeline.md). Keep that copy around if you want the historical diagrams and the older teaching-style walkthrough.
