#!/usr/bin/env python3
"""Generate deterministic multi-armor robot observations without ROS.

The output keeps robot and plate identity in a separate ``truth`` section. A
tracker consuming ``observations`` therefore has to solve the same association
problem as the live detector path.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MotionSegment:
    duration_s: float
    velocity_x_mps: float
    velocity_z_mps: float
    angular_velocity_rad_s: float


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    center_x_m: float
    center_z_m: float
    phase_rad: float
    radius_m: float
    plate_count: int
    segments: tuple[MotionSegment, ...]


@dataclass(frozen=True)
class Scenario:
    name: str
    robots: tuple[RobotSpec, ...]

    @property
    def duration_s(self) -> float:
        return max(sum(segment.duration_s for segment in robot.segments) for robot in self.robots)


def wrap_angle(angle: float) -> float:
    return math.remainder(angle, 2.0 * math.pi)


def robot_state(robot: RobotSpec, timestamp_s: float) -> tuple[float, float, float, float]:
    """Return center x/z, phase, and angular velocity at a scenario timestamp."""
    remaining = max(0.0, timestamp_s)
    x_m = robot.center_x_m
    z_m = robot.center_z_m
    phase_rad = robot.phase_rad
    angular_velocity_rad_s = 0.0
    for segment in robot.segments:
        elapsed = min(remaining, segment.duration_s)
        x_m += segment.velocity_x_mps * elapsed
        z_m += segment.velocity_z_mps * elapsed
        phase_rad += segment.angular_velocity_rad_s * elapsed
        angular_velocity_rad_s = segment.angular_velocity_rad_s
        remaining -= elapsed
        if remaining <= 0.0:
            break
    return x_m, z_m, wrap_angle(phase_rad), angular_velocity_rad_s


def plate_geometry(
    robot: RobotSpec, center_x_m: float, center_z_m: float, phase_rad: float, slot: int
) -> tuple[float, float, float, float]:
    """Return plate position and outward normal in the ground plane."""
    angle = phase_rad + (2.0 * math.pi * slot / robot.plate_count)
    normal_x = math.sin(angle)
    normal_z = -math.cos(angle)
    return (
        center_x_m + robot.radius_m * normal_x,
        center_z_m + robot.radius_m * normal_z,
        normal_x,
        normal_z,
    )


def is_visible(
    plate_x_m: float,
    plate_z_m: float,
    normal_x: float,
    normal_z: float,
    max_view_angle_deg: float,
) -> bool:
    if plate_z_m <= 0.0:
        return False
    camera_distance = math.hypot(plate_x_m, plate_z_m)
    to_camera_x = -plate_x_m / camera_distance
    to_camera_z = -plate_z_m / camera_distance
    facing_score = normal_x * to_camera_x + normal_z * to_camera_z
    return facing_score >= math.cos(math.radians(max_view_angle_deg))


def generate_frames(
    scenario: Scenario,
    *,
    fps: float = 60.0,
    seed: int = 1,
    position_noise_std_m: float = 0.005,
    normal_noise_std: float = 0.01,
    dropout_probability: float = 0.0,
    max_view_angle_deg: float = 78.0,
) -> list[dict]:
    """Generate observations and private evaluator truth for one scenario."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if not 0.0 <= dropout_probability <= 1.0:
        raise ValueError("dropout_probability must be between zero and one")

    rng = random.Random(seed)
    frame_count = int(math.floor(scenario.duration_s * fps)) + 1
    frames: list[dict] = []
    for frame_index in range(frame_count):
        timestamp_s = min(frame_index / fps, scenario.duration_s)
        observations: list[dict] = []
        truth_robots: list[dict] = []
        associations: dict[str, dict] = {}
        for robot in scenario.robots:
            center_x_m, center_z_m, phase_rad, angular_velocity_rad_s = robot_state(
                robot, timestamp_s
            )
            truth_robots.append(
                {
                    "robot_id": robot.robot_id,
                    "center_x_m": center_x_m,
                    "center_z_m": center_z_m,
                    "phase_rad": phase_rad,
                    "angular_velocity_rad_s": angular_velocity_rad_s,
                    "plate_count": robot.plate_count,
                    "radius_m": robot.radius_m,
                }
            )
            for slot in range(robot.plate_count):
                plate_x_m, plate_z_m, normal_x, normal_z = plate_geometry(
                    robot, center_x_m, center_z_m, phase_rad, slot
                )
                if not is_visible(
                    plate_x_m,
                    plate_z_m,
                    normal_x,
                    normal_z,
                    max_view_angle_deg,
                ):
                    continue
                if rng.random() < dropout_probability:
                    continue
                observation_id = f"f{frame_index:05d}o{len(observations):02d}"
                observations.append(
                    {
                        "observation_id": observation_id,
                        "position_x_m": plate_x_m + rng.gauss(0.0, position_noise_std_m),
                        "position_z_m": plate_z_m + rng.gauss(0.0, position_noise_std_m),
                        "normal_x": normal_x + rng.gauss(0.0, normal_noise_std),
                        "normal_z": normal_z + rng.gauss(0.0, normal_noise_std),
                        "confidence": 0.95,
                    }
                )
                associations[observation_id] = {
                    "robot_id": robot.robot_id,
                    "plate_slot": slot,
                }
        frames.append(
            {
                "scenario": scenario.name,
                "frame_index": frame_index,
                "timestamp_s": timestamp_s,
                "observations": observations,
                "truth": {"robots": truth_robots, "associations": associations},
            }
        )
    return frames


def scenarios() -> dict[str, Scenario]:
    four_plate = dict(radius_m=0.22, plate_count=4)
    return {
        "stationary_single_plate": Scenario(
            "stationary_single_plate",
            (
                RobotSpec(
                    "standalone_plate",
                    0.35,
                    3.5,
                    0.0,
                    radius_m=0.0,
                    plate_count=1,
                    segments=(MotionSegment(3.0, 0.0, 0.0, 0.0),),
                ),
            ),
        ),
        "nonspin_translate": Scenario(
            "nonspin_translate",
            (
                RobotSpec(
                    "robot_a",
                    -0.8,
                    4.0,
                    0.0,
                    segments=(MotionSegment(4.0, 0.4, 0.0, 0.0),),
                    **four_plate,
                ),
            ),
        ),
        "spin_stationary": Scenario(
            "spin_stationary",
            (
                RobotSpec(
                    "robot_a",
                    0.0,
                    4.0,
                    0.0,
                    segments=(MotionSegment(4.0, 0.0, 0.0, 7.0),),
                    **four_plate,
                ),
            ),
        ),
        "spin_translate": Scenario(
            "spin_translate",
            (
                RobotSpec(
                    "robot_a",
                    -0.8,
                    4.0,
                    0.0,
                    segments=(MotionSegment(4.0, 0.4, 0.0, 7.0),),
                    **four_plate,
                ),
            ),
        ),
        "spin_stop_reverse": Scenario(
            "spin_stop_reverse",
            (
                RobotSpec(
                    "robot_a",
                    0.0,
                    4.0,
                    0.0,
                    segments=(
                        MotionSegment(1.5, 0.2, 0.0, 7.0),
                        MotionSegment(1.0, 0.2, 0.0, 0.0),
                        MotionSegment(1.5, 0.2, 0.0, -7.0),
                    ),
                    **four_plate,
                ),
            ),
        ),
        "two_robot_crossing": Scenario(
            "two_robot_crossing",
            (
                RobotSpec(
                    "robot_a",
                    -1.2,
                    4.0,
                    0.0,
                    segments=(MotionSegment(4.0, 0.6, 0.0, 5.0),),
                    **four_plate,
                ),
                RobotSpec(
                    "robot_b",
                    1.2,
                    4.4,
                    0.3,
                    segments=(MotionSegment(4.0, -0.6, 0.0, 0.0),),
                    **four_plate,
                ),
            ),
        ),
    }


def write_jsonl(frames: Iterable[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(scenarios()), default="spin_stop_reverse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--position-noise-std-m", type=float, default=0.005)
    parser.add_argument("--dropout-probability", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = scenarios()[args.scenario]
    frames = generate_frames(
        scenario,
        fps=args.fps,
        seed=args.seed,
        position_noise_std_m=args.position_noise_std_m,
        dropout_probability=args.dropout_probability,
    )
    write_jsonl(frames, args.output)
    observation_count = sum(len(frame["observations"]) for frame in frames)
    print(
        json.dumps(
            {
                "scenario": scenario.name,
                "frames": len(frames),
                "observations": observation_count,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
