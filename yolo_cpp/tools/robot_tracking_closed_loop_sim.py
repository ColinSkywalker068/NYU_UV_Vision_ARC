#!/usr/bin/env python3
"""Run a deterministic robot tracker, gimbal, and fire-gate simulation.

This is an estimator/control harness. It does not model a launcher or projectile,
and its fire recommendation is only a boolean value written to the simulation
trace.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from robot_tracking_scenario import (
    RobotSpec,
    is_visible,
    plate_geometry,
    robot_state,
    scenarios,
    wrap_angle,
)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def world_to_camera(x: float, z: float, gimbal_yaw_rad: float) -> tuple[float, float]:
    cosine = math.cos(gimbal_yaw_rad)
    sine = math.sin(gimbal_yaw_rad)
    return cosine * x - sine * z, sine * x + cosine * z


def camera_to_world(x: float, z: float, gimbal_yaw_rad: float) -> tuple[float, float]:
    cosine = math.cos(gimbal_yaw_rad)
    sine = math.sin(gimbal_yaw_rad)
    return cosine * x + sine * z, -sine * x + cosine * z


@dataclass(frozen=True)
class GimbalConfig:
    proportional_gain: float = 10.0
    max_speed_rad_s: float = 5.0
    max_acceleration_rad_s2: float = 30.0


@dataclass(frozen=True)
class SolverConfig:
    prediction_horizon_s: float = 0.08
    fire_error_tolerance_deg: float = 0.6
    fire_max_gimbal_speed_rad_s: float = 0.6
    max_track_age_s: float = 0.05
    min_observation_frames: int = 5
    max_plate_view_angle_deg: float = 78.0


@dataclass
class RobotEstimate:
    initialized: bool = False
    center_x_m: float = 0.0
    center_z_m: float = 0.0
    velocity_x_mps: float = 0.0
    velocity_z_mps: float = 0.0
    phase_rad: float = 0.0
    angular_velocity_rad_s: float = 0.0
    timestamp_s: float = 0.0
    last_observation_s: float = -math.inf
    measurement_count: int = 0


class RobotEstimator:
    """Small alpha-beta robot-center and phase estimator for harness testing."""

    def __init__(self, radius_m: float, plate_count: int) -> None:
        self.radius_m = radius_m
        self.plate_count = plate_count
        self.state = RobotEstimate()
        self.position_alpha = 0.65
        self.position_beta = 0.08
        self.phase_alpha = 0.55
        self.phase_beta = 0.06

    def update(self, observations: list[dict], gimbal_yaw_rad: float, timestamp_s: float) -> None:
        if not observations:
            self.predict_to(timestamp_s)
            return

        centers: list[tuple[float, float]] = []
        plate_angles: list[float] = []
        for observation in observations:
            plate_x_m, plate_z_m = camera_to_world(
                observation["position_x_m"], observation["position_z_m"], gimbal_yaw_rad
            )
            normal_x, normal_z = camera_to_world(
                observation["normal_x"], observation["normal_z"], gimbal_yaw_rad
            )
            normal_length = max(1e-6, math.hypot(normal_x, normal_z))
            normal_x /= normal_length
            normal_z /= normal_length
            centers.append(
                (
                    plate_x_m - self.radius_m * normal_x,
                    plate_z_m - self.radius_m * normal_z,
                )
            )
            plate_angles.append(math.atan2(normal_x, -normal_z))

        center_x_m = statistics.fmean(center[0] for center in centers)
        center_z_m = statistics.fmean(center[1] for center in centers)
        if not self.state.initialized:
            self.state = RobotEstimate(
                initialized=True,
                center_x_m=center_x_m,
                center_z_m=center_z_m,
                phase_rad=plate_angles[0],
                timestamp_s=timestamp_s,
                last_observation_s=timestamp_s,
                measurement_count=1,
            )
            return

        dt = max(1e-4, timestamp_s - self.state.timestamp_s)
        predicted_x = self.state.center_x_m + self.state.velocity_x_mps * dt
        predicted_z = self.state.center_z_m + self.state.velocity_z_mps * dt
        residual_x = center_x_m - predicted_x
        residual_z = center_z_m - predicted_z
        self.state.center_x_m = predicted_x + self.position_alpha * residual_x
        self.state.center_z_m = predicted_z + self.position_alpha * residual_z
        self.state.velocity_x_mps += self.position_beta * residual_x / dt
        self.state.velocity_z_mps += self.position_beta * residual_z / dt

        predicted_phase = wrap_angle(
            self.state.phase_rad + self.state.angular_velocity_rad_s * dt
        )
        phase_residuals: list[float] = []
        for plate_angle in plate_angles:
            candidates = [
                wrap_angle(plate_angle - 2.0 * math.pi * slot / self.plate_count)
                for slot in range(self.plate_count)
            ]
            chosen = min(candidates, key=lambda value: abs(wrap_angle(value - predicted_phase)))
            phase_residuals.append(wrap_angle(chosen - predicted_phase))
        phase_residual = statistics.fmean(phase_residuals)
        self.state.phase_rad = wrap_angle(predicted_phase + self.phase_alpha * phase_residual)
        self.state.angular_velocity_rad_s = clamp(
            self.state.angular_velocity_rad_s + self.phase_beta * phase_residual / dt,
            -20.0,
            20.0,
        )
        self.state.timestamp_s = timestamp_s
        self.state.last_observation_s = timestamp_s
        self.state.measurement_count += 1

    def predict_to(self, timestamp_s: float) -> None:
        if not self.state.initialized or timestamp_s <= self.state.timestamp_s:
            return
        dt = timestamp_s - self.state.timestamp_s
        self.state.center_x_m += self.state.velocity_x_mps * dt
        self.state.center_z_m += self.state.velocity_z_mps * dt
        self.state.phase_rad = wrap_angle(
            self.state.phase_rad + self.state.angular_velocity_rad_s * dt
        )
        self.state.timestamp_s = timestamp_s

    def predicted_state(self, timestamp_s: float) -> tuple[float, float, float]:
        dt = max(0.0, timestamp_s - self.state.timestamp_s)
        return (
            self.state.center_x_m + self.state.velocity_x_mps * dt,
            self.state.center_z_m + self.state.velocity_z_mps * dt,
            wrap_angle(self.state.phase_rad + self.state.angular_velocity_rad_s * dt),
        )


def synthesize_observations(
    robot: RobotSpec,
    timestamp_s: float,
    gimbal_yaw_rad: float,
    rng: random.Random,
    *,
    horizontal_fov_deg: float,
    position_noise_std_m: float,
    dropout_probability: float,
    max_plate_view_angle_deg: float,
) -> list[dict]:
    center_x_m, center_z_m, phase_rad, _ = robot_state(robot, timestamp_s)
    observations: list[dict] = []
    for slot in range(robot.plate_count):
        plate_x_m, plate_z_m, normal_x, normal_z = plate_geometry(
            robot, center_x_m, center_z_m, phase_rad, slot
        )
        if not is_visible(
            plate_x_m,
            plate_z_m,
            normal_x,
            normal_z,
            max_plate_view_angle_deg,
        ):
            continue
        relative_bearing = wrap_angle(math.atan2(plate_x_m, plate_z_m) - gimbal_yaw_rad)
        if abs(relative_bearing) > math.radians(horizontal_fov_deg / 2.0):
            continue
        if rng.random() < dropout_probability:
            continue
        camera_x_m, camera_z_m = world_to_camera(plate_x_m, plate_z_m, gimbal_yaw_rad)
        camera_normal_x, camera_normal_z = world_to_camera(normal_x, normal_z, gimbal_yaw_rad)
        observations.append(
            {
                "position_x_m": camera_x_m + rng.gauss(0.0, position_noise_std_m),
                "position_z_m": camera_z_m + rng.gauss(0.0, position_noise_std_m),
                "normal_x": camera_normal_x + rng.gauss(0.0, 0.01),
                "normal_z": camera_normal_z + rng.gauss(0.0, 0.01),
                "confidence": 0.95,
            }
        )
    return observations


def best_plate_bearing(
    center_x_m: float,
    center_z_m: float,
    phase_rad: float,
    radius_m: float,
    plate_count: int,
    max_plate_view_angle_deg: float,
) -> tuple[float, int] | None:
    best: tuple[float, int, float] | None = None
    for slot in range(plate_count):
        angle = phase_rad + 2.0 * math.pi * slot / plate_count
        normal_x = math.sin(angle)
        normal_z = -math.cos(angle)
        plate_x_m = center_x_m + radius_m * normal_x
        plate_z_m = center_z_m + radius_m * normal_z
        if plate_z_m <= 0.0:
            continue
        distance = math.hypot(plate_x_m, plate_z_m)
        facing_score = normal_x * (-plate_x_m / distance) + normal_z * (-plate_z_m / distance)
        if facing_score < math.cos(math.radians(max_plate_view_angle_deg)):
            continue
        if best is None or facing_score > best[2]:
            best = (math.atan2(plate_x_m, plate_z_m), slot, facing_score)
    return None if best is None else (best[0], best[1])


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def run_closed_loop(
    robot: RobotSpec,
    *,
    fps: float = 60.0,
    control_rate_hz: float = 1000.0,
    seed: int = 1,
    position_noise_std_m: float = 0.005,
    dropout_probability: float = 0.0,
    enable_fire_recommendation: bool = False,
    gimbal_config: GimbalConfig = GimbalConfig(),
    solver_config: SolverConfig = SolverConfig(),
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    estimator = RobotEstimator(robot.radius_m, robot.plate_count)
    duration_s = sum(segment.duration_s for segment in robot.segments)
    control_dt = 1.0 / control_rate_hz
    vision_period = 1.0 / fps
    next_vision_s = 0.0
    command_yaw_rad = 0.0
    gimbal_yaw_rad = 0.0
    gimbal_velocity_rad_s = 0.0
    trace: list[dict] = []
    center_errors_m: list[float] = []
    command_errors_deg: list[float] = []
    fire_count = 0
    fire_truth_miss_count = 0
    first_fire_s: float | None = None

    step_count = int(math.ceil(duration_s * control_rate_hz)) + 1
    for step in range(step_count):
        timestamp_s = min(step * control_dt, duration_s)
        if timestamp_s + 1e-9 >= next_vision_s:
            observations = synthesize_observations(
                robot,
                timestamp_s,
                gimbal_yaw_rad,
                rng,
                horizontal_fov_deg=79.2,
                position_noise_std_m=position_noise_std_m,
                dropout_probability=dropout_probability,
                max_plate_view_angle_deg=solver_config.max_plate_view_angle_deg,
            )
            estimator.update(observations, gimbal_yaw_rad, timestamp_s)
            target_slot = -1
            true_target_slot = -1
            true_command_error_deg: float | None = None
            if estimator.state.initialized:
                impact_time_s = timestamp_s + solver_config.prediction_horizon_s
                estimated_x, estimated_z, estimated_phase = estimator.predicted_state(impact_time_s)
                selected = best_plate_bearing(
                    estimated_x,
                    estimated_z,
                    estimated_phase,
                    robot.radius_m,
                    robot.plate_count,
                    solver_config.max_plate_view_angle_deg,
                )
                if selected is not None:
                    command_yaw_rad, target_slot = selected

                true_x, true_z, true_phase, _ = robot_state(robot, impact_time_s)
                true_selected = best_plate_bearing(
                    true_x,
                    true_z,
                    true_phase,
                    robot.radius_m,
                    robot.plate_count,
                    solver_config.max_plate_view_angle_deg,
                )
                if true_selected is not None:
                    true_yaw_rad, true_target_slot = true_selected
                    true_command_error_deg = abs(
                        math.degrees(wrap_angle(command_yaw_rad - true_yaw_rad))
                    )
                    command_errors_deg.append(true_command_error_deg)

                current_true_x, current_true_z, current_true_phase, true_omega = robot_state(
                    robot, timestamp_s
                )
                center_error_m = math.hypot(
                    estimator.state.center_x_m - current_true_x,
                    estimator.state.center_z_m - current_true_z,
                )
                center_errors_m.append(center_error_m)
                tracking_age_s = timestamp_s - estimator.state.last_observation_s
                aim_error_rad = wrap_angle(command_yaw_rad - gimbal_yaw_rad)
                suggest_fire = (
                    enable_fire_recommendation
                    and target_slot >= 0
                    and estimator.state.measurement_count
                    >= solver_config.min_observation_frames
                    and tracking_age_s <= solver_config.max_track_age_s
                    and abs(math.degrees(aim_error_rad))
                    <= solver_config.fire_error_tolerance_deg
                    and abs(gimbal_velocity_rad_s)
                    <= solver_config.fire_max_gimbal_speed_rad_s
                )
                if suggest_fire:
                    fire_count += 1
                    if (
                        true_command_error_deg is not None
                        and true_command_error_deg > solver_config.fire_error_tolerance_deg
                    ):
                        fire_truth_miss_count += 1
                    if first_fire_s is None:
                        first_fire_s = timestamp_s
                trace.append(
                    {
                        "timestamp_s": timestamp_s,
                        "observation_count": len(observations),
                        "true_center_x_m": current_true_x,
                        "true_center_z_m": current_true_z,
                        "estimated_center_x_m": estimator.state.center_x_m,
                        "estimated_center_z_m": estimator.state.center_z_m,
                        "true_phase_rad": current_true_phase,
                        "estimated_phase_rad": estimator.state.phase_rad,
                        "true_angular_velocity_rad_s": true_omega,
                        "estimated_angular_velocity_rad_s": (
                            estimator.state.angular_velocity_rad_s
                        ),
                        "target_plate_slot": target_slot,
                        "true_target_plate_slot": true_target_slot,
                        "command_yaw_rad": command_yaw_rad,
                        "gimbal_yaw_rad": gimbal_yaw_rad,
                        "gimbal_velocity_rad_s": gimbal_velocity_rad_s,
                        "aim_error_deg": math.degrees(aim_error_rad),
                        "fire_recommendation": suggest_fire,
                        "evaluator_true_command_error_deg": true_command_error_deg,
                        "measurement_count": estimator.state.measurement_count,
                        "robot_radius_m": robot.radius_m,
                        "plate_count": robot.plate_count,
                    }
                )
            next_vision_s += vision_period

        yaw_error_rad = wrap_angle(command_yaw_rad - gimbal_yaw_rad)
        desired_velocity = clamp(
            gimbal_config.proportional_gain * yaw_error_rad,
            -gimbal_config.max_speed_rad_s,
            gimbal_config.max_speed_rad_s,
        )
        max_velocity_change = gimbal_config.max_acceleration_rad_s2 * control_dt
        gimbal_velocity_rad_s += clamp(
            desired_velocity - gimbal_velocity_rad_s,
            -max_velocity_change,
            max_velocity_change,
        )
        gimbal_yaw_rad = wrap_angle(gimbal_yaw_rad + gimbal_velocity_rad_s * control_dt)

    summary = {
        "frames": len(trace),
        "center_rmse_m": (
            math.sqrt(statistics.fmean(error * error for error in center_errors_m))
            if center_errors_m
            else math.nan
        ),
        "command_error_p95_deg": percentile(command_errors_deg, 0.95),
        "fire_recommendations_enabled": enable_fire_recommendation,
        "fire_recommendation_frames": fire_count,
        "fire_recommendation_truth_miss_frames": fire_truth_miss_count,
        "fire_recommendation_truth_alignment_fraction": (
            (fire_count - fire_truth_miss_count) / fire_count if fire_count else None
        ),
        "first_fire_recommendation_s": first_fire_s,
        "note": "Boolean recommendations only; projectile and firing are not simulated",
    }
    return trace, summary


def parse_args() -> argparse.Namespace:
    available = {
        name: scenario
        for name, scenario in scenarios().items()
        if len(scenario.robots) == 1
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=sorted(available), default="spin_stop_reverse")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--position-noise-std-m", type=float, default=0.005)
    parser.add_argument("--dropout-probability", type=float, default=0.0)
    parser.add_argument("--enable-fire-recommendation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenario = scenarios()[args.scenario]
    trace, summary = run_closed_loop(
        scenario.robots[0],
        fps=args.fps,
        seed=args.seed,
        position_noise_std_m=args.position_noise_std_m,
        dropout_probability=args.dropout_probability,
        enable_fire_recommendation=args.enable_fire_recommendation,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for sample in trace:
            stream.write(json.dumps(sample, sort_keys=True) + "\n")
    summary["scenario"] = scenario.name
    summary["output"] = str(args.output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
