# Robot Tracking and Sim-to-Real Plan

<!-- SCOPE: Consolidated implementation and validation plan for robot-level armor tracking, safe observation, simulation, replay, and field promotion. -->
<!-- DOC_KIND: plan -->
<!-- DOC_ROLE: canonical -->
<!-- READ_WHEN: Read before changing target association, spin estimation, fire recommendation, observation mode, or tracking validation. -->
<!-- SKIP_WHEN: Skip for detector export, Jetson access, or unrelated deployment work. -->
<!-- PRIMARY_SOURCES: yolo_cpp/src/tracker.cpp, yolo_cpp/src/armor_processor_node.cpp, yolo_cpp/src/aim_solver_node.cpp, yolo_cpp/tools/robot_tracking_scenario.py -->

## Goal

Track a physical robot continuously while its visible armor plates remain fixed, rotate, disappear, or switch. The same state model must cover zero angular velocity, ordinary turns, and continuous spin. Validate every stage on the development Mac before enabling robot-facing output.

This work stays on `codex/robot-tracking-sim2real`. The Type-C firmware remains responsible for real-time actuator control, physical limits, command freshness, and operator authorization. The Jetson owns visual observations, robot identity, armor association, motion estimation, prediction, and fire recommendations.

## Current Decision

The current armor detector is sufficient for the first implementation. It already provides boxes, corners, class/color information, armor number when available, and planar PnP inputs. Add detector training or a second model only if held-out replay shows that missed detections, wrong corners, or poor classification dominate tracking failures.

The current plate tracker is not sufficient as the final estimator. It tracks individual plates, resets downstream filtering when the selected plate ID changes, and does not preserve one robot identity across normal armor switches.

## Safety Contract

`enable_fire_recommendation` defaults to `false` in every launch. While false, both the legacy `suggest_fire` topic and `AimSolution.suggest_fire` remain false even if an upstream processor requests fire. A person must set `enable_fire_recommendation:=true` in the competition configuration and restart the stack to allow recommendations.

The fire gate is separate from movement transmission:

- `enable_fire_recommendation:=false` prevents a positive fire recommendation.
- `transmit_enable:=false` prevents controller-facing movement packets.
- The planned observation-only launch omits the solver and serial transmitter entirely.

The browser preview from the full stack is not an isolation boundary. Use the observation-only launch after it is implemented for live perception debugging.

## Target State

Maintain one persistent state per physical robot:

```text
center position and velocity
rotation phase and angular velocity
armor count
front/side radii and height offsets
plate association hypotheses
state covariance and last-observation age
```

For armor slot `i`:

```text
plate_i(t) = center(t) + rotation(phase(t) + slot_offset_i) * radius_i
```

The non-spinning case is the same model with angular velocity near zero. Do not reset robot identity when entering or leaving spin mode.

## Work Plan

### 1. Fail-closed observation and fire behavior

- Keep fire recommendations opt-in.
- Add an observation-only launch containing camera, detector, robot tracker, preview, and recorder.
- Exclude aim command publishers and controller transmission.
- Add a prominent `OBSERVATION ONLY` status to the preview.
- Add an automated assertion that the profile publishes no command or fire topics and opens no writable serial transport.

Exit gate: startup, restart, target acquisition, and target loss produce zero positive fire recommendations and zero serial writes.

### 2. Native deterministic simulation

Use [`robot_tracking_scenario.py`](../../yolo_cpp/tools/robot_tracking_scenario.py) on the Mac. It generates detector-boundary observations while keeping robot and plate identity only in evaluator truth.

Required scenarios:

- translating robot with no spin
- stationary spin
- translating spin
- spin, stop, and reversal
- two robots crossing
- later additions for occlusion, delayed frames, out-of-order frames, duplicate detections, unequal radii, and different armor counts

Run the current native checks:

```bash
python3 yolo_cpp/tools/test_robot_tracking_scenario.py
python3 yolo_cpp/tools/robot_tracking_scenario.py \
  --scenario spin_stop_reverse \
  --output /tmp/robot-tracking-spin-stop-reverse.jsonl
```

Exit gate: deterministic scenarios cover spin and non-spin transitions without leaking truth identity into tracker input.

### 3. Robot-level association baseline

- Introduce a robot track above plate tracks.
- Associate observations using predicted plate geometry and Mahalanobis distance.
- Preserve robot ID through plate entry and exit.
- Keep non-spinning, clockwise, and counter-clockwise hypotheses until evidence separates them.
- Publish association confidence, split/merge events, and rejection reasons.

Exit gate: one rotating robot remains one robot; two crossing robots do not merge; a stopped spinner retains identity.

### 4. Capture-time state and pose quality

- Preserve exposure time through detector, tracker, processor, and solver.
- Interpolate gimbal attitude at exposure time.
- Reject stale or unbracketed attitude.
- Evaluate both planar pose hypotheses when ambiguous.
- Calculate reprojection error and carry observation uncertainty into association.

Exit gate: a stationary plate remains stationary during gimbal motion, and timing residuals meet the angular-error budget at measured maximum gimbal speed.

### 5. Robot estimator

- Start with a small EKF or equivalent center/velocity plus phase/rate filter.
- Use bounded, robot-specific geometry priors.
- Grow uncertainty during missing observations.
- Treat near-zero angular velocity as the ordinary non-spin case.
- Add a multi-hypothesis or particle filter only if replay shows plate ambiguity is the dominant remaining failure.

Exit gate: center motion has no artificial velocity spike at a plate switch, and predicted plate error improves at the measured impact horizon for both spin and non-spin datasets.

### 6. Future-plate and shot-window prediction

- Predict robot center and each plate to total impact time.
- Choose a plate that will face the launcher and remain reachable.
- Keep tracking validity separate from firing validity.
- Require fresh state, bounded uncertainty, valid trajectory, and measured yaw/pitch alignment before recommending fire.

Exit gate: controlled trials improve damage per second and time to defeat moving spinners without materially degrading non-spinning targets.

### 7. Detector and camera improvement only when measured

- Tune fixed exposure, gain, focus, and calibration under match lighting and motion.
- Record continuous raw and annotated video with synchronized telemetry.
- Mine failures into session-separated training and validation sets.
- Retrain or replace the detector only after evidence identifies it as the bottleneck.

Exit gate: any detector change improves robot-track continuity and corner quality on held-out sessions without unacceptable Jetson latency regression.

## Sim-to-Real Ladder

Promote the same estimator through these levels. Do not skip a level after a state-model or timing change.

| Level | Environment | Evidence required |
|---|---|---|
| 0 | Native Mac deterministic simulation | Unit tests, truth-state error, ID switches, split/merge counts, deterministic replay |
| 1 | Mac fault injection | Noise, dropout, latency, out-of-order data, geometry mismatch, transition tests |
| 2 | Docker ROS integration | Same observation boundary, launch tests, topic contract, recorded output, no-fire assertions |
| 3 | Recorded real-camera replay | Held-out lighting, blur, occlusion, crossings, spin/non-spin balance, repeatable metrics |
| 4 | Live observation-only robot | No command path, measured timing, previewed robot/plate IDs, stationary-target sweep |
| 5 | Actuator bench without launcher | Low-rate aim response, expiry, limits, overshoot, no fire authority |
| 6 | Controlled launcher range | Impact-time prediction, hit rate, damage rate, ammunition and heat efficiency |
| 7 | Scrimmage | Reliability, reacquisition, defeat time, tactical value, rollback readiness |

At each level, compare the candidate against the current tracker on identical inputs. Report spin and non-spin results separately so one class cannot hide regressions in the other.

## Primary Metrics

- physical robot ID switches
- false robot merges and splits
- track coverage and reacquisition time
- robot-center error and continuity
- angular velocity and phase error
- predicted plate error at 50, 100, and 200 ms horizons
- capture-to-track and capture-to-command latency percentiles
- positive fire-recommendation count while disabled; required value is zero
- serial write count in observation-only mode; required value is zero
- hit rate, damage per second, ammunition used, and time to defeat in controlled firing tests

## Maintenance

**Update Triggers:**
- Tracker state, observation contract, launch isolation, or promotion gates change.
- A new scenario or robot geometry becomes required.
- Replay evidence changes the detector or estimator priority.

**Verification:**
- [ ] Native scenario tests pass on macOS.
- [ ] Fire recommendations remain opt-in.
- [ ] Spin and non-spin metrics are reported separately.
- [ ] Observation-only mode has no command path before live use.
- [ ] Every hardware claim is backed by a recorded test.

**Last Updated:** 2026-09-29
