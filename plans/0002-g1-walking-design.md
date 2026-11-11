# 0002: Walking G1 embodiments for LLM agents (g1_base, g1_mobile)

Status: draft for review (2026-11-11)
Issue: RC-355 (Robocurve Linear), for the Pantograph robot safety hackathon (2026-11-14)

## Goal

Let an LLM agent make a Unitree G1 walk, and walk and use both arms in the
same trial, through Inspect Robots. Add two embodiments next to `g1_arms`
without changing it or the `gr00t` policy:

- `g1_base`: walking only, a 3-D per-step body displacement.
- `g1_mobile`: walking plus both hands by palm position, an 11-D per-step
  displacement, with Dex1-1 grippers or no hands.

## Why displacement, and why a separate subpackage

LLM agents act in small, checkable moves through the agent's `move_by`. A
body displacement per step (`base_forward`, `base_left`, `base_turn`) and a
palm displacement per step map onto that directly; joint-space control of
legs is not something an LLM can do. The robot's own locomotion controller
keeps it balanced.

`g1_arms` is an absolute 16-D joint contract shared with GR00T. Mixing
walking into it would change that contract, so the walking embodiments live
in `inspect_robots_unitree_g1.walking` with their own config, driver and
tests. They share only the Unitree install hint and the Dex1-1 figures
(stroke, gains) that `g1_arms` already uses.

## Contract

- `g1_base`: declared `joint_delta` over the three planar body degrees of
  freedom, because the core has no base-motion control mode. Bounds are the
  speed caps over `control_hz` (0.3, 0.2, 0.2 m/s and 0.5 rad/s at 10 Hz).
  State: `base_pose` (odometry from the reset pose), `base_vel`, `body_rpy`.
- `g1_mobile`: `eef_delta_pos`, `frame="base"`, `gripper="continuous"`. Dims:
  the three walking dims, `left_hand_x/y/z`, `right_hand_x/y/z` (torso frame),
  `left_grip`, `right_grip` (0 open, 1 closed). State adds `hand_pos` (6) and
  `grip` (2). Grip bounds are zero when no hand is configured on the real
  robot, which `doctor` reports as a zero-width warning on purpose.

## Real robot (`backend=sdk`)

- Walking: `unitree_sdk2py` `LocoClient.SetVelocity(vx, vy, omega, duration)`
  with `duration = velocity_hold_s` (0.3 s), so the robot stops on its own
  when commands stop, including when the process dies. A streamer thread
  re-sends at `stream_hz` and stands the robot when a step's deadline passes.
- The driver never changes the FSM mode. Connecting refuses zero-torque and
  damping (FSM ids 0 and 1). The operator sets walking mode with the remote.
- Arms (`g1_mobile`): `rt/arm_sdk` as `g1_arms` does it. The weight ramps from
  0 to 1 over 2 s while holding the measured pose; each joint command moves at
  most `arm_joint_step` per tick; close ramps the weight back to 0. Position
  IK (numpy, damped least squares, 29-DoF rev 1.0 arm from unitree_ros) moves
  shoulder and elbow and holds the wrist.
- Dex1-1 (`hands=dex1`): `rt/dex1/{left,right}/cmd` and `/state`, rate-limited
  from the measured stroke. Reset opens; close leaves the grippers holding.
- A read-only robot check, `inspect-robots-unitree-g1-robot-check`, listens to
  the state topics and asks the FSM id, and never creates a publisher.

## Simulation

- `kinematic`: dependency-free stand-in for tests and mock runs.
- `isaaclab`: Isaac Lab v2.3 on Isaac Sim 5.1, on simulation time.
  `g1_base` uses `Isaac-Velocity-Flat-G1-Play-v0` with its pretrained policy.
  `g1_mobile` uses the locomanipulation scene with the Agile lower-body
  policy; with `hands=dex1` it swaps in a G1 with Dex1-1 grippers converted
  from `g1_29dof_mode_15_with_dex1_1.urdf` (the walking policy reads body
  joints only, so it walks unchanged).

Verified with an LLM agent (Claude Opus 5.5) in Isaac: walk to a target;
walk to a table and move a cube (Dex3); walk to a table, grasp and lift a
cube (Dex1-1).

## Open before hardware

- The odometry topic (`rt/odommodestate`) and its velocity frame.
- The FSM id reported in walking mode on the lent robot's firmware.
- Whether Unitree's AES key changes how `unitree_sdk2py` connects.
- The Dex1-1 open and closed direction and the grasp point (`tool_offset_m`).
- The G1 mode: mode 15 wrists sit 5 mm further out than rev 1.0.
- The head camera angle: at a table it does not see the hands in simulation.

## Testing

Hardware-free tests with injected drivers, clock, threads, atexit and signal
modules (`tests/test_walking_*.py`), at the package's 100 percent statement
and branch coverage. Real SDK seams are `pragma: no cover`.
