# inspect-robots-unitree-g1

[![CI](https://github.com/robocurve/inspect-robots-unitree-g1/actions/workflows/ci.yml/badge.svg)](https://github.com/robocurve/inspect-robots-unitree-g1/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/inspect-robots-unitree-g1)](https://pypi.org/project/inspect-robots-unitree-g1/)
[![Python](https://img.shields.io/pypi/pyversions/inspect-robots-unitree-g1)](https://pypi.org/project/inspect-robots-unitree-g1/)

Inspect Robots adapters for a standing Unitree G1 with 7-DoF arms and an
Isaac-GR00T PolicyServer. The package registers the `g1_arms` embodiment and
the `gr00t` policy. Both declare one absolute 16-D joint-position contract.

The locomotion controller remains responsible for balance, legs, and waist.
This adapter publishes arm targets to `rt/arm_sdk` and blends them into the
running controller through the weight joint.

## Install

Install the package on the G1 adapter machine:

```bash
pip install inspect-robots-unitree-g1
pip install "unitree_sdk2py @ git+https://github.com/unitreerobotics/unitree_sdk2_python"
pip install pyzmq msgpack msgpack-numpy opencv-python-headless
```

A known-good SDK combination is Python 3.12, unitree_sdk2py 1.0.1, and
cyclonedds 0.10.2. If the CycloneDDS wheel is unavailable, build CycloneDDS
from source, set `CYCLONEDDS_HOME`, then reinstall the Python package.

The adapter machine must be PC2 or a tethered Linux host with layer-2 access
to `192.168.123.0/24`. Run a compatible raw-JPEG ZMQ image server for the head
D435i on PC2. The builtin reader uses a conflated SUB socket. Inject
`camera_reader` when your image server uses another protocol.

Run Isaac-GR00T on the GPU machine with its PolicyServer and the G1 full-body
modality checkpoint. At the pinned API seam, this command loads the checkpoint
and listens on port 5555:

```bash
export GR00T_CHECKPOINT=/path/to/your/g1-checkpoint
export GR00T_EMBODIMENT_TAG=your_checkpoint_g1_tag
python - <<'PY'
import os

from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyServer

policy = Gr00tPolicy(
    embodiment_tag=os.environ["GR00T_EMBODIMENT_TAG"],
    model_path=os.environ["GR00T_CHECKPOINT"],
    device="cuda:0",
)
PolicyServer.start_server(policy, port=5555)
PY
```

Use the embodiment tag advertised by that checkpoint. This package
intentionally does not depend on Isaac-GR00T itself.

## Preflight

Preflight checks declarations without connecting to hardware or a server:

```bash
inspect-robots-unitree-g1-preflight --dry-run
inspect-robots-unitree-g1-preflight --task cubepick-reach --json --dry-run
```

The JSON form includes `dry_run`. A green result proves dimensions, semantics,
cameras, state keys, and optional scene realizability. It cannot prove action
relativity, hand polarity, network reachability, or safe physical clearance.

## Run on hardware

Example `config.ini`:

```ini
[policy]
name = gr00t

[policy.args]
host = 192.168.123.10
port = 5555
actions_are_relative = false

[embodiment]
name = g1_arms

[embodiment.args]
iface = eth0
hand_type = dex3
cam_server_address = tcp://192.168.123.164:5556
unattended = false
```

Then use the normal Inspect Robots run or eval command for your task. The
policy server defaults to port 5555. The camera defaults to port 5556 so the
two services do not collide.

## Safety

> [!WARNING]
> A G1 arm-sdk target and actual-pose mismatch during an abrupt weight change
> can snap the arms at high speed. Never replace the measured-pose weight ramp
> with a direct weight flip.

On the first reset, the adapter reads the observed arm pose, seeds every
commanded target from it, ramps arm-sdk weight from 0 to 1 while holding that
pose, then streams to the zero home pose. Later resets read and reseed from the
fresh pose before streaming home, with no weight change.

Close first opens both hands. It then streams the arms to `rest_pose`, or to
the pose observed at close start when no rest pose is configured. Finally it
ramps weight from 1 to 0 and disconnects. Disconnect is attempted even when a
publish or ramp fails.

> [!WARNING]
> No public arm-sdk firmware watchdog is documented. If a process dies while
> weight is 1, the controller can keep holding its last target. The adapter
> registers both an `atexit` hook and a SIGTERM handler. SIGKILL cannot be
> intercepted. Keep an operator holding the remote within reach for every run.

Remote `L1+A` enters damping mode. The robot goes limp and physically sinks,
so the operator must also be ready to support the robot and protect nearby
people and objects.

> [!WARNING]
> Stock Isaac-GR00T at the pinned N1.7 seam decodes relative arm outputs to
> absolute physical units on the server. Keep `actions_are_relative=false` for
> that server. Set it to true only for an older or custom server that returns
> offsets. A mismatch cannot be detected by compatibility checks and can cause
> runaway targets. Staff the stop control and slow-jog every new server build.

When relative conversion is enabled, every chunk row is added independently
to the same observed arm anchor. It is never cumulatively summed, and the hand
slots are never modified.

> [!WARNING]
> `control_hz` must match the checkpoint data frame rate. A compatible shape
> does not prove compatible timing. Verify the dataset rate before motion.

Before the first grasp, bench-verify Dex1 polarity at low speed. The cited
xr_teleoperate source maps 0 rad to closed and 5.4 rad to open, so this package
maps normalized 1 to 5.4 rad. Hardware revisions and calibration can differ.

Always run preflight, clear the workspace, verify camera orientation, inspect
the measured pose, and perform a slow single-joint jog before policy control.

## Configuration

### G1 embodiment

| Field | Default | Meaning |
|---|---:|---|
| `iface` | `eth0` | DDS network interface, or `None` for default routing |
| `hand_type` | `dex3` | `dex1` gripper or `dex3` three-finger hand |
| `control_hz` | `10` | Framework action rate |
| `stream_hz` | `50` | Synchronous arm publish rate |
| `max_joint_speed` | `3.0` | Per-joint hard speed cap in rad/s |
| `weight_ramp_s` | `2.0` | Blend-up and blend-down duration |
| `kp_arm`, `kd_arm` | `60`, `1.5` | Shoulder and elbow gains |
| `kp_wrist`, `kd_wrist` | `40`, `1.5` | Conservative package wrist gains |
| `home_pose` | zero arms, open hands | Reset target |
| `rest_pose` | `None` | Close target, or measured close-start pose |
| `dex1_stroke` | `5.4` | Full open motor stroke in radians |
| `dex1_max_speed` | `2.7` | Dex1 motor speed limit in rad/s |
| `hand_deadband` | `0.05` | Normalized on-change publish gate |
| `dex3_thumb_swing` | `0.0` | Closed-reference thumb0 swing |
| `cam_server_address` | `tcp://192.168.123.164:5556` | JPEG SUB endpoint |
| `cam_timeout_s` | `5.0` | Camera receive timeout |
| `unattended` | `false` | Skip readiness and scoring prompts |

The arm limits come from the official 29-DOF G1 URDF, pulled 0.05 radians
inward. The official 50 Hz arm example supplies the arm gains. The lower wrist
gain is this package's bench-tuned conservative choice.

Dex1 defaults use kp 5.0 and kd 0.05. Dex3 defaults use kp 1.5 and kd 0.2.
Dex3 motor ordering differs by side, as required by the Unitree driver. The
open pose is zero. Closed reference poses are conservative interior values
derived from the left and right dex3_1 URDF limits because xr_teleoperate does
not publish fixed grasp poses.

### GR00T policy

| Field | Default | Meaning |
|---|---:|---|
| `host` | `127.0.0.1` | PolicyServer host |
| `port` | `5555` | PolicyServer port |
| `timeout_s` | `15` | REQ send and receive timeout |
| `actions_are_relative` | `false` | Enable legacy anchor-add conversion |
| `action_horizon` | `16` | Maximum returned prefix |
| `replan_interval` | `8` | Inspect Robots chunk metadata |
| `image_key` | `video.ego_view` | Dotted wire template key |
| `state_keys` | G1 N1.7 mapping | Dotted state source templates |
| `action_keys` | G1 N1.7 mapping | Returned action assembly templates |

The SourceSpec DSL supports `("packed", start, stop)`, `("hand", side)`,
`("state", field_key)`, and `("const", dim, value)`. Dotted config keys are
split at the first dot. Wire inner keys are unprefixed. Returned actions may
use either unprefixed names or `action.`-prefixed names. Unknown returned keys,
including locomotion outputs, are ignored.

### The 16-D contract

| Slots | Values | Units |
|---|---|---|
| 0 to 6 | left shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw | rad |
| 7 | left hand | normalized, 1 open |
| 8 to 14 | right shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw | rad |
| 15 | right hand | normalized, 1 open |

The scalar hand contract is a deliberate v1 simplification. Dex3 expands the
scalar along a fixed power-grasp path. Per-joint hand actions, locomotion,
waist and neck control, 23-DOF G1 arms, GR00T-SONIC, simulation, UnifoLM-VLA,
and lerobot-served policy backends are future work.

## Development

```bash
uv venv
uv pip install -e ".[dev]"
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest --cov --cov-report=term-missing
```

Hardware, camera, ZMQ, and server behavior are fully injectable. Package
import requires only Inspect Robots and NumPy.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
