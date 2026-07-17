# 0001: Unitree G1 embodiment + GR00T policy plugin

Status: draft (critique loop in progress)
Issue: #1

## Goal

Ship the Unitree G1 sibling of inspect-robots-franka/-yam/-so101: a plugin
registering a `g1_arms` embodiment (real G1 dual-arm control over
`rt/arm_sdk` while the balance controller keeps the robot standing) and a
`gr00t` policy (ZMQ client for Isaac-GR00T inference servers), declaring
one shared 16-D `joint_pos` contract with zero compat errors and zero
warnings.

Reference material: session scratchpad `unitree-g1-research.md` (stack
research 2026-07-17) and `franka/framework-contract.md`. Templates:
../inspect-robots-franka (proven scaffolding, delta-integration policy
pattern), ../inspect-robots-agibot-a2 plan (bimanual + intra-step
streaming pattern; that repo may still be in flight, its plan is the
reference), ../inspect-robots-yam.

## Stack decision (from the research)

- **Driver: unitree_sdk2_python** (BSD-3, git-only, NOT on PyPI: the
  guided-install seam `_unitree.py` carries
  `pip install "unitree_sdk2py @
  git+https://github.com/unitreerobotics/unitree_sdk2_python"` plus the
  known-good pin note (python 3.12 + unitree_sdk2py 1.0.1 +
  cyclonedds 0.10.2, and the CYCLONEDDS_HOME source-build fallback).
  Publish `LowCmd_` (unitree_hg IDL) on `rt/arm_sdk`, subscribe
  `LowState_` on `rt/lowstate`, stamp CRC on every publish.
  `ChannelFactoryInitialize(0, iface)` needs L2 adjacency on
  192.168.123.0/24: the README says the adapter runs on the robot's PC2
  or a tethered Linux box.
- **Arms-while-balancing is the supported pattern** (xr_teleoperate
  "motion mode"): arm_sdk blends into the running loco controller via the
  weight joint (`motor_cmd[29].q` in [0, 1]). Safety choreography is
  mandatory and lives in the embodiment (see Safety).
- **Joints**: left arm indices 15-21, right arm 22-28 (shoulder
  pitch/roll/yaw, elbow, wrist roll/pitch/yaw), radians. Waist 12-14 is
  NOT commanded (locked to the controller). The 23-DOF variant (5-DOF
  arms, wrist pitch/yaw invalid) is out of scope for v1 and rejected with
  a clear ConfigError if `arm_dof != 7` is requested; the field exists so
  the error message can say why.
- **Hands: pluggable end-effector drivers, uniform 16-D contract.** Both
  supported hand types are collapsed to ONE normalized scalar per hand
  (1 = open house convention), mirroring GR00T's own G1 config where
  "hands are controlled by binary signals like a gripper":
  - `dex1` (gripper): `rt/dex1/left|right/cmd`, `MotorCmds_`; scalar maps
    to the 0-5.4 rad stroke, rate-limited in rad/s
    (`dex1_max_speed=2.7` default, a conservative 2 s full stroke;
    xr_teleoperate's 0.18 rad/cycle at 250 Hz is 45 rad/s, so ours is
    deliberately far under it); per-publish deltas are derived from the
    publish rate. The claimed 1=open -> 0.0 rad mapping is flagged in
    the README as bench-verify-before-first-grasp (polarity unproven
    from public sources).
  - `dex3` (three-finger): `rt/dex3/left|right/cmd`, `HandCmd_` (7
    motors); scalar drives a power-grasp curl interpolating each joint
    between its open and closed reference pose (constants transcribed
    from xr_teleoperate's hand driver; thumb swing held configurable).
  Observation reads back the mean normalized closure. Documented
  prominently as a deliberate v1 simplification; the 7-DOF-per-hand
  action mode is future work.
- **Camera: ZMQ JPEG client seam.** The G1's head D435i hangs off PC2
  with no DDS image topic; the de facto pattern is the
  xr_teleoperate/lerobot image server (ZMQ, JPEG frames). The embodiment
  takes an injected `camera_reader` (house pattern) plus a builtin ZMQ
  JPEG reader configured by `cam_server_address` (default
  `tcp://192.168.123.164:5556`; NOT 5555, to avoid colliding with the
  GR00T policy port, and both are config). Lazy imports: `zmq`, `cv2`
  (JPEG decode).
- **Policy: `gr00t`** speaking the Isaac-GR00T PolicyServer wire protocol
  (ZMQ REQ/REP, msgpack-numpy payloads, default port 5555), implemented
  directly with lazily imported `pyzmq` + `msgpack` + `msgpack_numpy`
  (no Isaac-GR00T dependency). Verified wire facts (hardcode; cite
  server_client.py / gr00t_policy.py at a pinned ref):
  - Request envelope `{"endpoint": "get_action", "data": {"observation":
    ..., "options": ...}}`; reply is an `(action, info)` LIST; action
    arrays are `(B, T, D)`.
  - Observations are NESTED dicts `{"video": {...}, "state": {...},
    "language": {...}}` with strict asserts: video `(B,T,H,W,C)` uint8,
    state `(B,T,D)` float32, language key
    `annotation.human.task_description`. The adapter batches as
    `(1, 1, ...)` and casts dtypes.
  - The G1 configs require state keys `left_leg, right_leg, waist,
    left_arm, right_arm, left_hand, right_hand`, and the arm-relative
    config RETURNS extra action keys (`waist`, `base_height_command`,
    `navigate_command`): extra returned keys are ignored by
    construction.
  - **Relative-vs-absolute (safety-critical, verified):** the current
    N1.7 PolicyServer converts relative arm actions to absolute
    SERVER-SIDE (`decode_action` undoes normalization and relativity),
    so wire actions from a stock server are already absolute radians.
    Default `actions_are_relative=False` (pass-through). The True path
    exists only for older/custom servers that return anchor-relative
    offsets, and its math is ANCHOR-ADD PER ELEMENT (`obs_arm + rel[i]`
    for each chunk row independently), NEVER cumulative summation:
    GR00T relativity is anchored to the last state timestep per row,
    and cumsum would compound offsets into runaway targets. A
    hand-computed anchor-add test locks this. The README carries a
    prominent warning: verify which convention your server version
    emits, slow-jog first, e-stop staffed.
  - Wire templating: the obs builder maps each required wire state key
    to a source: a packed-16D slice (`left_arm`, `right_arm`), a scalar
    hand expansion, a LOWSTATE PASSTHROUGH (`left_leg`, `right_leg`,
    `waist`: the embodiment already subscribes rt/lowstate and exposes
    these as additional StateFields, which conformance permits since
    exactly one field is (16,)), or a constant fill. Scalar-to-N-D hand
    expansion uses packing.py's open/closed reference poses (Dex3) or
    the 1-D stroke map (Dex1). UnifoLM-VLA and lerobot-served policies
    are future backends (README mentions them; no code in v1).
- **Out of scope v1**: locomotion/waist/neck control, 23-DOF arm
  support, per-joint hand actions, UnifoLM-VLA client, lerobot backend,
  GR00T-SONIC latent-action mode, sim backends.

## The 16-D contract

- `DIM_LABELS = ("left_j1".."left_j7", "left_gripper",
  "right_j1".."right_j7", "right_gripper")` (blockwise left-then-right,
  yam/A2 convention); `ARM_DOF=7`, `ARM_WIDTH=8`, `TOTAL_DIM=16`,
  `GRIPPER_IDXS=(7, 15)`, `STATE_KEY="joint_pos"`, unit
  `"rad+normalized"`, single StateField shape `(16,)`.
- `ActionSemantics(control_mode="joint_pos", rotation_repr="none",
  gripper="continuous", frame="base", dim_labels=DIM_LABELS)`. Arms are
  absolute radians (stock servers emit absolute actions; the optional
  anchor-add path converts older servers' relative offsets before
  emitting, so declared semantics stay honest either way). Grippers normalized, 1 = open.
- Default joint limits: G1 arm limits transcribed from the official
  URDF/datasheet by the implementer (cited in config.py), pulled 0.05 rad
  inward (franka margin pattern); gripper slots [0, 1].
- Default home pose: arms hanging relaxed pose from the arm_sdk example's
  release target (transcribed at implementation, cited), grippers open.
- Cameras: `head_cam` (the D435i color stream via the ZMQ reader; policy
  maps it to GR00T's `ego_view`).
- `control_hz=10.0` with **intra-step streaming** at `stream_hz=50.0`
  (arm_sdk's documented consumer rate): `step()` publishes
  `n = ceil(stream_hz/control_hz)` linearly interpolated micro-commands
  per step, anchored at the LAST PUBLISHED command (first step: the
  post-homing seed; the baseline is re-seeded from the observed pose at
  every connect/reset),
  each delta-capped at `max_joint_speed/stream_hz` (default
  `max_joint_speed=3.0` rad/s, deliberately far under xr_teleoperate's
  20 rad/s clip), paced by injected clock/sleep, kp/kd from config
  (defaults 60/1.5 arms per the official 50 Hz example; a lower wrist
  gain option exists as config, marked as bench-tuned defaults, not
  citations: the official example uses uniform gains and xr_teleoperate
  uses 80/3 + 40/1.5 at 250 Hz, so our combination is our own
  conservative choice). Same synchronous-interpolation
  design as the A2 plan: no background threads, fully fake-testable.
- Policy `control_hz=None`; embodiment declares `SELF_PACED`.

## Safety (the load-bearing section for this robot)

- **Weight ramp choreography** (all in the embodiment, tested):
  - Connect (first `reset()`): read current arm pose from lowstate,
    seed all commanded targets with it, then ramp `motor_cmd[29].q`
    0 -> 1 over `weight_ramp_s` (default 2.0) while holding the observed
    pose. Only then ramp to `home_pose` through the streaming path.
  - `close()`: ramp arms back to the pose observed at close start OR
    `rest_pose` if set (arm-only semantics; hands opened first), then
    ramp weight 1 -> 0 over `weight_ramp_s`, then disconnect. Idempotent;
    disconnect always attempted; handle cleared.
  - Unitree documents that an abrupt weight flip with target/actual
    mismatch snaps the arm at high speed: the ramp is not optional and
    the README says so.
- **Crash backstop**: there is no documented firmware watchdog on
  arm_sdk; a dead publisher with weight=1 leaves the controller holding
  the last command. The embodiment registers an `atexit` hook AND a
  SIGTERM handler (injected `atexit_module` and `signal_module` seams
  for tests; the previous SIGTERM handler is chained and restored on
  clean close) on connect that run the close choreography, and
  unregisters both on clean close. atexit alone does not cover SIGTERM,
  and SIGKILL is uncoverable: the README documents that residual gap
  next to the no-watchdog warning. README additionally
  documents the operator-level e-stop (remote L1+A = damping mode; the
  robot goes limp and sinks) and requires an operator in reach for every
  attended run.
- **Hard clamps**: absolute joint-limit clip + per-micro-command delta
  cap inside `step()`, independent of any Approver (house invariant).
- **First-run verification**: README requires the preflight plus a
  slow-jog check, and warns that `actions_are_relative` mismatches
  cannot be detected by compat (same warning pattern as franka's
  velocity flag).

## Package layout

```
inspect-robots-unitree-g1/
├── src/inspect_robots_unitree_g1/
│   ├── __init__.py / CLAUDE.md / py.typed
│   ├── packing.py         # 16-D constants, validate_dim, pack/split, hand scalar maps
│   ├── config.py          # G1Config, Gr00tConfig, shared space builders
│   ├── embodiment.py      # G1Embodiment + ArmDriver/HandDriver protocols + weight choreography
│   ├── policy.py          # Gr00tPolicy (ZMQ msgpack client + delta integration)
│   ├── operator.py        # OperatorIO (yam's EOF-hardened version)
│   ├── preflight.py       # inspect-robots-unitree-g1-preflight CLI (incl. dry_run in --json)
│   └── _unitree.py        # lazy sdk loader + install commands + cyclonedds guidance
├── tests/                 # franka battery + weight-choreography and atexit tests
├── plans/0001-g1-gr00t-design.md
├── .github/workflows/{ci,canary,release}.yml
├── .pre-commit-config.yaml / .env.example / CITATION.cff
├── pyproject.toml / uv.lock / README.md / CLAUDE.md / LICENSE / .gitignore
```

## Module contracts

### packing.py (pure)

Constants above; strict `validate_dim`; `pack/split`; `arm_slots(vec) ->
(14,)` in SDK index order (left 15-21 then right 22-28);
`dex1_scalar_to_stroke(wire) -> float` (1=open -> 0.0 rad, 0=closed ->
5.4 rad stroke default, config-scaled) and inverse;
`dex3_scalar_to_joints(wire, open_pose, closed_pose) -> (7,)` linear
interpolation and inverse (mean closure). All conversions tested with
asymmetric values.

### config.py

- House `_FromKwargs` + `_FLOAT_TUPLE_FIELDS`.
- `G1Config` (frozen): `iface="eth0"` (DDS network interface; None is
  allowed for default-route setups), `hand_type` in {"dex1", "dex3"}
  (default "dex3"), `arm_dof=7` (only 7 accepted; clear error otherwise),
  `control_hz=10.0`, `stream_hz=50.0` (integer multiple, validated),
  `max_joint_speed=3.0`, `weight_ramp_s=2.0`, `kp_arm=60.0`,
  `kd_arm=1.5`, `kp_wrist`, `kd_wrist` (cited defaults),
  `joint_low/joint_high`, `home_pose`, `rest_pose=None`,
  `dex1_stroke=5.4`, `dex1_max_speed=2.7` (rad/s), `hand_kp/hand_kd`,
  `cam_server_address="tcp://192.168.123.164:5556"`, `cam_timeout_s=5.0`,
  `unattended=False`, `docs_extra=""`. Post-init validation throughout.
- `Gr00tConfig` (frozen): `host="127.0.0.1"`, `port=5555`,
  `timeout_s=15.0` (REQ/REP with poller so a dead server raises instead
  of blocking forever), `actions_are_relative=True`,
  `action_horizon=16` (GR00T chunks are long, 16-50; default matches the
  executed prefix practice), `replan_interval=8`, `name="gr00t"`,
  `image_key="video.ego_view"`, `state_keys` template mapping our
  packed slices to `state.left_arm/right_arm/left_hand/right_hand`, and
  `action_keys` template for the reverse (modality-config alignment is
  config, not code). Explicit `PolicyConfig` wiring; nothing secret in
  asdict (no api_key equivalent here, but the test asserts the config
  class stays out of policy.config anyway).
- Shared builders: `ACTION_SEMANTICS`, `action_box()`,
  `observation_space()`.

### embodiment.py

- `ArmDriver` Protocol (injected via `arm_driver_factory`):
  `read_arm_joints() -> (14,)`, `publish_arm(q14, weight: float, kp, kd)
  -> None` (weight stamped every publish), `disconnect()`.
- `HandDriver` Protocol (injected via `hand_driver_factory`):
  `read_closure() -> (2,)` normalized, `publish_closure(left: float,
  right: float) -> None`, `disconnect()`.
- Defaults (pragma'd) built through `_unitree.py`: arm via
  rt/arm_sdk LowCmd_ with CRC; dex1/dex3 per `hand_type`; both raise the
  guided install error when the SDK is absent.
- `camera_reader` seam + builtin ZMQ JPEG reader (lazy zmq/cv2; timeout
  -> helpful fault).
- `G1Embodiment`: inert init (config + all seams + injected
  `atexit_module`); lazy connect at first reset with the weight-ramp
  choreography above; `step()` = validate -> clamp -> interpolated
  micro-publishes with delta cap and constant weight 1 -> hand closure
  publish (deadband `hand_deadband=0.05` on-change gating, franka
  gripper-gating pattern) -> pace -> observe (arm joints + hand closure
  packed 16-D; head_cam image) -> poll_end/confirm ->
  StepResult; `close()` = choreography; RUNTIME_REQUIREMENTS Mapping
  (`unitree_sdk2py`, `zmq`, `cv2`, `msgpack` with install remedies);
  DEVICE_SLOTS none (network robot; README documents); bind_task; docs
  with all 16 labels + docs_extra.

### policy.py

- `Gr00tPolicy(config=None, *, infer_fn=None, clock=None, **flat)`,
  entry point `gr00t`. `act()`: require `head_cam` + `joint_pos` state
  -> build the NESTED wire obs (video/state/language groups, (1,1,...)
  batching, dtype casts, filler/lowstate keys per the template) ->
  `infer_fn(obs) -> mapping of (B,T,D) action arrays` (envelope and
  (action, info) unwrap live in the transport) -> select our action
  keys, IGNORE extra returned keys -> reassemble into (N, 16) ->
  validate shapes/finiteness -> when `actions_are_relative=True` (NOT
  the default), anchor-add per element (`obs_arm + rel[i]`, no cumsum;
  hand slots never touched) -> truncate to `action_horizon` ->
  `ActionChunk(control_hz from shared config, latency measured)`.
- `_default_infer` (pragma'd transport shell): pyzmq REQ socket with
  RCVTIMEO/SNDTIMEO (upstream's own mechanism, not a poller);
  msgpack-numpy encode/decode; connect to `tcp://host:port`. On timeout
  the REQ state machine is permanently stuck: the wrapper MUST
  `close(linger=0)`, recreate, and reconnect the socket before
  re-raising (upstream does exactly this). The timeout/recreate wrapper
  is structured as a testable pure class over an injected socket
  factory, so a fake socket exercises the recreate path outside the
  pragma. Guided error listing the lazy deps if missing.
- `gr00t-seam` CI job: fetches Isaac-GR00T's `server_client.py` and
  `gr00t_policy.py` raw at a PINNED ref and asserts the load-bearing
  protocol facts still hold (the `{"endpoint", "data"}` envelope
  literal, `get_action` endpoint name, MsgSerializer packb/unpackb call
  shape, the `(action, info)` reply structure, the observation
  top-level keys), plus installs `pyzmq msgpack msgpack-numpy` and
  imports them. Asserting only pyzmq/msgpack APIs would validate
  nothing about the server contract; the raw-source assertions are what
  make drift fail loudly. Bumping the pinned ref is the deliberate
  drift-review moment.

### operator.py / preflight.py / _unitree.py

House patterns; preflight JSON includes `dry_run`; `_unitree.py` carries
the git install command, the cyclonedds pin/build guidance, and the L2
adjacency note in its error text.

### __init__.py public API (pinned)

`__all__` = `G1Config`, `Gr00tConfig`, `G1Embodiment`, `Gr00tPolicy`,
`OperatorIO`, `STATE_KEY`, `TOTAL_DIM`, `DIM_LABELS`, `build`,
`run_preflight`, `__version__`.

## pyproject

- Base deps: `inspect-robots>=0.12`, `numpy>=1.24` only (zmq/msgpack/
  msgpack-numpy/cv2 all lazy, guided; keeps import-hygiene strict).
- dev extra: house set. No hardware extra (git-only SDK).
- Entry points: embodiment `g1_arms`; policy `gr00t`. Console script
  `inspect-robots-unitree-g1-preflight`.
- mypy overrides: `unitree_sdk2py.*`, `zmq.*`, `msgpack.*`,
  `msgpack_numpy.*`, `cv2.*`.
- Everything else identical to franka.

## CI

`quality`, `test` matrix, `import-hygiene` (--no-deps + locked pins;
assert `zmq`, `msgpack`, `msgpack_numpy`, `cv2`, `unitree_sdk2py`,
`torch` absent), `gr00t-seam` (above), `ci-ok` needing all four,
`alert-red-main`; canary/release byte-copied. Ruleset already active.

## Test plan (franka battery plus robot-specific)

- test_packing.py: constants/labels/validate_dim/pack/split; dex1 and
  dex3 scalar conversions with asymmetric values and clipping.
- test_config.py: from_kwargs rejection; tuple parsing; validation
  (stream ratio, arm_dof rejection message, hand_type validation,
  pose-in-limits).
- test_embodiment.py: inert init; connect choreography ORDER (seed with
  observed pose -> weight ramp up -> home ramp; hand-computed weight
  values per publish); step interpolation micro-targets anchored at the
  last published command and delta cap (hand-computed); baseline reseed
  at reset; constant weight 1 during steps; hand deadband gating; dex1
  rad/s rate limit derived per publish; clamp backstop; pacing;
  observation packing (asymmetric closure) incl. the extra
  leg/waist StateFields from lowstate; operator success/failure;
  unattended; close choreography ORDER (hands open -> arm ramp ->
  weight ramp down -> disconnect) + idempotency + disconnect-on-error;
  atexit AND SIGTERM handlers registered on connect and
  unregistered/restored on close (injected fakes); bind_task; docs;
  RUNTIME_REQUIREMENTS via conformance.
- test_policy.py: nested wire obs structure (video/state/language
  groups, batching dims, dtype casts, language key literal); state
  slice mapping + filler/lowstate keys (asymmetric); anchor-add
  relative path (hand-computed per-element expectations, hands
  untouched) and the DEFAULT pass-through path; extra returned action
  keys ignored; reassembly; truncation; PolicyConfig wiring; the
  REQ-recreate wrapper exercised with a fake socket factory (timeout ->
  close(linger=0) -> recreate -> raise); num_inferences; helpful
  errors.
- test_operator.py / test_preflight.py / test_unitree.py (loader
  messages: git URL, cyclonedds guidance).
- test_compat.py: zero/zero; cubepick-reach realizable; negatives.
- test_embodiment_docs.py / test_api_snapshot.py /
  test_eval_end_to_end.py (fake drivers + fake infer_fn through eval()).

## README (yam structure; house style)

Sections: badges/intro, Install (adapter machine on the robot subnet:
package + SDK git install + cyclonedds notes; PC2: image server pointer;
GPU machine: Isaac-GR00T inference server command), Preflight, Run on
hardware (config.ini; L2 adjacency; port-collision note 5555 policy vs
camera), Safety (weight ramp choreography, no-watchdog crash backstop +
atexit/SIGTERM and the SIGKILL gap, L1+A damping e-stop and what it
physically does, relative-flag warning incl. server-side conversion,
control_hz-must-match-checkpoint-fps note with the same prominence,
first-run slow jog + bench-verify dex polarity), Configuration (field tables; 16-D unit
table; hand collapse semantics), Development, Citation, License.

## Sequencing

1. Critique loop until clean.
2. Codex implements on feat/g1-plugin; `uv lock` before first push;
   Fable reviews the diff.
3. Push; PR (Closes #1) green; fresh-eyes review loop; merge.
4. Post-merge: PyPI pending publisher (owner action).
