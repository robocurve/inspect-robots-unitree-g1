# inspect-robots-unitree-g1 agent guide

Inspect Robots adapters for Unitree G1 arms controlled through arm_sdk and
Isaac-GR00T policy servers, plus walking embodiments (`g1_base`,
`g1_mobile`) for LLM agents in `walking/`.

## The one big idea

The `g1_arms` embodiment and `gr00t` policy share one absolute 16-D
`joint_pos` contract. It contains seven left arm radians, one normalized left
hand, seven right arm radians, and one normalized right hand.

## Layout

- `src/inspect_robots_unitree_g1/` contains the package.
- `tests/` contains injected hardware-free tests.
- `plans/0001-g1-gr00t-design.md` is the accepted binding design.
- `src/inspect_robots_unitree_g1/walking/` holds `g1_base` and `g1_mobile`
  (design in `plans/0002-g1-walking-design.md`). It shares no state with
  `g1_arms`; tests are `tests/test_walking_*.py` with fakes in
  `tests/walking_fakes.py`.

## Working here

- Set `UV_CACHE_DIR=$PWD/.uv-cache` for uv commands.
- Run ruff check, ruff format check, strict mypy, and pytest with coverage.
- Keep optional SDK, ZMQ, msgpack, and OpenCV imports lazy.
- Keep 100 percent statement and branch coverage.

## Safety invariants

- Weight is blended in while holding the measured pose before homing.
- Every reset reseeds the command baseline from measured arm joints.
- Every action is hard-clamped and speed-limited in the embodiment.
- Close opens hands, parks arms, blends weight to zero, then disconnects.
- Construction performs no hardware, network, camera, signal, or stdin work.
- Success reaches scoring only through `termination_reason="success"`.
- Walking: every velocity command carries an on-robot hold, a streamer stands
  the robot once a step's deadline passes, and the driver never changes the
  FSM mode. `g1_mobile` commands arms only between reset's homing and close's
  release, ramps the arm_sdk weight in and out, and rate-limits every joint.
- The read-only robot check (`walking/preflight.py`) must never create a
  publisher.

## CI and releases

- CI installs from `uv.lock`.
- `ci-ok` needs every blocking job.
- Versions come from git tags through hatch-vcs.

## Writing style

- Do not use em dashes in prose.
- Do not use decorative emoji.
- Use plain headers without trailing colons.
