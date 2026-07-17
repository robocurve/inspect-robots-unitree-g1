# inspect-robots-unitree-g1 agent guide

Inspect Robots adapters for Unitree G1 arms controlled through arm_sdk and
Isaac-GR00T policy servers.

## The one big idea

The `g1_arms` embodiment and `gr00t` policy share one absolute 16-D
`joint_pos` contract. It contains seven left arm radians, one normalized left
hand, seven right arm radians, and one normalized right hand.

## Layout

- `src/inspect_robots_unitree_g1/` contains the package.
- `tests/` contains injected hardware-free tests.
- `plans/0001-g1-gr00t-design.md` is the accepted binding design.

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

## CI and releases

- CI installs from `uv.lock`.
- `ci-ok` needs every blocking job.
- Versions come from git tags through hatch-vcs.

## Writing style

- Do not use em dashes in prose.
- Do not use decorative emoji.
- Use plain headers without trailing colons.
