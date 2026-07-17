# inspect_robots_unitree_g1 module map

| Module | Responsibility |
|--------|----------------|
| `packing.py` | Pure 16-D packing and hand conversions. |
| `config.py` | Frozen configs and shared spaces. |
| `embodiment.py` | Weight choreography, streaming, camera, and operator logic. |
| `policy.py` | Nested GR00T wire adapter and reconnecting REQ transport. |
| `_unitree.py` | Guided loader for the git-only Unitree SDK. |
| `operator.py` | Injectable readiness and scoring prompts. |
| `preflight.py` | Hardware-free compatibility CLI. |
| `__init__.py` | Reviewed public API. |

## Invariants

- Construction is inert.
- The published baseline is measured again at every reset.
- Arm weight changes only through a streamed ramp.
- Relative GR00T rows are each anchored to the same observation.
- Hand slots are never changed by relative arm conversion.
