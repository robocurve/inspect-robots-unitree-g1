from inspect_robots.conformance import (
    assert_embodiment_conformant,
    device_slots,
    missing_runtime_requirements,
)

from inspect_robots_unitree_g1.embodiment import G1Embodiment
from inspect_robots_unitree_g1.policy import Gr00tPolicy


def test_declarations_and_runtime_requirements() -> None:
    assert_embodiment_conformant(G1Embodiment().info)
    assert device_slots(G1Embodiment) == ()
    assert set(G1Embodiment.RUNTIME_REQUIREMENTS) == {"unitree_sdk2py", "zmq", "cv2"}
    assert set(Gr00tPolicy.RUNTIME_REQUIREMENTS) == {"zmq", "msgpack", "msgpack_numpy"}
    assert "unitree_sdk2py" in missing_runtime_requirements(G1Embodiment)
