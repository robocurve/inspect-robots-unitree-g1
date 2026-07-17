import numpy as np
from inspect_robots import eval as robots_eval

from conftest import FakeArm, FakeAtexit, FakeHand, FakeSignal
from inspect_robots_unitree_g1.config import G1Config, Gr00tConfig
from inspect_robots_unitree_g1.embodiment import G1Embodiment
from inspect_robots_unitree_g1.operator import OperatorIO
from inspect_robots_unitree_g1.policy import Gr00tPolicy


def test_full_eval_propagates_success_and_metadata() -> None:
    arm, hand = FakeArm(), FakeHand()
    response = {
        "left_arm": np.zeros((1, 1, 7)),
        "right_arm": np.zeros((1, 1, 7)),
        "left_hand": np.ones((1, 1, 1)),
        "right_hand": np.ones((1, 1, 1)),
    }
    policy = Gr00tPolicy(
        Gr00tConfig(action_horizon=1), infer_fn=lambda _: response, clock=lambda: 0
    )
    embodiment = G1Embodiment(
        G1Config(weight_ramp_s=0.02, max_joint_speed=100),
        arm_driver_factory=lambda _: arm,
        hand_driver_factory=lambda _: hand,
        camera_reader=lambda: np.zeros((2, 2, 3), dtype=np.uint8),
        operator=OperatorIO(input_fn=lambda prompt: "yes" if "succeed" in prompt else ""),
        poll_end=lambda: True,
        clock=lambda: 0,
        sleep_fn=lambda _: None,
        atexit_module=FakeAtexit(),
        signal_module=FakeSignal(),
    )
    logs = robots_eval("cubepick-reach", policy, embodiment, sinks=[], seed=0)
    assert logs[0].status == "success"
    assert logs[0].results.metrics["success_at_end"] == 1.0
    assert logs[0].eval.policy_config["action_horizon"] == 1
    assert embodiment.num_steps == 1
