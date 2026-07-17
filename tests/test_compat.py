from typing import Any

from inspect_robots.compat import check_compatibility
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.spaces import ActionSemantics, Box

from inspect_robots_unitree_g1.config import action_box, observation_space
from inspect_robots_unitree_g1.embodiment import G1Embodiment
from inspect_robots_unitree_g1.policy import Gr00tPolicy


class Policy:
    config = PolicyConfig()

    def __init__(self, info: PolicyInfo) -> None:
        self.info = info

    def reset(self, scene: object) -> None:
        return None

    def act(self, observation: object) -> Any:
        raise AssertionError


def test_pair_is_zero_zero_and_state_keys_are_identical() -> None:
    policy, embodiment = Gr00tPolicy(), G1Embodiment()
    report = check_compatibility(policy, embodiment)
    assert report.errors == []
    assert report.warnings == []
    assert policy.info.observation_space.state_keys == embodiment.info.observation_space.state_keys


def test_builtin_task_is_realizable() -> None:
    task = resolve("task", "cubepick-reach")
    assert check_compatibility(Gr00tPolicy(), G1Embodiment(), task).errors == []


def test_negative_dimension_rate_and_control_mode() -> None:
    wrong = PolicyInfo(
        name="wrong", action_space=Box(shape=(7,), semantics=ActionSemantics("joint_pos"))
    )
    assert any(
        issue.code == "action_dim"
        for issue in check_compatibility(Policy(wrong), G1Embodiment()).errors  # type: ignore[arg-type]
    )
    rated = PolicyInfo(
        name="rated",
        action_space=action_box(),
        observation_space=observation_space(),
        control_hz=30,
    )
    assert [
        issue.code
        for issue in check_compatibility(Policy(rated), G1Embodiment()).warnings  # type: ignore[arg-type]
    ] == ["control_rate"]
    velocity = PolicyInfo(
        name="velocity",
        action_space=Box(
            shape=(16,),
            semantics=ActionSemantics(
                "joint_vel",
                gripper="continuous",
                frame="base",
                dim_labels=tuple(map(str, range(16))),
            ),
        ),
        observation_space=observation_space(),
    )
    assert any(
        issue.code == "control_mode"
        for issue in check_compatibility(Policy(velocity), G1Embodiment()).errors  # type: ignore[arg-type]
    )
