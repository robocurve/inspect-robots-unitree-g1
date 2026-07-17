import dataclasses

import numpy as np
import pytest
from inspect_robots.errors import ConfigError

from inspect_robots_unitree_g1.config import (
    DEFAULT_ACTION_KEYS,
    DEFAULT_HOME_POSE,
    DEFAULT_STATE_KEYS,
    G1Config,
    Gr00tConfig,
    action_box,
    observation_space,
)


def test_defaults_and_shared_spaces() -> None:
    cfg = G1Config()
    assert cfg.hand_kp == 1.5
    assert cfg.hand_kd == 0.2
    assert DEFAULT_HOME_POSE[7] == DEFAULT_HOME_POSE[15] == 1.0
    assert np.all(cfg.low < cfg.high)
    box = action_box(cfg)
    assert box.shape == (16,)
    assert box.semantics is not None and box.semantics.dim_labels is not None
    obs = observation_space()
    assert obs.camera_names == {"head_cam"}
    assert obs.state_keys == {"joint_pos", "left_leg", "right_leg", "waist"}


def test_from_kwargs_tuple_parsing_and_unknown() -> None:
    home = ",".join(str(value) for value in DEFAULT_HOME_POSE)
    assert G1Config.from_kwargs(home_pose=home).home_pose == DEFAULT_HOME_POSE
    assert G1Config.from_kwargs(home_pose=DEFAULT_HOME_POSE).home_pose == DEFAULT_HOME_POSE
    assert (
        G1Config.from_kwargs(home_pose=home, rest_pose=DEFAULT_HOME_POSE).rest_pose
        == DEFAULT_HOME_POSE
    )
    with pytest.raises(TypeError, match="unexpected"):
        G1Config.from_kwargs(nope=1)
    with pytest.raises(ValueError, match="comma-separated"):
        G1Config.from_kwargs(home_pose="bad")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"hand_type": "nope"}, "hand_type"),
        ({"control_hz": 0.0}, "control_hz"),
        ({"stream_hz": 45.0}, "integer multiple"),
        ({"kp_arm": -1.0}, "kp_arm"),
        ({"joint_low": (0.0,)}, "16 entries"),
        ({"rest_pose": (0.0,)}, "16 entries"),
        ({"joint_low": (float("nan"),) * 16}, "finite"),
        ({"joint_low": (2.0,) * 16, "joint_high": (1.0,) * 16}, "below"),
        ({"home_pose": (99.0,) * 16}, "inside"),
        ({"dex1_stroke": 0.0}, "dex1_stroke"),
        ({"hand_deadband": 1.0}, "hand_deadband"),
        ({"dex3_thumb_swing": 2.0}, "thumb_swing"),
        ({"hand_kp": -1.0}, "hand_kp"),
        ({"cam_server_address": ""}, "must not be empty"),
    ],
)
def test_g1_validation(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        G1Config(**changes)  # type: ignore[arg-type]


def test_arm_dof_uses_framework_config_error() -> None:
    with pytest.raises(ConfigError, match="23-DOF"):
        G1Config(arm_dof=5)


def test_dex1_gain_defaults_and_rest_validation() -> None:
    cfg = G1Config(hand_type="dex1")
    assert (cfg.hand_kp, cfg.hand_kd) == (5.0, 0.05)
    assert G1Config(hand_kp=2.0, hand_kd=3.0).hand_kd == 3.0
    with pytest.raises(ValueError, match="rest_pose"):
        G1Config(rest_pose=tuple([0.0] * 7 + [2.0] + [0.0] * 8))


def test_gr00t_defaults_are_independent_and_exact() -> None:
    left = Gr00tConfig()
    right = Gr00tConfig()
    assert left.actions_are_relative is False
    assert left.state_keys == DEFAULT_STATE_KEYS
    assert left.action_keys == DEFAULT_ACTION_KEYS
    assert left.state_keys is not right.state_keys
    assert dataclasses.asdict(left)["host"] == "127.0.0.1"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"host": ""}, "host"),
        ({"port": True}, "port"),
        ({"timeout_s": 0.0}, "timeout"),
        ({"action_horizon": 0}, "action_horizon"),
        ({"name": ""}, "image_key"),
        ({"image_key": "ego"}, "image_key"),
        ({"image_key": "images.ego"}, "video"),
        ({"hand_type": "bad"}, "hand_type"),
        ({"dex1_stroke": 0.0}, "dex1_stroke"),
        ({"dex3_thumb_swing": 2.0}, "thumb_swing"),
        ({"control_hz": 0.0}, "control_hz"),
        ({"state_keys": {"bad": ("packed", 0, 7)}}, "invalid source"),
        ({"state_keys": {"state.x": ("bad", 0, 1)}}, "invalid source tag"),
        ({"state_keys": {"state.x": ("packed", 7, 7)}}, "invalid slice"),
        ({"state_keys": {"state.x": ("packed", 1)}}, "must be"),
        ({"state_keys": {"state.x": ("hand", "middle")}}, "left or right"),
        ({"state_keys": {"state.x": ("state", "")}}, "must name"),
        ({"state_keys": {"state.x": ("state", "other")}}, "not declared"),
        ({"state_keys": {"state.x": ("const", 0, 1.0)}}, "positive_dim"),
        ({"state_keys": {"other.x": ("const", 1, 0.0)}}, "state.*group"),
        ({"action_keys": {"action.x": ("state", "waist")}}, "invalid source tag"),
        ({"action_keys": {"other.x": ("hand", "left")}}, "action.*group"),
    ],
)
def test_gr00t_validation(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Gr00tConfig(**changes)  # type: ignore[arg-type]
