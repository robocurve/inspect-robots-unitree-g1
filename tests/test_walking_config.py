import numpy as np
import pytest

from inspect_robots_unitree_g1.walking.config import (
    DIM_LABELS,
    LeggedConfig,
    action_box,
    observation_space,
)


def test_defaults_give_per_step_bounds_from_speed_caps() -> None:
    cfg = LeggedConfig()
    np.testing.assert_allclose(cfg.high, (0.05, 0.03, 0.1))
    np.testing.assert_allclose(cfg.low, (-0.03, -0.03, -0.1))
    box = action_box()
    assert box.semantics is not None
    assert box.semantics.control_mode == "joint_delta"
    assert box.semantics.dim_labels == DIM_LABELS
    np.testing.assert_allclose(box.high, cfg.high)


def test_bounds_scale_with_control_rate() -> None:
    cfg = LeggedConfig(control_hz=5.0, max_forward_speed=1.0)
    assert cfg.high[0] == pytest.approx(0.2)


def test_observation_space_follows_camera_config() -> None:
    space = observation_space(LeggedConfig(camera_name="head", camera_height=10, camera_width=20))
    assert space.camera_names == frozenset({"head"})
    assert space.cameras[0].height == 10
    assert space.state_keys == frozenset({"base_pose", "base_vel", "body_rpy"})
    assert observation_space().camera_names == frozenset({"front_cam"})


def test_from_kwargs_coerces_cli_strings() -> None:
    cfg = LeggedConfig.from_kwargs(
        backend="kinematic",
        control_hz="20",
        unattended="yes",
        headless="off",
        geofence_m="none",
        camera_height="240",
        isaac_task="Isaac-Custom-v0",
        max_turn_rate=0.5,
    )
    assert cfg.backend == "kinematic"
    assert cfg.control_hz == 20.0
    assert cfg.unattended is True
    assert cfg.headless is False
    assert cfg.geofence_m is None
    assert cfg.camera_height == 240
    assert cfg.isaac_task == "Isaac-Custom-v0"
    assert cfg.max_turn_rate == 0.5
    assert LeggedConfig.from_kwargs(geofence_m="2.5").geofence_m == 2.5


def test_from_kwargs_rejects_unknown_and_bad_booleans() -> None:
    with pytest.raises(TypeError, match="unknown LeggedConfig option"):
        LeggedConfig.from_kwargs(speed=1)
    with pytest.raises(ValueError, match="must be a boolean"):
        LeggedConfig.from_kwargs(unattended="maybe")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "ros"}, "backend must be one of"),
        ({"control_hz": 0.0}, "control_hz must be finite"),
        ({"max_turn_rate": float("nan")}, "max_turn_rate must be finite"),
        ({"stream_hz": 5.0}, "stream_hz must be >= control_hz"),
        ({"command_timeout_s": -1.0}, "command_timeout_s"),
        ({"geofence_m": 0.0}, "geofence_m must be finite"),
        ({"camera_height": 0}, "camera_height"),
        ({"camera_name": " "}, "camera_name"),
    ],
)
def test_invalid_config_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LeggedConfig(**kwargs)  # type: ignore[arg-type]
