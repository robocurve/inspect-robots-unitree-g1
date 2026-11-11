"""The G1 walking embodiment: config, contract, locomotion driver, and a full eval."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots import eval as robots_eval
from inspect_robots.conformance import assert_embodiment_conformant, check_evaluation_interface
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.scene import Scene
from inspect_robots.scorer import episode_length
from inspect_robots.task import Task
from inspect_robots.tools import control_tools
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_unitree_g1.walking.config import (
    DIM_LABELS,
    G1Config,
    LeggedConfig,
    action_box,
    observation_space,
)
from inspect_robots_unitree_g1.walking.driver import KinematicLeggedDriver
from inspect_robots_unitree_g1.walking.embodiment import G1Embodiment
from inspect_robots_unitree_g1.walking.g1 import G1LocoDriver
from walking_fakes import FakeAtexit, FakeClock, FakeSignal, ManualThread, Rig

# ---------------------------------------------------------------------- config


def test_g1_defaults_are_gentler_than_the_shared_base() -> None:
    cfg = G1Config()
    np.testing.assert_allclose(cfg.high, (0.03, 0.02, 0.05))
    np.testing.assert_allclose(cfg.low, (-0.02, -0.02, -0.05))
    assert cfg.stream_hz == 20.0
    assert cfg.camera_name == "head_cam"
    assert cfg.isaac_task == "Isaac-Velocity-Flat-G1-Play-v0"
    assert G1Config.ISAAC_CAMERA_LINK == "torso_link"


def test_g1_from_kwargs_knows_its_own_fields_only() -> None:
    cfg = G1Config.from_kwargs(network_interface="enp3s0", velocity_hold_s="0.5")
    assert cfg.network_interface == "enp3s0"
    assert cfg.velocity_hold_s == 0.5
    with pytest.raises(TypeError, match="unknown G1Config option"):
        G1Config.from_kwargs(robot_ip="10.0.0.2")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"velocity_hold_s": 0.0}, "velocity_hold_s must be finite"),
        ({"rpc_timeout_s": -1.0}, "rpc_timeout_s must be finite"),
        ({"velocity_hold_s": 0.05}, "at least two stream periods"),
        ({"network_interface": " "}, "network_interface must be non-empty"),
        ({"odom_topic": ""}, "odom_topic must be non-empty"),
        ({"max_forward_speed": 0.0}, "max_forward_speed must be finite"),
    ],
)
def test_invalid_g1_config_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        G1Config(**kwargs)


# ---------------------------------------------------------------------- contract


def test_g1_contract_conformant_and_agent_ready() -> None:
    emb = G1Embodiment()
    assert_embodiment_conformant(emb.info)
    assert check_evaluation_interface(emb).ok
    assert emb.info.name == "g1_base"
    assert emb.info.morphology == "unitree-g1"
    assert emb.info.environment_id == "g1:sdk"
    assert emb.info.is_simulated is False
    tools = control_tools(emb.info.action_space, emb.info.observation_space, emb.info.control_hz)
    assert tools.library.names == ("move_by", "sleep")
    assert "base_forward: [-0.02, 0.03]" in tools.bounds_text
    docs = emb.info.docs or ""
    assert "humanoid" in docs
    bullets = [line for line in docs.splitlines() if line.startswith("- ")]
    for label in DIM_LABELS:
        assert sum(line.startswith(f"- {label}:") for line in bullets) == 1


def test_g1_entry_point_and_flat_kwargs() -> None:
    emb = resolve("embodiment", "g1_base", backend="kinematic", max_turn_rate="0.3")
    assert isinstance(emb, G1Embodiment)
    assert emb.info.environment_id == "g1:kinematic"
    assert emb.info.is_simulated is True
    assert np.asarray(emb.info.action_space.high)[2] == pytest.approx(0.03)


def test_wrong_config_type_rejected() -> None:
    with pytest.raises(TypeError, match="G1Embodiment needs a G1Config; got LeggedConfig"):
        G1Embodiment(LeggedConfig())


def test_g1_ready_prompt_names_walking_mode() -> None:
    rig = Rig(G1Config(), embodiment_cls=G1Embodiment)
    rig.reset()
    assert "walking mode" in rig.prompts[0]
    rig.emb.close()


# ---------------------------------------------------------------------- driver


class FakeLoco:
    def __init__(self, fsm: Any = 500, fsm_code: int = 0) -> None:
        self.fsm = fsm
        self.fsm_code = fsm_code
        self.code = 0
        self.calls: list[tuple[float, float, float, float]] = []

    def SetVelocity(self, vx: float, vy: float, omega: float, duration: float = 1.0) -> int:
        self.calls.append((vx, vy, omega, duration))
        return self.code

    def GetFsmId(self) -> tuple[int, Any]:
        return self.fsm_code, self.fsm


def lowstate(yaw: float = 0.0) -> Any:
    return SimpleNamespace(imu_state=SimpleNamespace(rpy=[0.01, -0.02, yaw]))


def odom(x: float, y: float, yaw: float, vx: float, vy: float, wz: float) -> Any:
    return SimpleNamespace(
        position=[x, y, 0.75],
        velocity=[vx, vy, 0.0],
        yaw_speed=wz,
        imu_state=SimpleNamespace(rpy=[0.0, 0.0, yaw]),
    )


def make_driver(**kwargs: Any) -> tuple[G1LocoDriver, FakeLoco, FakeClock]:
    clock = FakeClock()
    loco = FakeLoco()
    driver = G1LocoDriver(G1Config(), loco=loco, clock=clock, **kwargs)
    return driver, loco, clock


def test_check_ready_accepts_walking_and_refuses_limp_or_failed() -> None:
    driver, loco, _ = make_driver()
    assert driver.check_ready() == 500
    assert driver.fsm_id == 500
    for limp in (0, 1):
        loco.fsm = limp
        with pytest.raises(RuntimeError, match="limp"):
            driver.check_ready()
    loco.fsm_code = 3104
    with pytest.raises(RuntimeError, match="GetFsmId failed with code 3104"):
        driver.check_ready()


def test_every_command_carries_the_on_robot_hold() -> None:
    driver, loco, _ = make_driver()
    driver.send_velocity(0.2, -0.1, 0.3)
    driver.stand()
    assert loco.calls == [(0.2, -0.1, 0.3, 0.3), (0.0, 0.0, 0.0, 0.3)]
    loco.code = 3102
    with pytest.raises(RuntimeError, match="SetVelocity failed with code 3102"):
        driver.send_velocity(0.1, 0.0, 0.0)


def test_state_needs_lowstate_first() -> None:
    driver, _, _ = make_driver()
    with pytest.raises(RuntimeError, match="rt/lowstate"):
        driver.read_state()


def test_fresh_odometry_wins_and_velocity_is_rotated_into_the_body_frame() -> None:
    driver, _, clock = make_driver()
    driver.on_lowstate(lowstate(yaw=0.4))
    # Facing +y in the world, moving +y at 0.2 m/s: that is forward in the body.
    driver.on_odom(odom(1.0, 2.0, math.pi / 2, 0.0, 0.2, 0.1))
    state = driver.read_state()
    np.testing.assert_allclose(state.pose, (1.0, 2.0, math.pi / 2))
    np.testing.assert_allclose(state.velocity, (0.2, 0.0, 0.1), atol=1e-12)
    np.testing.assert_allclose(state.rpy, (0.01, -0.02, 0.4))
    # Stale odometry falls back to dead reckoning from the last odometry pose.
    clock.now += 1.0
    driver.on_lowstate(lowstate(yaw=0.4))  # the IMU keeps arriving; only odometry stopped
    stale = driver.read_state()
    np.testing.assert_allclose(stale.velocity, 0.0)
    np.testing.assert_allclose(stale.pose[:2], (1.0, 2.0))


def test_dead_reckoning_integrates_commands_on_imu_yaw() -> None:
    driver, _, clock = make_driver()
    # Before any IMU sample, heading integrates the commanded turn.
    driver.send_velocity(0.0, 0.0, 1.0)
    clock.now += 0.5
    driver.send_velocity(0.2, 0.1, 0.0)
    driver.on_lowstate(lowstate(yaw=math.pi / 2))
    clock.now += 1.0
    driver.on_lowstate(lowstate(yaw=math.pi / 2))
    state = driver.read_state()
    # One second facing +y in the world: forward 0.2 -> +y, left 0.1 -> -x.
    np.testing.assert_allclose(state.pose, (-0.1, 0.2, math.pi / 2), atol=1e-12)
    np.testing.assert_allclose(state.velocity, (0.2, 0.1, 0.0))


def test_stale_lowstate_stops_the_robot() -> None:
    driver, _, clock = make_driver()
    driver.on_lowstate(lowstate())
    driver.read_state()
    clock.now += 1.0  # rt/lowstate stopped arriving
    with pytest.raises(RuntimeError, match=r"rt/lowstate is 1\.00 s old"):
        driver.read_state()


def test_camera_reader_is_required_for_frames() -> None:
    driver, _, _ = make_driver()
    with pytest.raises(RuntimeError, match="no G1 camera reader"):
        driver.read_frame()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with_cam, _, _ = make_driver(camera_reader=lambda: frame)
    assert with_cam.read_frame() is frame


def test_disconnect_stops_then_closes_everything_even_on_errors() -> None:
    closed: list[str] = []

    def failing() -> None:
        closed.append("a")
        raise OSError("close a")

    driver, loco, _ = make_driver(closers=(failing, lambda: closed.append("b")))
    loco.code = 1
    with pytest.raises(RuntimeError, match="SetVelocity failed"):
        driver.disconnect()
    assert closed == ["a", "b"]
    assert loco.calls[-1] == (0.0, 0.0, 0.0, 0.3)
    closed.clear()
    loco.code = 0
    with pytest.raises(OSError, match="close a"):
        driver.disconnect()
    assert closed == ["a", "b"]
    clean, _, _ = make_driver(closers=(lambda: closed.append("c"),))
    clean.disconnect()
    assert closed[-1] == "c"


# ---------------------------------------------------------------------- full eval


class WalkForward:
    def __init__(self) -> None:
        cfg = G1Config()
        self.info = PolicyInfo(
            name="walk-forward",
            action_space=action_box(cfg),
            observation_space=observation_space(cfg),
        )
        self.config = PolicyConfig(action_horizon=5)

    def reset(self, scene: Scene) -> None:
        return None

    def act(self, observation: Observation) -> ActionChunk:
        return ActionChunk(actions=[Action(data=np.asarray((0.03, 0.0, 0.0)))] * 5)


def test_eval_walks_the_kinematic_g1_forward(tmp_path: Path) -> None:
    cfg = G1Config(backend="kinematic", unattended=True, camera_height=48, camera_width=64)
    clock = FakeClock()
    driver = KinematicLeggedDriver(cfg, clock=clock, response_time_s=0.0)
    emb: G1Embodiment

    def sleep_while_streaming(duration: float) -> None:
        emb.stream_once()
        clock.sleep(duration)

    emb = G1Embodiment(
        cfg,
        driver_factory=lambda _: driver,
        clock=clock,
        sleep_fn=sleep_while_streaming,
        thread_factory=ManualThread,
        atexit_module=FakeAtexit(),
        signal_module=FakeSignal(),
    )
    task = Task(
        name="walk",
        scenes=[Scene(id="w", instruction="walk forward")],
        scorer=episode_length(),
        max_steps=20,
    )
    logs = robots_eval(task, WalkForward(), emb, sinks=[], seed=0, log_dir=str(tmp_path))
    assert logs[0].status == "success", logs[0].error
    # 20 steps at the 0.3 m/s cap for 0.1 s each is 0.6 m.
    np.testing.assert_allclose(driver.read_state().pose[:2], (0.6, 0.0), atol=1e-6)
    emb.close()
    assert driver.connected is False
