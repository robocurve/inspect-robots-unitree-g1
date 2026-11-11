"""g1_mobile: walking plus both hands in one trial."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots import eval as robots_eval
from inspect_robots.conformance import assert_embodiment_conformant, check_evaluation_interface
from inspect_robots.guard import DeltaLimitGuard
from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.registry import resolve
from inspect_robots.scene import Scene
from inspect_robots.scorer import episode_length
from inspect_robots.task import Task
from inspect_robots.tools import control_tools
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_unitree_g1.walking.arm_kinematics import G1_ARMS, fk
from inspect_robots_unitree_g1.walking.config import MOBILE_DIM_LABELS, G1Config, G1MobileConfig
from inspect_robots_unitree_g1.walking.g1 import (
    ARM_MOTOR_INDICES,
    G1MobileSdkDriver,
    dex1_stroke_to_grip,
    grip_to_dex1_stroke,
)
from inspect_robots_unitree_g1.walking.mobile import (
    HOME_HANDS,
    G1MobileEmbodiment,
    KinematicMobileDriver,
)
from walking_fakes import FakeAtexit, FakeClock, FakeSignal, ManualThread, Rig

# ---------------------------------------------------------------------- config


def test_mobile_bounds_and_grips() -> None:
    sim = G1MobileConfig(backend="kinematic")
    assert sim.high.shape == (11,)
    np.testing.assert_allclose(sim.high[3:9], 0.015)
    np.testing.assert_allclose(sim.high[9:], 0.2)
    np.testing.assert_allclose(sim.low[9:], -0.2)
    real = G1MobileConfig()
    assert real.grips_enabled is False
    np.testing.assert_allclose(real.high[9:], 0.0)
    assert G1MobileConfig(backend="kinematic", hands="none").grips_enabled is False
    low, high = sim.hand_box
    assert low[0, 1] == 0.05 and high[0, 1] == 0.38
    assert low[1, 1] == -0.38 and high[1, 1] == -0.05


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hand_speed": 0.0}, "hand_speed must be finite"),
        ({"hands": "inspire"}, "hands must be"),
        ({"tool_offset_m": 0.3}, "tool_offset_m"),
        ({"tool_offset_m": float("nan")}, "tool_offset_m"),
        ({"dex1_stroke": 0.0}, "dex1_stroke must be finite"),
        ({"hand_x_min": 0.5}, "min must be below max"),
        ({"hand_z_max": -0.1}, "min must be below max"),
        ({"hand_y_inner": 0.4}, "hand_y_inner < hand_y_outer"),
        ({"hand_y_inner": -0.1}, "hand_y_inner < hand_y_outer"),
    ],
)
def test_invalid_mobile_config_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        G1MobileConfig(**kwargs)


# ---------------------------------------------------------------------- contract


def test_mobile_contract_works_with_the_agent_and_guards() -> None:
    emb = G1MobileEmbodiment(backend="kinematic")
    assert_embodiment_conformant(emb.info)
    assert check_evaluation_interface(emb).ok
    assert emb.info.name == "g1_mobile"
    assert emb.info.environment_id == "g1-mobile:kinematic"
    box = emb.info.action_space
    assert box.semantics is not None and box.semantics.control_mode == "eef_delta_pos"
    assert box.semantics.dim_labels == MOBILE_DIM_LABELS
    assert emb.info.observation_space.state_keys >= {"hand_pos", "grip", "base_pose"}
    tools = control_tools(box, emb.info.observation_space, emb.info.control_hz)
    assert tools.library.names == ("move_by", "sleep")
    assert "left_hand_x: [-0.015, 0.015]" in tools.bounds_text
    DeltaLimitGuard(box)
    docs = emb.info.docs or ""
    bullets = [line for line in docs.splitlines() if line.startswith("- ")]
    for label in MOBILE_DIM_LABELS:
        assert sum(line.startswith(f"- {label}:") for line in bullets) == 1


def test_mobile_entry_point() -> None:
    emb = resolve("embodiment", "g1_mobile", backend="kinematic", hand_speed="0.1")
    assert isinstance(emb, G1MobileEmbodiment)
    assert np.asarray(emb.info.action_space.high)[3] == pytest.approx(0.01)


# ---------------------------------------------------------------------- embodiment


def mobile_rig(**cfg: Any) -> tuple[Rig, KinematicMobileDriver]:
    config = G1MobileConfig(backend="kinematic", unattended=True, **cfg)
    rig = Rig(config, embodiment_cls=G1MobileEmbodiment)
    driver = KinematicMobileDriver(config, clock=rig.clock, response_time_s=0.0)
    rig.driver = driver  # type: ignore[assignment]
    rig.emb._driver_factory = lambda _: driver

    def sleep_while_streaming(duration: float) -> None:
        rig.emb.stream_once()
        rig.clock.sleep(duration)

    rig.emb._sleep = sleep_while_streaming
    rig.emb._wall_sleep = rig.clock.sleep
    return rig, driver


def test_arms_stay_put_until_reset_hands_them_over() -> None:
    rig, driver = mobile_rig()
    rig.emb._connect()
    before, _ = driver.read_hands()
    rig.emb.stream_once()
    np.testing.assert_allclose(driver.read_hands()[0], before)
    rig.emb.close()


def test_reset_glides_hands_home_and_steps_move_them() -> None:
    rig, driver = mobile_rig()
    obs = rig.reset()
    np.testing.assert_allclose(obs.state["hand_pos"].reshape(2, 3), HOME_HANDS, atol=2e-3)
    # Homing took about distance / hand speed.
    assert 10 <= len(rig.clock.sleeps) <= 30
    rig.clock.sleeps.clear()
    for _ in range(4):
        result = rig.emb.step(Action(data=np.asarray((0, 0, 0, 0.015, 0, 0.01, 0, 0, 0, 0.2, 0))))
    palms = result.observation.state["hand_pos"].reshape(2, 3)
    np.testing.assert_allclose(palms[0], HOME_HANDS[0] + (0.06, 0.0, 0.04), atol=2e-3)
    np.testing.assert_allclose(palms[1], HOME_HANDS[1], atol=2e-3)
    np.testing.assert_allclose(result.observation.state["grip"], (0.8, 0.0))
    rig.emb.close()
    np.testing.assert_allclose(driver.read_hands()[1], 0.0)


def test_workspace_box_and_reach_projection() -> None:
    rig, driver = mobile_rig()
    rig.reset()
    # Push the left palm far forward and down: the box stops x at 0.40, and the
    # low front corner is out of reach, so the target is pulled back to reach.
    for _ in range(40):
        rig.emb.step(Action(data=np.asarray((0, 0, 0, 0.015, 0, -0.015, 0, 0, 0, 0, 0))))
    target = rig.emb._hand_targets[0]
    palms, _ = driver.read_hands()
    assert target[0] <= 0.40 + 1e-9
    np.testing.assert_allclose(palms[0], target, atol=1e-9)
    # Pulling back responds immediately: no wound-up target beyond reach.
    before = palms[0].copy()
    rig.emb.step(Action(data=np.asarray((0, 0, 0, -0.015, 0, 0, 0, 0, 0, 0, 0))))
    after = driver.read_hands()[0][0]
    assert after[0] < before[0] - 0.01
    rig.emb.close()


def test_a_newer_step_is_not_overwritten_by_an_older_projection() -> None:
    rig, driver = mobile_rig()
    rig.reset()
    emb = rig.emb
    original = driver.send_hands

    def racy_send(targets: Any, grips: Any) -> Any:
        # A step lands while the backend is still solving the previous targets.
        reached = original(targets, grips)
        with emb._lock:
            emb._hand_targets = emb._hand_targets + 0.001
        return reached

    driver.send_hands = racy_send  # type: ignore[method-assign]
    moved = emb._hand_targets + 0.001
    emb.stream_once()
    np.testing.assert_allclose(emb._hand_targets, moved)
    emb.close()


def test_non_arm_backend_and_bad_hand_data_are_refused() -> None:
    config = G1MobileConfig(backend="kinematic", unattended=True)
    rig = Rig(config, embodiment_cls=G1MobileEmbodiment)
    with pytest.raises(TypeError, match="needs a backend that drives arms"):
        rig.reset()
    assert rig.driver.events[-1] == "disconnect"

    rig, driver = mobile_rig()
    rig.reset()
    driver.send_hands = lambda t, g: np.zeros((1, 3))  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="invalid reachable palms"):
        rig.emb.stream_once()
    driver.read_hands = lambda: (np.zeros((2, 3)), np.zeros(3))  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="wrong shape"):
        rig.emb._extra_state(driver)
    driver.read_hands = lambda: (np.full((2, 3), np.nan), np.zeros(2))  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="non-finite"):
        rig.emb._extra_state(driver)


def test_release_that_never_finishes_is_reported() -> None:
    rig, driver = mobile_rig()
    rig.reset()
    driver.release_hands = lambda: False  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="did not finish handing back"):
        rig.emb.close()
    # The backend was still disconnected.
    assert driver.connected is False
    # A second close without reset has nothing to release.
    rig.emb._release_extra(driver)


def test_a_takeover_that_never_finishes_is_reported() -> None:
    rig, driver = mobile_rig()
    driver.engage_hands = lambda: False  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="did not finish taking over"):
        rig.reset()


# ---------------------------------------------------------------------- real arm driver


class FakeArmPublisher:
    def __init__(self) -> None:
        self.sent: list[tuple[np.ndarray, float]] = []

    def publish(self, q14: Any, weight: float, kp: Any, kd: Any) -> None:
        assert kp.shape == kd.shape == (14,)
        self.sent.append((np.asarray(q14).copy(), weight))


class FakeLoco:
    def SetVelocity(self, vx: float, vy: float, omega: float, duration: float = 1.0) -> int:
        return 0

    def GetFsmId(self) -> tuple[int, Any]:
        return 0, 500


def lowstate_with_arms(q14: Any) -> Any:
    motors = [SimpleNamespace(q=0.0) for _ in range(35)]
    for slot, index in enumerate(ARM_MOTOR_INDICES):
        motors[index].q = float(q14[slot])
    return SimpleNamespace(imu_state=SimpleNamespace(rpy=[0.0, 0.0, 0.0]), motor_state=motors)


def test_sdk_arm_driver_holds_ramps_rate_limits_and_releases() -> None:
    cfg = G1MobileConfig(arm_weight_ramp_s=0.5, stream_hz=20.0)
    publisher = FakeArmPublisher()
    driver = G1MobileSdkDriver(cfg, loco=FakeLoco(), clock=FakeClock(), arm_publisher=publisher)
    with pytest.raises(RuntimeError, match="arm joint angles"):
        driver.read_hands()
    assert driver.release_hands() is True  # nothing commanded yet
    measured = np.concatenate([G1_ARMS[0].nominal, G1_ARMS[1].nominal])
    driver.on_lowstate(lowstate_with_arms(measured))
    palms, grips = driver.read_hands()
    np.testing.assert_allclose(palms[0], fk(G1_ARMS[0], G1_ARMS[0].nominal))
    np.testing.assert_allclose(grips, 0.0)
    # First command: hold where the arms are, weight starts ramping.
    reached = driver.send_hands(palms, np.ones(2))
    np.testing.assert_allclose(reached, palms, atol=1e-3)
    q0, w0 = publisher.sent[-1]
    np.testing.assert_allclose(q0, measured, atol=1e-3)
    assert w0 == pytest.approx(0.1)
    # A far target moves each joint at most arm_joint_step per tick.
    for _ in range(12):
        driver.send_hands(HOME_HANDS, np.zeros(2))
    steps = np.diff(np.stack([q for q, _ in publisher.sent]), axis=0)
    assert np.max(np.abs(steps)) <= cfg.arm_joint_step + 1e-12
    assert driver.weight == 1.0
    # Release ramps the weight back down while holding the last command.
    held = publisher.sent[-1][0]
    results = [driver.release_hands() for _ in range(10)]
    assert results[-1] is True and results[0] is False
    assert publisher.sent[-1][1] == 0.0
    np.testing.assert_allclose(publisher.sent[-1][0], held)
    assert driver.release_hands() is True


class FakeDex1:
    def __init__(self, strokes: Any = None) -> None:
        self.strokes = strokes
        self.sent: list[np.ndarray] = []

    def read_strokes(self) -> Any:
        if self.strokes is None:
            raise RuntimeError("Dex1 state topics have not produced samples yet")
        return self.strokes

    def publish_strokes(self, strokes: Any) -> None:
        self.sent.append(np.asarray(strokes).copy())


def test_dex1_grip_mapping_is_open_at_full_stroke() -> None:
    np.testing.assert_allclose(grip_to_dex1_stroke([0.0, 1.0, 2.0], 5.4), [5.4, 0.0, 0.0])
    np.testing.assert_allclose(dex1_stroke_to_grip([5.4, 2.7, -1.0], 5.4), [0.0, 0.5, 1.0])


def test_dex1_config_enables_grips_on_the_real_robot() -> None:
    cfg = G1MobileConfig(hands="dex1")
    assert cfg.backend == "sdk" and cfg.grips_enabled is True
    np.testing.assert_allclose(cfg.high[9:], cfg.grip_rate / cfg.control_hz)


def test_sdk_driver_drives_dex1_grippers_from_the_measured_stroke() -> None:
    cfg = G1MobileConfig(hands="dex1", stream_hz=20.0)
    publisher = FakeArmPublisher()
    grippers = FakeDex1()
    driver = G1MobileSdkDriver(
        cfg, loco=FakeLoco(), clock=FakeClock(), arm_publisher=publisher, grippers=grippers
    )
    measured = np.concatenate([G1_ARMS[0].nominal, G1_ARMS[1].nominal])
    driver.on_lowstate(lowstate_with_arms(measured))
    with pytest.raises(RuntimeError, match="Dex1 state"):
        driver.read_hands()
    grippers.strokes = [5.4, 2.7]  # left open, right half closed
    palms, grips = driver.read_hands()
    np.testing.assert_allclose(grips, [0.0, 0.5])
    # Nothing moves, grippers included, until the arms are taken over.
    while not driver.engage_hands():
        pass
    assert grippers.sent == []
    # Close both: strokes start where they are and move at most 2.7/20 rad per tick.
    for _ in range(3):
        driver.send_hands(palms, np.ones(2))
    step = cfg.dex1_max_speed / cfg.stream_hz
    np.testing.assert_allclose(grippers.sent[0], [5.4 - step, 2.7 - step])
    np.testing.assert_allclose(grippers.sent[-1], [5.4 - 3 * step, 2.7 - 3 * step])
    # Release ramps the arms out without touching the grippers.
    sent = len(grippers.sent)
    while not driver.release_hands():
        pass
    assert len(grippers.sent) == sent
    # A new hand-over starts again from the measured stroke.
    grippers.strokes = [1.0, 1.0]
    while not driver.engage_hands():
        pass
    driver.send_hands(palms, np.ones(2))
    np.testing.assert_allclose(grippers.sent[-1], [1.0 - step, 1.0 - step])
    grippers.strokes = [np.nan, 0.0]
    with pytest.raises(ValueError, match="invalid strokes"):
        driver.read_hands()


def test_tool_offset_moves_the_controlled_point_past_the_palm() -> None:
    cfg = G1MobileConfig(tool_offset_m=0.1)
    driver = G1MobileSdkDriver(
        cfg, loco=FakeLoco(), clock=FakeClock(), arm_publisher=FakeArmPublisher()
    )
    measured = np.concatenate([G1_ARMS[0].nominal, G1_ARMS[1].nominal])
    driver.on_lowstate(lowstate_with_arms(measured))
    palms, _ = driver.read_hands()
    plain = fk(G1_ARMS[0], G1_ARMS[0].nominal)
    assert np.linalg.norm(palms[0] - plain) == pytest.approx(0.1)
    kinematic = KinematicMobileDriver(
        G1MobileConfig(backend="kinematic", tool_offset_m=0.1), clock=FakeClock()
    )
    np.testing.assert_allclose(kinematic.read_hands()[0], palms)
    explicit = KinematicMobileDriver(
        G1MobileConfig(backend="kinematic", tool_offset_m=0.1), clock=FakeClock(), arms=G1_ARMS
    )
    np.testing.assert_allclose(explicit.read_hands()[0][0], plain)


# ---------------------------------------------------------------------- full eval


class WalkThenReach:
    """Walk forward for 10 steps, then raise the right hand for 10 steps."""

    def __init__(self, cfg: G1MobileConfig) -> None:
        self.info = PolicyInfo(
            name="walk-then-reach",
            action_space=cfg.make_action_box(),
            observation_space=cfg.make_observation_space(),
        )
        self.config = PolicyConfig(action_horizon=10)
        self.calls = 0

    def reset(self, scene: Scene) -> None:
        self.calls = 0

    def act(self, observation: Observation) -> ActionChunk:
        self.calls += 1
        if self.calls == 1:
            step = np.asarray((0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), dtype=np.float64)
        else:
            step = np.asarray((0, 0, 0, 0, 0, 0, 0.015, 0, 0.015, 0, 0.2), dtype=np.float64)
        return ActionChunk(actions=[Action(data=step)] * 10)


def test_eval_walks_then_reaches(tmp_path: Path) -> None:
    cfg = G1MobileConfig(backend="kinematic", unattended=True, camera_height=48, camera_width=64)
    clock = FakeClock()
    driver = KinematicMobileDriver(cfg, clock=clock, response_time_s=0.0)
    emb: G1MobileEmbodiment

    def sleep_while_streaming(duration: float) -> None:
        emb.stream_once()
        clock.sleep(duration)

    emb = G1MobileEmbodiment(
        cfg,
        driver_factory=lambda _: driver,
        clock=clock,
        sleep_fn=sleep_while_streaming,
        thread_factory=ManualThread,
        atexit_module=FakeAtexit(),
        signal_module=FakeSignal(),
    )
    task = Task(
        name="walk-reach",
        scenes=[Scene(id="w", instruction="walk, then reach")],
        scorer=episode_length(),
        max_steps=20,
    )
    logs = robots_eval(task, WalkThenReach(cfg), emb, sinks=[], seed=0, log_dir=str(tmp_path))
    assert logs[0].status == "success", logs[0].error
    np.testing.assert_allclose(driver.read_state().pose[:2], (0.3, 0.0), atol=1e-6)
    palms, grips = driver.read_hands()
    np.testing.assert_allclose(palms[1], HOME_HANDS[1] + (0.15, 0.0, 0.15), atol=3e-3)
    assert grips[1] == pytest.approx(1.0)
    emb.close()


def test_plain_g1_config_is_not_accepted() -> None:
    with pytest.raises(TypeError, match="G1MobileEmbodiment needs a G1MobileConfig"):
        G1MobileEmbodiment(G1Config())


def test_sim_paced_homing_latches_targets_before_physics() -> None:
    rig, _ = mobile_rig()
    rig.emb._sim_paced = True
    rig.emb._sleep = rig.clock.sleep
    obs = rig.reset()
    np.testing.assert_allclose(obs.state["hand_pos"].reshape(2, 3), HOME_HANDS, atol=2e-2)
    rig.emb.close()


def test_takeover_holds_the_measured_pose_and_reseeds_on_every_reset() -> None:
    cfg = G1MobileConfig(arm_weight_ramp_s=0.5, stream_hz=20.0)
    publisher = FakeArmPublisher()
    driver = G1MobileSdkDriver(cfg, loco=FakeLoco(), clock=FakeClock(), arm_publisher=publisher)
    first = np.concatenate([G1_ARMS[0].nominal, G1_ARMS[1].nominal])
    first[4] = 0.4  # a wrist away from the IK's nominal: it must not move during takeover
    driver.on_lowstate(lowstate_with_arms(first))
    far = HOME_HANDS + 0.1
    # send_hands before takeover holds: it engages and reports the held palms.
    held = driver.send_hands(far, np.zeros(2))
    np.testing.assert_allclose(held[0], fk(G1_ARMS[0], first[:7]))
    while not driver.engage_hands():
        pass
    assert driver.weight == 1.0
    for q, _ in publisher.sent:
        np.testing.assert_allclose(q, first)  # the measured pose, unchanged, every tick
    driver.send_hands(far, np.zeros(2))
    assert not np.allclose(publisher.sent[-1][0], first)  # moves only once engaged
    # A second reset without close: the baseline is reseeded from the new measured pose.
    second = first + 0.2
    driver.on_lowstate(lowstate_with_arms(second))
    assert driver.engage_hands() is True  # already engaged: reseed and go
    np.testing.assert_allclose(publisher.sent[-1][0], second)
