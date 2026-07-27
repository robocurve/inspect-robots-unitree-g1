from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from inspect_robots.scene import Scene
from inspect_robots.types import Action

from conftest import FakeArm, FakeAtexit, FakeClock, FakeHand, FakeSignal
from inspect_robots_unitree_g1 import embodiment as module
from inspect_robots_unitree_g1.config import G1Config
from inspect_robots_unitree_g1.embodiment import G1Embodiment
from inspect_robots_unitree_g1.operator import OperatorIO

SCENE = Scene(id="x", instruction="move")


def build(
    *,
    cfg: G1Config | None = None,
    arm: FakeArm | None = None,
    hand: FakeHand | None = None,
    operator: OperatorIO | None = None,
    poll_end: Any = None,
    atexit: FakeAtexit | None = None,
    signal: FakeSignal | None = None,
) -> tuple[G1Embodiment, FakeArm, FakeHand, FakeClock, FakeAtexit, FakeSignal]:
    actual_cfg = cfg or G1Config(unattended=True, weight_ramp_s=0.04, max_joint_speed=100.0)
    actual_arm = arm or FakeArm()
    actual_hand = hand or FakeHand()
    clock = FakeClock()
    exit_module = atexit or FakeAtexit()
    signal_module = signal or FakeSignal()
    embodiment = G1Embodiment(
        actual_cfg,
        arm_driver_factory=lambda _: actual_arm,
        hand_driver_factory=lambda _: actual_hand,
        camera_reader=lambda: np.zeros((2, 3, 3), dtype=np.uint8),
        operator=operator,
        poll_end=poll_end,
        clock=clock,
        sleep_fn=clock.sleep,
        atexit_module=exit_module,
        signal_module=signal_module,
    )
    return embodiment, actual_arm, actual_hand, clock, exit_module, signal_module


def test_init_is_inert_and_info() -> None:
    calls: list[str] = []
    embodiment = G1Embodiment(
        arm_driver_factory=lambda _: calls.append("arm") or FakeArm(),
        hand_driver_factory=lambda _: calls.append("hand") or FakeHand(),
    )
    assert calls == []
    assert embodiment.info.name == "g1_arms"
    assert embodiment.info.control_hz == 10.0
    assert embodiment.info.capabilities == {"self_paced"}
    embodiment.close()


def test_first_reset_exact_weight_order_seed_home_and_observation() -> None:
    observed = np.linspace(-0.2, 0.2, 14)
    embodiment, arm, _, clock, exit_module, signal_module = build(arm=FakeArm(observed))
    obs = embodiment.reset(SCENE)
    assert [item[1] for item in arm.publishes[:3]] == [0.0, 0.5, 1.0]
    for item in arm.publishes[:3]:
        np.testing.assert_allclose(item[0], observed)
    assert all(item[1] == 1.0 for item in arm.publishes[3:])
    np.testing.assert_allclose(arm.publishes[-1][0], np.zeros(14), atol=1e-12)
    assert arm.publishes[0][2].tolist() == [60.0] * 4 + [40.0] * 3 + [60.0] * 4 + [40.0] * 3
    assert arm.publishes[0][3].tolist() == [1.5] * 14
    assert obs.state["joint_pos"][7] == 0.25
    assert obs.state["joint_pos"][15] == 0.75
    np.testing.assert_array_equal(obs.state["left_leg"], np.arange(6))
    np.testing.assert_array_equal(obs.state["right_leg"], np.arange(6, 12))
    np.testing.assert_array_equal(obs.state["waist"], np.arange(12, 15))
    assert obs.images["head_cam"].shape == (2, 3, 3)
    assert obs.instruction == "move"
    assert all(value == pytest.approx(0.02) for value in clock.sleeps)
    assert len(exit_module.registered) == 1
    assert signal_module.handlers[-1][0] == signal_module.SIGTERM
    embodiment.close()


def test_later_reset_reseeds_without_weight_ramp() -> None:
    embodiment, arm, _, _, _, _ = build()
    embodiment.reset(SCENE)
    arm.publishes.clear()
    arm.joints = np.full(14, 0.4)
    embodiment.reset(SCENE)
    assert arm.publishes[0][1] == 1.0
    np.testing.assert_allclose(arm.publishes[0][0], 0.32)
    assert all(item[1] == 1.0 for item in arm.publishes)


def test_step_interpolation_delta_cap_clamp_and_pacing() -> None:
    cfg = G1Config(
        unattended=True,
        weight_ramp_s=0.02,
        max_joint_speed=0.5,
        home_pose=(0.0,) * 7 + (1.0,) + (0.0,) * 7 + (1.0,),
    )
    embodiment, arm, _, clock, _, _ = build(cfg=cfg)
    embodiment.reset(SCENE)
    arm.publishes.clear()
    clock.sleeps.clear()
    command = np.full(16, 99.0)
    result = embodiment.step(Action(data=command))
    assert len(arm.publishes) == 5
    expected = np.arange(1, 6) * (0.5 / 50.0)
    np.testing.assert_allclose([item[0][0] for item in arm.publishes], expected)
    assert all(item[1] == 1.0 for item in arm.publishes)
    assert all(value == pytest.approx(0.02) for value in clock.sleeps)
    assert result.terminated is False
    assert embodiment.num_steps == 1
    embodiment.close()


def test_step_is_anchored_at_last_published_command() -> None:
    embodiment, arm, _, _, _, _ = build()
    embodiment.reset(SCENE)
    arm.publishes.clear()
    first = np.zeros(16)
    first[0] = 0.5
    embodiment.step(Action(data=first))
    baseline = arm.publishes[-1][0][0]
    arm.publishes.clear()
    second = np.zeros(16)
    second[0] = -0.5
    embodiment.step(Action(data=second))
    assert arm.publishes[0][0][0] == pytest.approx(baseline + (-0.5 - baseline) / 5)
    embodiment.close()


def test_hand_deadband_dex3_and_dex1_rate_limit() -> None:
    embodiment, _, hand, _, _, _ = build()
    embodiment.reset(SCENE)
    hand.publishes.clear()
    command = np.zeros(16)
    command[7], command[15] = 0.98, 0.98
    embodiment.step(Action(data=command))
    assert hand.publishes == []
    command[7], command[15] = 0.9, 0.1
    embodiment.step(Action(data=command))
    assert hand.publishes == [(0.9, 0.1)]
    embodiment.close()

    dex1_cfg = G1Config(hand_type="dex1", unattended=True, weight_ramp_s=0.02, max_joint_speed=100)
    embodiment, _, hand, _, _, _ = build(cfg=dex1_cfg, hand=FakeHand(np.asarray([0.5, 0.5])))
    embodiment.reset(SCENE)
    hand.publishes.clear()
    command[7], command[15] = 1.0, 0.0
    embodiment.step(Action(data=command))
    assert len(hand.publishes) == 5
    assert hand.publishes[-1][1] == pytest.approx(0.9)
    embodiment.close()


def test_operator_success_failure_unattended_and_bind_task() -> None:
    answers = iter(["", "", "yes"])
    operator = OperatorIO(input_fn=lambda _: next(answers))
    embodiment, _, _, _, _, _ = build(
        cfg=G1Config(weight_ramp_s=0.02, max_joint_speed=100),
        operator=operator,
        poll_end=lambda: True,
    )
    embodiment.bind_task(SimpleNamespace(max_steps=12))
    embodiment.reset(SCENE)
    result = embodiment.step(Action(data=np.zeros(16)))
    assert result.terminated and result.termination_reason == "success"
    assert result.info == {"operator_confirmed": True}
    embodiment.close()

    answers = iter(["", "", "n"])
    embodiment, _, _, _, _, _ = build(
        cfg=G1Config(weight_ramp_s=0.02, max_joint_speed=100),
        operator=OperatorIO(input_fn=lambda _: next(answers)),
        poll_end=lambda: True,
    )
    embodiment.reset(SCENE)
    assert embodiment.step(Action(data=np.zeros(16))).termination_reason == "failure"
    embodiment.close()


def test_close_order_rest_weight_down_disconnect_and_idempotency() -> None:
    cfg = G1Config(
        unattended=True,
        weight_ramp_s=0.04,
        max_joint_speed=100,
        rest_pose=(0.1,) * 7 + (1.0,) + (0.2,) * 7 + (1.0,),
    )
    embodiment, arm, hand, _, exit_module, signal_module = build(cfg=cfg)
    embodiment.reset(SCENE)
    arm.publishes.clear()
    hand.publishes.clear()
    embodiment.close()
    assert hand.publishes[0] == (1.0, 1.0)
    assert [item[1] for item in arm.publishes[-3:]] == [1.0, 0.5, 0.0]
    np.testing.assert_allclose(arm.publishes[-1][0], [0.1] * 7 + [0.2] * 7)
    assert hand.events[-1] == "hand_disconnect"
    assert arm.events[-1] == "arm_disconnect"
    assert len(exit_module.unregistered) == 1
    assert signal_module.handlers[-1][1] is None
    count = len(arm.events)
    embodiment.close()
    assert len(arm.events) == count


def test_close_uses_observed_pose_and_disconnects_on_publish_error() -> None:
    embodiment, arm, hand, _, _, _ = build()
    embodiment.reset(SCENE)
    arm.joints = np.full(14, 0.3)
    embodiment.close()
    np.testing.assert_allclose(arm.publishes[-1][0], 0.3)

    embodiment, arm, hand, _, _, _ = build()
    embodiment.reset(SCENE)

    def fail(left: float, right: float) -> None:
        raise RuntimeError("hand fail")

    hand.publish_closure = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="hand fail"):
        embodiment.close()
    assert arm.events[-1] == "arm_disconnect"
    assert hand.events[-1] == "hand_disconnect"
    embodiment.close()


def test_disconnect_errors_preserve_first_and_clear_handles() -> None:
    arm, hand = FakeArm(), FakeHand()
    arm.disconnect_error = ValueError("arm close")
    hand.disconnect_error = RuntimeError("hand close")
    embodiment, _, _, _, _, _ = build(arm=arm, hand=hand)
    embodiment.reset(SCENE)
    with pytest.raises(RuntimeError, match="hand close"):
        embodiment.close()
    embodiment.close()


def test_sigterm_chains_previous_and_backstop_registration_failure() -> None:
    chained: list[int] = []
    signals = FakeSignal(previous=lambda signum, frame: chained.append(signum))
    embodiment, _, _, _, _, signal_module = build(signal=signals)
    embodiment.reset(SCENE)
    handler = signal_module.handlers[0][1]
    handler(signal_module.SIGTERM, None)
    assert chained == [15]

    class BrokenSignal(FakeSignal):
        def signal(self, signum: int, handler: Any) -> None:
            raise RuntimeError("signal fail")

    exit_module = FakeAtexit()
    embodiment, arm, hand, _, _, _ = build(atexit=exit_module, signal=BrokenSignal())
    with pytest.raises(RuntimeError, match="signal fail"):
        embodiment.reset(SCENE)
    assert exit_module.unregistered
    assert arm.events[-1] == "arm_disconnect"
    assert hand.events[-1] == "hand_disconnect"


def test_connect_hand_failure_disconnects_arm() -> None:
    arm = FakeArm()
    embodiment = G1Embodiment(
        G1Config(unattended=True),
        arm_driver_factory=lambda _: arm,
        hand_driver_factory=lambda _: (_ for _ in ()).throw(RuntimeError("no hand")),
        camera_reader=lambda: np.zeros((1, 1, 3), dtype=np.uint8),
    )
    with pytest.raises(RuntimeError, match="no hand"):
        embodiment.reset(SCENE)
    assert arm.events == ["arm_disconnect"]


@pytest.mark.parametrize(
    ("arm_values", "hand_values", "message"),
    [
        (np.zeros(13), np.zeros(2), "arm driver"),
        (np.zeros(14), np.zeros(3), "hand driver"),
        (np.full(14, np.nan), np.zeros(2), "arm driver"),
    ],
)
def test_driver_read_validation(
    arm_values: np.ndarray, hand_values: np.ndarray, message: str
) -> None:
    embodiment, _, _, _, _, _ = build(arm=FakeArm(arm_values), hand=FakeHand(hand_values))
    with pytest.raises(ValueError, match=message):
        embodiment.reset(SCENE)


def test_state_validation_camera_validation_and_step_before_reset() -> None:
    embodiment = G1Embodiment(camera_reader=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="callable"):
        embodiment.reset(SCENE)
    with pytest.raises(RuntimeError, match="before reset"):
        embodiment.step(Action(data=np.zeros(16)))

    arm = FakeArm()
    arm.body = {"left_leg": np.zeros(5), "right_leg": np.zeros(6), "waist": np.zeros(3)}
    embodiment, _, _, _, _, _ = build(arm=arm)
    with pytest.raises(ValueError, match="left_leg"):
        embodiment.reset(SCENE)
    embodiment.close()


def test_builtin_camera_factory_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        module,
        "_zmq_camera_reader",
        lambda _: lambda: np.zeros((1, 2, 3), dtype=np.uint8),
    )
    arm, hand, clock = FakeArm(), FakeHand(), FakeClock()
    embodiment = G1Embodiment(
        G1Config(unattended=True, weight_ramp_s=0.02, max_joint_speed=100),
        arm_driver_factory=lambda _: arm,
        hand_driver_factory=lambda _: hand,
        clock=clock,
        sleep_fn=clock.sleep,
        atexit_module=FakeAtexit(),
        signal_module=FakeSignal(),
    )
    assert embodiment.reset(SCENE).images["head_cam"].shape == (1, 2, 3)
    embodiment.close()


def test_internal_guard_branches_and_dex3_pose() -> None:
    cfg = G1Config(dex3_thumb_swing=0.2)
    left, right = module._dex3_closed_poses(cfg)
    assert left[0] == right[0] == 0.2

    embodiment, arm, hand, _, _, _ = build()
    embodiment._register_backstops()
    embodiment._register_backstops()
    embodiment._unregister_backstops()
    embodiment._unregister_backstops()
    embodiment._arm, embodiment._hand = arm, hand
    with pytest.raises(RuntimeError, match="baseline"):
        embodiment._stream_arm(np.zeros(14), weight=1.0)
    with pytest.raises(RuntimeError, match="baseline"):
        embodiment._ramp_arms_to(np.zeros(14), weight=1.0)
    with pytest.raises(RuntimeError, match="hand baseline"):
        embodiment._publish_hands_if_changed(np.zeros(2))
    with pytest.raises(RuntimeError, match="hand baseline"):
        embodiment._move_hands_to(np.zeros(2))
    embodiment._last_arm = np.zeros(14)
    embodiment._stream_arm(np.zeros(14), weight=1.0)
    embodiment._arm, embodiment._hand = arm, None
    embodiment.close()
    assert arm.events[-1] == "arm_disconnect"


def test_sigterm_without_callable_previous_and_arm_publish_failure() -> None:
    embodiment, arm, _, _, _, signal_module = build(signal=FakeSignal(previous=0))
    embodiment.reset(SCENE)
    signal_module.handlers[0][1](15, None)

    embodiment, arm, _, _, _, _ = build()
    embodiment.reset(SCENE)

    def fail(*args: Any) -> None:
        raise RuntimeError("arm publish")

    arm.publish_arm = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="arm publish"):
        embodiment.close()
    assert arm.events[-1] == "arm_disconnect"


def test_close_read_failure_still_attempts_hand_and_weight_release() -> None:
    embodiment, arm, hand, _, _, _ = build()
    embodiment.reset(SCENE)
    arm.publishes.clear()

    def fail_read() -> np.ndarray:
        raise RuntimeError("read fail")

    arm.read_arm_joints = fail_read  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="read fail"):
        embodiment.close()
    assert hand.publishes[-1] == (1.0, 1.0)
    assert [item[1] for item in arm.publishes[-3:]] == [1.0, 0.5, 0.0]


def test_close_reports_weight_release_failure_after_successful_park() -> None:
    embodiment, arm, _, _, _, _ = build()
    embodiment.reset(SCENE)
    original = arm.publish_arm
    calls = 0

    def fail_during_weight(*args: Any) -> None:
        nonlocal calls
        calls += 1
        if calls > 5:
            raise RuntimeError("weight fail")
        original(*args)

    arm.publish_arm = fail_during_weight  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="weight fail"):
        embodiment.close()
    assert arm.events[-1] == "arm_disconnect"


def test_reset_observes_arm_pose_after_stand_clear_wait() -> None:
    arm = FakeArm()

    def input_fn(_prompt: str = "") -> str:
        arm.joints = np.full(14, 0.3)
        return ""

    embodiment, _, _, _, _, _ = build(
        cfg=G1Config(weight_ramp_s=0.02, max_joint_speed=100),
        arm=arm,
        operator=OperatorIO(input_fn=input_fn, output_fn=lambda _msg: None),
        poll_end=lambda: False,
    )
    embodiment.reset(SCENE)
    first_publish = arm.publishes[0][0]
    assert first_publish == pytest.approx(np.full(14, 0.3))


def test_step_rejects_non_finite_actions() -> None:
    embodiment, arm, _, _, _, _ = build()
    embodiment.reset(SCENE)
    published_before = len(arm.publishes)
    bad = np.zeros(16)
    bad[2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        embodiment.step(Action(data=bad))
    assert len(arm.publishes) == published_before


def test_observe_requires_complete_body_state() -> None:
    arm = FakeArm()
    arm.body = {"left_leg": np.zeros(6), "right_leg": np.zeros(6)}
    embodiment, _, _, _, _, _ = build(arm=arm)
    with pytest.raises(ValueError, match="missing 'waist'"):
        embodiment.reset(SCENE)


def test_close_preserves_first_error_when_park_ramp_also_fails() -> None:
    embodiment, arm, hand, _, _, _ = build()
    embodiment.reset(SCENE)
    arm.joints = np.full(14, 0.3)

    def hand_fail(left: float, right: float) -> None:
        raise RuntimeError("hand fail")

    def arm_fail(q14: Any, weight: float, kp: Any, kd: Any) -> None:
        raise RuntimeError("arm fail")

    hand.publish_closure = hand_fail  # type: ignore[method-assign]
    arm.publish_arm = arm_fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="hand fail"):
        embodiment.close()
    assert arm.events[-1] == "arm_disconnect"
    assert hand.events[-1] == "hand_disconnect"


def test_step_observation_failure_triggers_close() -> None:
    def failing_camera() -> np.ndarray:
        raise RuntimeError("camera failure")

    embodiment, arm, _, _, _, _ = build()
    embodiment.reset(SCENE)
    embodiment._camera_reader = failing_camera
    arm.events.clear()

    with pytest.raises(RuntimeError, match="camera failure"):
        embodiment.step(Action(data=np.zeros(16)))

    assert "arm_disconnect" in arm.events


def test_reset_observation_failure_triggers_close() -> None:
    def failing_camera() -> np.ndarray:
        raise RuntimeError("camera failure")

    embodiment, arm, _, _, _, _ = build()
    embodiment._camera_reader = failing_camera
    arm.events.clear()

    with pytest.raises(RuntimeError, match="camera failure"):
        embodiment.reset(SCENE)

    assert "arm_disconnect" in arm.events


def test_step_stream_failure_triggers_close() -> None:
    embodiment, arm, _, _, _, _ = build()
    embodiment.reset(SCENE)

    def failing_publish(*args: Any) -> None:
        raise RuntimeError("stream failure")

    arm.publish_arm = failing_publish  # type: ignore[method-assign]
    arm.events.clear()

    with pytest.raises(RuntimeError, match="stream failure"):
        embodiment.step(Action(data=np.zeros(16)))

    assert "arm_disconnect" in arm.events
